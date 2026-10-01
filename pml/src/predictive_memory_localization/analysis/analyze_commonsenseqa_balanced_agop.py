from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.analysis.analyze_editing_grouped_prediction import OUTCOMES, to_float, write_csv
from predictive_memory_localization.analysis.analyze_rome500_controlled_incremental import cols_matching, numeric_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Balanced CommonsenseQA AGOP validation for ROME editing. "
            "Uses existing error rows and repeatedly balances positives/negatives."
        )
    )
    parser.add_argument("--error-rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outcome", default="rewrite_margin_delta", choices=OUTCOMES)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=67)
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def add_length_features(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        copied = dict(row)
        length = int(float(row.get("target_new_words") or 0))
        copied["length_is_1"] = 1.0 if length == 1 else 0.0
        copied["length_is_2"] = 1.0 if length == 2 else 0.0
        copied["length_is_ge3"] = 1.0 if length >= 3 else 0.0
        copied["length_bin"] = "1" if length == 1 else "2" if length == 2 else "ge3"
        out.append(copied)
    return out


def feature_sets(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    length = [
        col
        for col in [
            "target_new_words",
            "target_new_chars",
            "pre_rewrite_num_tokens",
            "length_is_1",
            "length_is_2",
            "length_is_ge3",
        ]
        if numeric_available(rows, col)
    ]
    difficulty = [
        col
        for col in [
            "pre_rewrite_logprob",
            "pre_rewrite_mean_logprob",
            "pre_rewrite_num_tokens",
            "pre_old_logprob",
            "pre_old_mean_logprob",
            "pre_mean_rephrase_new_logprob",
            "target_new_words",
            "target_new_chars",
            "prompt_words",
            "prompt_chars",
        ]
        if numeric_available(rows, col)
    ]
    agop_compact = cols_matching(
        rows,
        prefix="agop_feature_",
        includes=["effective_rank", "spectral_entropy", "top1_ratio", "topk_ratio", "trace", "condition_number"],
        excludes=["proj_layer"],
    )
    agop_spectrum = cols_matching(
        rows,
        prefix="agop_feature_agop_layer_",
        includes=[
            "condition_number",
            "effective_rank",
            "eigengap",
            "fro_norm",
            "spectral_entropy",
            "top1_ratio",
            "topk_ratio",
            "trace",
        ],
    )
    projection_last = cols_matching(rows, prefix="agop_feature_proj_layer_-1_")
    baseline = cols_matching(rows, prefix="baseline_feature_", excludes=["top_saliency_layer"])
    sets = {
        "length": length,
        "difficulty": difficulty,
        "agop_compact": agop_compact,
        "agop_spectrum": agop_spectrum,
        "projection_last": projection_last,
        "baseline": baseline,
        "length_plus_agop_compact": length + agop_compact,
        "difficulty_plus_agop_compact": difficulty + agop_compact,
        "difficulty_plus_agop_spectrum": difficulty + agop_spectrum,
        "difficulty_plus_projection_last": difficulty + projection_last,
        "difficulty_plus_baseline": difficulty + baseline,
    }
    return {name: cols for name, cols in sets.items() if cols}


def build_xy(rows: List[Dict[str, Any]], features: Sequence[str], outcome: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x_rows = []
    y_rows = []
    ids = []
    for row in rows:
        label_value = to_float(row.get(outcome))
        if label_value is None:
            continue
        values = [to_float(row.get(feature)) for feature in features]
        if not all(value is not None for value in values):
            continue
        x_rows.append([float(value) for value in values])
        y_rows.append(1 if label_value >= 0.5 else 0)
        ids.append(str(row.get("id", "")))
    return np.array(x_rows, dtype=float), np.array(y_rows, dtype=int), ids


def auc_ap(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(np.unique(y_true)) < 2:
        return math.nan, math.nan
    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))


def balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n = min(len(pos), len(neg))
    if n < 8:
        return np.array([], dtype=int)
    return np.concatenate(
        [
            rng.choice(pos, n, replace=False),
            rng.choice(neg, n, replace=False),
        ]
    )


def evaluate_repeat(
    rows: List[Dict[str, Any]],
    *,
    outcome: str,
    feature_set_map: Dict[str, List[str]],
    seed: int,
    cv_folds: int,
) -> List[Dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    out = []
    for feature_set, features in feature_set_map.items():
        X, y, _ = build_xy(rows, features, outcome)
        idx = balanced_indices(y, rng)
        if len(idx) == 0:
            out.append(
                {
                    "feature_set": feature_set,
                    "available": False,
                    "reason": "not enough positives/negatives after balancing",
                }
            )
            continue
        Xb = X[idx]
        yb = y[idx]
        n_splits = min(cv_folds, int(np.bincount(yb).min()))
        if n_splits < 2:
            out.append(
                {
                    "feature_set": feature_set,
                    "available": False,
                    "reason": "not enough positives/negatives for CV",
                }
            )
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_predict(model, Xb, yb, cv=cv, method="predict_proba")[:, 1]
        auroc, ap = auc_ap(yb, scores)
        out.append(
            {
                "feature_set": feature_set,
                "available": True,
                "n_balanced": int(len(yb)),
                "n_folds": int(n_splits),
                "n_features": int(Xb.shape[1]),
                "auroc": auroc,
                "average_precision": ap,
            }
        )
    return out


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for feature_set in sorted({row["feature_set"] for row in rows}):
        vals = [row for row in rows if row["feature_set"] == feature_set and row.get("available")]
        if not vals:
            out.append({"feature_set": feature_set, "available": False})
            continue
        for metric in ["auroc", "average_precision"]:
            arr = np.array([float(row[metric]) for row in vals if row.get(metric) not in ("", None)], dtype=float)
            if arr.size == 0:
                continue
            out.append(
                {
                    "feature_set": feature_set,
                    "metric": metric,
                    "available": True,
                    "n_repeats": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "ci_low": float(np.quantile(arr, 0.025)),
                    "ci_high": float(np.quantile(arr, 0.975)),
                }
            )
    return out


def delta_summary(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {
        (row.get("feature_set"), row.get("metric")): row
        for row in summary_rows
        if row.get("available") and row.get("mean") not in ("", None)
    }
    comparisons = [
        ("length_plus_agop_compact", "length", "agop_compact_over_length"),
        ("difficulty_plus_agop_compact", "difficulty", "agop_compact_over_difficulty"),
        ("difficulty_plus_agop_spectrum", "difficulty", "agop_spectrum_over_difficulty"),
        ("difficulty_plus_projection_last", "difficulty", "projection_last_over_difficulty"),
        ("difficulty_plus_baseline", "difficulty", "baseline_over_difficulty"),
    ]
    out = []
    for plus, base, label in comparisons:
        for metric in ["auroc", "average_precision"]:
            plus_row = by.get((plus, metric))
            base_row = by.get((base, metric))
            if not plus_row or not base_row:
                continue
            out.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "base_feature_set": base,
                    "plus_feature_set": plus,
                    "base_mean": base_row["mean"],
                    "plus_mean": plus_row["mean"],
                    "delta_mean": float(plus_row["mean"]) - float(base_row["mean"]),
                }
            )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in add_length_features(read_csv(args.error_rows_csv))
        if row.get("source_dataset") == "tau/commonsense_qa"
    ]
    sets = feature_sets(rows)
    repeat_rows: List[Dict[str, Any]] = []
    for i in range(args.n_repeats):
        repeat = evaluate_repeat(
            rows,
            outcome=args.outcome,
            feature_set_map=sets,
            seed=args.seed + i,
            cv_folds=args.cv_folds,
        )
        for row in repeat:
            row.update({"repeat": i, "outcome": args.outcome})
        repeat_rows.extend(repeat)

    summary_rows = summarize(repeat_rows)
    deltas = delta_summary(summary_rows)
    write_csv(out_dir / "balanced_repeat_rows.csv", repeat_rows)
    write_csv(out_dir / "balanced_summary.csv", summary_rows)
    write_csv(out_dir / "balanced_deltas.csv", deltas)
    payload = {
        "n_rows": len(rows),
        "outcome": args.outcome,
        "n_repeats": args.n_repeats,
        "cv_folds": args.cv_folds,
        "summary": summary_rows,
        "deltas": deltas,
    }
    (out_dir / "balanced_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "n_rows": len(rows),
                "n_repeats": args.n_repeats,
                "n_feature_sets": len(sets),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
