from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.analysis.analyze_suppression_prediction import average_precision, auroc
from predictive_memory_localization.common import load_jsonl


OUTCOMES = [
    "efficacy",
    "rewrite_margin_delta",
    "rephrase_margin_delta",
    "locality",
    "locality_logprob_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grouped/source-wise prediction analysis for editing outcomes."
    )
    parser.add_argument("--editing-results-jsonl", required=True)
    parser.add_argument("--baseline-joined-csv", required=True)
    parser.add_argument("--agop-joined-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def source_by_id(results_jsonl: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in load_jsonl(results_jsonl):
        task = row.get("task", {})
        source = task.get("source", {}) if isinstance(task, dict) else {}
        dataset = source.get("dataset") or "unknown"
        row_id = row.get("id")
        if row_id:
            out[str(row_id)] = str(dataset)
    return out


def merge_rows(
    baseline_rows: List[Dict[str, Any]],
    agop_rows: List[Dict[str, Any]],
    sources: Dict[str, str],
) -> List[Dict[str, Any]]:
    baseline_by_id = {str(row.get("id")): row for row in baseline_rows if row.get("id")}
    merged = []
    for agop in agop_rows:
        row_id = str(agop.get("id", ""))
        if not row_id:
            continue
        base = baseline_by_id.get(row_id, {})
        row: Dict[str, Any] = {
            "id": row_id,
            "source_dataset": sources.get(row_id, "unknown"),
        }
        for outcome in OUTCOMES:
            row[outcome] = agop.get(outcome, base.get(outcome, ""))
        for key, value in base.items():
            if key.startswith("feature_"):
                row[f"baseline_{key}"] = value
        for key, value in agop.items():
            if key.startswith("feature_"):
                row[f"agop_{key}"] = value
        merged.append(row)
    return merged


def numeric_columns(rows: List[Dict[str, Any]], prefix: str) -> List[str]:
    cols = sorted({key for row in rows for key in row if key.startswith(prefix)})
    out = []
    for col in cols:
        values = [to_float(row.get(col)) for row in rows]
        values = [value for value in values if value is not None]
        if len(values) >= 10 and len(set(values)) > 1:
            out.append(col)
    return out


def feature_sets(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    baseline = [
        col
        for col in numeric_columns(rows, "baseline_feature_")
        if not col.endswith("top_saliency_layer")
    ]
    agop_all = [
        col
        for col in numeric_columns(rows, "agop_feature_")
        if not col.endswith("agop_layers") and not col.endswith("top_saliency_layer")
    ]
    agop_layer = [col for col in agop_all if "agop_layer_neg" in col]
    agop_aggregate = [col for col in agop_all if "agop_layer_neg" not in col]
    projection_layer = [col for col in agop_all if "proj_layer_" in col]
    agop_spectral = [col for col in agop_all if "agop_" in col and "proj_layer_" not in col]
    projection_last = [col for col in projection_layer if "proj_layer_-1_" in col]
    projection_compact = [
        col
        for col in projection_layer
        if any(metric in col for metric in ["neg_mean", "pos_mean", "neg_accuracy", "mean_gap", "cohen_d"])
    ]
    agop_neg5_spectrum = [
        col
        for col in agop_layer
        if "agop_layer_neg5_" in col
        and any(metric in col for metric in ["condition_number", "fro_norm", "trace", "eigengap", "effective_rank", "topk_ratio", "top1_ratio"])
    ]
    agop_neg13_topvec = [
        col
        for col in agop_layer
        if "agop_layer_neg13_" in col
        and any(metric in col for metric in ["top_vec", "top1_ratio", "topk_ratio"])
    ]
    agop_compact_spectrum = [
        col
        for col in agop_spectral
        if any(metric in col for metric in ["effective_rank", "spectral_entropy", "top1_ratio", "topk_ratio", "trace", "condition_number"])
    ]
    return {
        "baseline_aggregate": baseline,
        "projection_layer": projection_layer,
        "projection_last": projection_last,
        "projection_compact": projection_compact,
        "agop_aggregate": agop_aggregate,
        "agop_layer": agop_layer,
        "agop_all": agop_spectral,
        "agop_neg5_spectrum": agop_neg5_spectrum,
        "agop_neg13_topvec": agop_neg13_topvec,
        "agop_compact_spectrum": agop_compact_spectrum,
        "projection_plus_agop_layer": projection_layer + agop_layer,
        "projection_plus_agop_all": projection_layer + agop_spectral,
        "projection_last_plus_agop_neg5": projection_last + agop_neg5_spectrum,
        "projection_compact_plus_agop_compact": projection_compact + agop_compact_spectrum,
    }


def remap(rows: List[Dict[str, Any]], features: Sequence[str], label_key: str):
    X = []
    y = []
    groups = []
    kept_rows = []
    for row in rows:
        label = to_float(row.get(label_key))
        if label is None:
            continue
        values = [to_float(row.get(feature)) for feature in features]
        if not all(value is not None for value in values):
            continue
        X.append([float(value) for value in values])
        y.append(1 if label >= 0.5 else 0)
        groups.append(row["id"])
        kept_rows.append(row)
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups), kept_rows


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, metric_name: str, n_iters: int, seed: int):
    if n_iters <= 0 or len(labels) == 0:
        return {}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_iters):
        idx = rng.integers(0, len(labels), len(labels))
        sample_y = labels[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_scores = scores[idx]
        if metric_name == "auroc":
            values.append(auroc(sample_scores, sample_y))
        elif metric_name == "ap":
            values.append(average_precision(sample_scores, sample_y))
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        f"{metric_name}_bootstrap_mean": float(np.mean(arr)),
        f"{metric_name}_ci_low": float(np.quantile(arr, 0.025)),
        f"{metric_name}_ci_high": float(np.quantile(arr, 0.975)),
    }


def grouped_logistic_report(
    rows: List[Dict[str, Any]],
    features: Sequence[str],
    label_key: str,
    *,
    bootstrap_iters: int,
    seed: int,
) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False, "reason": "scikit-learn is not installed"}

    X, y, groups, _ = remap(rows, features, label_key)
    if X.shape[0] < 20 or X.shape[1] == 0:
        return {"available": False, "reason": "not enough complete rows/features"}
    if len(np.unique(y)) < 2:
        return {"available": False, "reason": "only one class"}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups), int(np.bincount(y).min()))
    if n_splits < 2:
        return {"available": False, "reason": "not enough folds/classes"}
    cv = GroupKFold(n_splits=n_splits)
    if len(unique_groups) == len(groups):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_kwargs = {}
    else:
        cv_kwargs = {"groups": groups}
    try:
        probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba", **cv_kwargs)[:, 1]
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}
    out = {
        "available": True,
        "n": int(len(y)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "cv_splits": int(n_splits),
        "cv_auroc": auroc(probs, y),
        "cv_average_precision": average_precision(probs, y),
    }
    out.update(bootstrap_ci(probs, y, "auroc", bootstrap_iters, seed))
    out.update(bootstrap_ci(probs, y, "ap", bootstrap_iters, seed + 1))
    return out


def split_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    splits = {"all": rows}
    for row in rows:
        splits.setdefault(row.get("source_dataset", "unknown"), []).append(row)
    return splits


def outcome_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"n": len(rows), "sources": {}}
    for source, source_rows in split_rows(rows).items():
        if source == "all":
            continue
        out["sources"][source] = len(source_rows)
    for outcome in OUTCOMES:
        vals = [to_float(row.get(outcome)) for row in rows]
        vals = [value for value in vals if value is not None]
        if not vals:
            continue
        out[outcome] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "positive_rate_threshold_0_5": float(np.mean(np.array(vals) >= 0.5)),
        }
    return out


def flatten(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for split, split_report in summary["splits"].items():
        for feature_set, outcome_reports in split_report["feature_sets"].items():
            for outcome, report in outcome_reports.items():
                if not report.get("available"):
                    rows.append(
                        {
                            "split": split,
                            "feature_set": feature_set,
                            "outcome": outcome,
                            "available": False,
                            "reason": report.get("reason", ""),
                        }
                    )
                    continue
                rows.append(
                    {
                        "split": split,
                        "feature_set": feature_set,
                        "outcome": outcome,
                        "available": True,
                        "n": report["n"],
                        "n_features": report["n_features"],
                        "positive_rate": report["positive_rate"],
                        "cv_auroc": report["cv_auroc"],
                        "auroc_ci_low": report.get("auroc_ci_low", ""),
                        "auroc_ci_high": report.get("auroc_ci_high", ""),
                        "cv_average_precision": report["cv_average_precision"],
                        "ap_ci_low": report.get("ap_ci_low", ""),
                        "ap_ci_high": report.get("ap_ci_high", ""),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = merge_rows(
        read_csv(args.baseline_joined_csv),
        read_csv(args.agop_joined_csv),
        source_by_id(args.editing_results_jsonl),
    )
    sets = feature_sets(rows)
    summary: Dict[str, Any] = {
        "editing_results_jsonl": args.editing_results_jsonl,
        "n_rows": len(rows),
        "outcome_summary": outcome_summary(rows),
        "feature_set_sizes": {name: len(cols) for name, cols in sets.items()},
        "splits": {},
    }
    for split_name, split in split_rows(rows).items():
        summary["splits"][split_name] = {
            "n": len(split),
            "feature_sets": {
                set_name: {
                    outcome: grouped_logistic_report(
                        split,
                        cols,
                        outcome,
                        bootstrap_iters=args.bootstrap_iters,
                        seed=args.seed,
                    )
                    for outcome in OUTCOMES
                }
                for set_name, cols in sets.items()
            },
        }
    summary_path = out_dir / "editing_grouped_prediction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "editing_grouped_prediction_table.csv", flatten(summary))
    write_csv(out_dir / "editing_grouped_prediction_rows.csv", rows)
    print(json.dumps({"summary": str(summary_path), "n_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
