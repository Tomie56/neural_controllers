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

from predictive_memory_localization.analysis.analyze_editing_grouped_prediction import (
    average_precision,
    auroc,
    grouped_logistic_report,
    to_float,
    write_csv,
)
from predictive_memory_localization.analysis.analyze_rome500_error_ablation import feature_ablation_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze fragile/superficial edit signatures from existing ROME/MEMIT result rows."
    )
    parser.add_argument("--error-rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-post-margin", type=float, default=2.0)
    parser.add_argument("--min-rephrase-delta", type=float, default=0.0)
    parser.add_argument("--max-locality-drop", type=float, default=-1.0)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_true(value: Any) -> bool:
    val = to_float(value)
    return bool(val is not None and val >= 0.5)


def safe_float(value: Any, default: float = 0.0) -> float:
    val = to_float(value)
    return default if val is None else float(val)


def numeric_available(rows: List[Dict[str, Any]], col: str, min_rows: int = 20) -> bool:
    vals = [to_float(row.get(col)) for row in rows]
    vals = [val for val in vals if val is not None]
    return len(vals) >= min_rows and len(set(vals)) > 1


def diagnostic_features(rows: List[Dict[str, Any]]) -> List[str]:
    candidates = [
        "pre_rewrite_logprob",
        "pre_rewrite_mean_logprob",
        "pre_rewrite_num_tokens",
        "pre_old_logprob",
        "pre_old_mean_logprob",
        "pre_old_num_tokens",
        "pre_mean_rephrase_new_logprob",
        "pre_mean_locality_logprob",
        "target_new_words",
        "target_new_chars",
        "ground_truth_words",
        "ground_truth_chars",
        "prompt_words",
        "prompt_chars",
        "subject_words",
        "subject_chars",
        "subject_prompt_char_ratio",
        "num_paraphrase_prompts",
        "num_locality_prompts",
    ]
    return [col for col in candidates if numeric_available(rows, col)]


def label_rows(
    rows: List[Dict[str, Any]],
    *,
    min_post_margin: float,
    min_rephrase_delta: float,
    max_locality_drop: float,
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        post_rewrite = to_float(item.get("post_rewrite_logprob"))
        post_old = to_float(item.get("post_old_logprob"))
        pre_rewrite = to_float(item.get("pre_rewrite_logprob"))
        pre_old = to_float(item.get("pre_old_logprob"))
        post_margin = None if post_rewrite is None or post_old is None else post_rewrite - post_old
        pre_margin = None if pre_rewrite is None or pre_old is None else pre_rewrite - pre_old
        margin_gain = None if post_margin is None or pre_margin is None else post_margin - pre_margin
        rephrase_delta = to_float(item.get("rephrase_margin_delta"))
        locality_delta = to_float(item.get("locality_logprob_delta"))
        standard_success = is_true(item.get("efficacy"))
        weak_post_margin = bool(post_margin is not None and post_margin < min_post_margin)
        weak_rephrase = bool(rephrase_delta is not None and rephrase_delta < min_rephrase_delta)
        locality_damage = bool(locality_delta is not None and locality_delta < max_locality_drop)
        fragile_success = bool(standard_success and (weak_post_margin or weak_rephrase or locality_damage))
        robust_success = bool(
            standard_success
            and not weak_post_margin
            and not weak_rephrase
            and not locality_damage
        )
        item.update(
            {
                "post_margin": post_margin if post_margin is not None else "",
                "pre_margin": pre_margin if pre_margin is not None else "",
                "margin_gain": margin_gain if margin_gain is not None else "",
                "standard_success_bool": standard_success,
                "weak_post_margin_bool": weak_post_margin,
                "weak_rephrase_bool": weak_rephrase,
                "locality_damage_bool": locality_damage,
                "fragile_success_bool": fragile_success,
                "robust_success_bool": robust_success,
                "superficial_risk_score": float(weak_post_margin) + float(weak_rephrase) + float(locality_damage),
            }
        )
        out.append(item)
    return out


def split_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    splits = {"all": rows}
    for row in rows:
        splits.setdefault(str(row.get("source_dataset", "unknown")), []).append(row)
    return splits


def analysis_splits(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    splits = split_rows(rows)
    out = dict(splits)
    for split_name, split_subset in splits.items():
        success_subset = [row for row in split_subset if row.get("standard_success_bool")]
        if success_subset:
            out[f"{split_name}:success_only"] = success_subset
    return out


def label_rate(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(row.get(key)) for row in rows]))


def outcome_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n": len(rows),
        "standard_success_rate": label_rate(rows, "standard_success_bool"),
        "fragile_success_rate_all": label_rate(rows, "fragile_success_bool"),
        "robust_success_rate_all": label_rate(rows, "robust_success_bool"),
    }
    successes = [row for row in rows if row.get("standard_success_bool")]
    out["n_standard_success"] = len(successes)
    if successes:
        out["fragile_given_success_rate"] = label_rate(successes, "fragile_success_bool")
        out["robust_given_success_rate"] = label_rate(successes, "robust_success_bool")
        out["weak_post_margin_given_success_rate"] = label_rate(successes, "weak_post_margin_bool")
        out["weak_rephrase_given_success_rate"] = label_rate(successes, "weak_rephrase_bool")
        out["locality_damage_given_success_rate"] = label_rate(successes, "locality_damage_bool")
    for key in ["post_margin", "margin_gain", "rewrite_margin_delta", "rephrase_margin_delta", "locality_logprob_delta"]:
        vals = [to_float(row.get(key)) for row in rows]
        vals = [val for val in vals if val is not None]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_median"] = float(np.median(vals))
    return out


def feature_sets_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    sets = feature_ablation_sets(rows)
    diagnostics = diagnostic_features(rows)
    selected = {
        "diagnostic": diagnostics,
        "baseline_aggregate": sets.get("baseline_aggregate", []),
        "agop_compact_spectrum": sets.get("agop_compact_spectrum", []),
        "agop_spectrum_all_layers": sets.get("agop_spectrum_all_layers", []),
        "agop_topvec_all_layers": sets.get("agop_topvec_all_layers", []),
        "agop_neg5_spectrum": sets.get("agop_neg5_spectrum", []),
        "agop_neg13_topvec": sets.get("agop_neg13_topvec", []),
        "diagnostic_plus_agop_compact": diagnostics + sets.get("agop_compact_spectrum", []),
        "diagnostic_plus_agop_spectrum": diagnostics + sets.get("agop_spectrum_all_layers", []),
        "diagnostic_plus_agop_topvec": diagnostics + sets.get("agop_topvec_all_layers", []),
    }
    return {key: cols for key, cols in selected.items() if cols}


def bool_label_report(
    rows: List[Dict[str, Any]],
    features: Sequence[str],
    label_key: str,
    *,
    bootstrap_iters: int,
    seed: int,
) -> Dict[str, Any]:
    numeric_rows = []
    for row in rows:
        item = dict(row)
        item[label_key] = 1.0 if bool(row.get(label_key)) else 0.0
        numeric_rows.append(item)
    return grouped_logistic_report(
        numeric_rows,
        features,
        label_key,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
    )


def prediction_rows(
    rows: List[Dict[str, Any]],
    *,
    bootstrap_iters: int,
    seed: int,
) -> List[Dict[str, Any]]:
    out = []
    labels = [
        "standard_success_bool",
        "fragile_success_bool",
        "robust_success_bool",
        "weak_post_margin_bool",
        "weak_rephrase_bool",
        "locality_damage_bool",
    ]
    for split_name, split_subset in analysis_splits(rows).items():
        sets = feature_sets_for_rows(split_subset)
        for feature_set, cols in sets.items():
            for label in labels:
                report = bool_label_report(
                    split_subset,
                    cols,
                    label,
                    bootstrap_iters=bootstrap_iters,
                    seed=seed,
                )
                row: Dict[str, Any] = {
                    "split": split_name,
                    "feature_set": feature_set,
                    "label": label,
                    "n_features": len(cols),
                    "available": bool(report.get("available")),
                    "reason": report.get("reason", ""),
                }
                if report.get("available"):
                    row.update(
                        {
                            "n": report.get("n"),
                            "positive_rate": report.get("positive_rate"),
                            "cv_auroc": report.get("cv_auroc"),
                            "cv_average_precision": report.get("cv_average_precision"),
                            "auroc_ci_low": report.get("auroc_ci_low", ""),
                            "auroc_ci_high": report.get("auroc_ci_high", ""),
                        }
                    )
                out.append(row)
    return out


def controlled_delta_rows(pred_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {
        (row.get("split"), row.get("label"), row.get("feature_set")): row
        for row in pred_rows
        if row.get("available") and row.get("cv_auroc") not in ("", None)
    }
    comparisons = [
        ("diagnostic", "diagnostic_plus_agop_compact", "agop_compact_over_diagnostic"),
        ("diagnostic", "diagnostic_plus_agop_spectrum", "agop_spectrum_over_diagnostic"),
        ("diagnostic", "diagnostic_plus_agop_topvec", "agop_topvec_over_diagnostic"),
    ]
    out = []
    for split in sorted({row.get("split") for row in pred_rows}):
        for label in sorted({row.get("label") for row in pred_rows}):
            for base_set, plus_set, name in comparisons:
                base = by_key.get((split, label, base_set))
                plus = by_key.get((split, label, plus_set))
                if not base or not plus:
                    continue
                out.append(
                    {
                        "split": split,
                        "label": label,
                        "comparison": name,
                        "n": plus.get("n"),
                        "base_auroc": base.get("cv_auroc"),
                        "plus_auroc": plus.get("cv_auroc"),
                        "delta_auroc": safe_float(plus.get("cv_auroc")) - safe_float(base.get("cv_auroc")),
                        "base_ap": base.get("cv_average_precision"),
                        "plus_ap": plus.get("cv_average_precision"),
                        "delta_ap": safe_float(plus.get("cv_average_precision")) - safe_float(base.get("cv_average_precision")),
                    }
                )
    return out


def single_feature_rows(rows: List[Dict[str, Any]], labels: Sequence[str]) -> List[Dict[str, Any]]:
    cols = []
    for prefix in ["agop_feature_", "baseline_feature_"]:
        for col in sorted({key for row in rows for key in row if key.startswith(prefix)}):
            values = [to_float(row.get(col)) for row in rows]
            values = [val for val in values if val is not None]
            if len(values) >= 20 and len(set(values)) > 1:
                cols.append(col)
    out = []
    for label in labels:
        y = np.array([1 if row.get(label) else 0 for row in rows], dtype=int)
        if len(np.unique(y)) < 2:
            continue
        for col in cols:
            pairs = [(to_float(row.get(col)), 1 if row.get(label) else 0) for row in rows]
            pairs = [(x, yy) for x, yy in pairs if x is not None]
            if len(pairs) < 20:
                continue
            x = np.array([p[0] for p in pairs], dtype=float)
            yy = np.array([p[1] for p in pairs], dtype=int)
            out.append(
                {
                    "label": label,
                    "feature": col,
                    "n": len(pairs),
                    "auroc": auroc(x, yy),
                    "average_precision": average_precision(x, yy),
                    "positive_mean": float(np.mean(x[yy == 1])) if int(yy.sum()) else "",
                    "negative_mean": float(np.mean(x[yy == 0])) if int((yy == 0).sum()) else "",
                }
            )
    return sorted(out, key=lambda row: abs(float(row["auroc"]) - 0.5), reverse=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = label_rows(
        read_csv(args.error_rows_csv),
        min_post_margin=args.min_post_margin,
        min_rephrase_delta=args.min_rephrase_delta,
        max_locality_drop=args.max_locality_drop,
    )
    pred_rows = prediction_rows(rows, bootstrap_iters=args.bootstrap_iters, seed=args.seed)
    delta_rows = controlled_delta_rows(pred_rows)
    single_rows = single_feature_rows(
        rows,
        ["fragile_success_bool", "robust_success_bool", "weak_post_margin_bool", "locality_damage_bool"],
    )

    write_csv(out_dir / "superficial_editing_rows.csv", rows)
    write_csv(out_dir / "superficial_prediction_table.csv", pred_rows)
    write_csv(out_dir / "superficial_controlled_deltas.csv", delta_rows)
    write_csv(out_dir / "superficial_single_feature_report.csv", single_rows)

    summary = {
        "error_rows_csv": args.error_rows_csv,
        "thresholds": {
            "min_post_margin": args.min_post_margin,
            "min_rephrase_delta": args.min_rephrase_delta,
            "max_locality_drop": args.max_locality_drop,
        },
        "overall": outcome_summary(rows),
        "splits": {split: outcome_summary(split_rows_) for split, split_rows_ in split_rows(rows).items()},
        "top_prediction_rows": sorted(
            [row for row in pred_rows if row.get("available") and row.get("cv_auroc") not in ("", None)],
            key=lambda row: float(row["cv_auroc"]),
            reverse=True,
        )[:30],
        "controlled_deltas": delta_rows,
        "top_single_features": single_rows[:30],
    }
    (out_dir / "superficial_editing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "n_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
