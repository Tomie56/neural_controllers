from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.analysis.analyze_editing_grouped_prediction import grouped_logistic_report, to_float, write_csv
from predictive_memory_localization.analysis.analyze_rome500_error_ablation import feature_ablation_sets
from predictive_memory_localization.analysis.analyze_superficial_editing import diagnostic_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict attack/reversal prompt robustness from PML features.")
    parser.add_argument("--error-rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attack-success-threshold", type=float, default=1.0)
    parser.add_argument("--attack-margin-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def label_rows(rows: List[Dict[str, Any]], attack_success_threshold: float, attack_margin_threshold: float) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        attack_success = to_float(item.get("attack_success_rate"))
        attack_margin = to_float(item.get("post_mean_attack_new_minus_old_logprob"))
        old_resurgence = to_float(item.get("attack_old_resurgence_rate"))
        standard_success = (to_float(item.get("efficacy")) or 0.0) >= 0.5
        has_attack = attack_success is not None and attack_margin is not None
        item["has_attack_eval_bool"] = bool(has_attack)
        item["attack_success_bool"] = bool(
            has_attack
            and attack_success >= attack_success_threshold
            and attack_margin > attack_margin_threshold
        )
        item["attack_old_resurgence_bool"] = bool(has_attack and old_resurgence is not None and old_resurgence > 0.0)
        item["attack_robust_success_bool"] = bool(standard_success and item["attack_success_bool"])
        item["attack_fragile_success_bool"] = bool(standard_success and has_attack and not item["attack_success_bool"])
        out.append(item)
    return out


def analysis_splits(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    attack_rows = [row for row in rows if row.get("has_attack_eval_bool")]
    splits: Dict[str, List[Dict[str, Any]]] = {"all": attack_rows}
    for row in attack_rows:
        splits.setdefault(str(row.get("source_dataset", "unknown")), []).append(row)
    for split_name, split_rows in list(splits.items()):
        success_only = [row for row in split_rows if (to_float(row.get("efficacy")) or 0.0) >= 0.5]
        if success_only:
            splits[f"{split_name}:success_only"] = success_only
    return splits


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


def bool_report(
    rows: List[Dict[str, Any]],
    features: Sequence[str],
    label: str,
    *,
    bootstrap_iters: int,
    seed: int,
) -> Dict[str, Any]:
    numeric_rows = []
    for row in rows:
        item = dict(row)
        item[label] = 1.0 if row.get(label) else 0.0
        numeric_rows.append(item)
    return grouped_logistic_report(
        numeric_rows,
        features,
        label,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
    )


def prediction_rows(rows: List[Dict[str, Any]], bootstrap_iters: int, seed: int) -> List[Dict[str, Any]]:
    labels = [
        "attack_success_bool",
        "attack_old_resurgence_bool",
        "attack_robust_success_bool",
        "attack_fragile_success_bool",
    ]
    out = []
    for split_name, split_rows in analysis_splits(rows).items():
        sets = feature_sets_for_rows(split_rows)
        for feature_set, cols in sets.items():
            for label in labels:
                report = bool_report(split_rows, cols, label, bootstrap_iters=bootstrap_iters, seed=seed)
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
            for base_set, plus_set, comparison in comparisons:
                base = by_key.get((split, label, base_set))
                plus = by_key.get((split, label, plus_set))
                if not base or not plus:
                    continue
                out.append(
                    {
                        "split": split,
                        "label": label,
                        "comparison": comparison,
                        "n": plus.get("n"),
                        "base_auroc": base.get("cv_auroc"),
                        "plus_auroc": plus.get("cv_auroc"),
                        "delta_auroc": float(plus.get("cv_auroc")) - float(base.get("cv_auroc")),
                        "base_ap": base.get("cv_average_precision"),
                        "plus_ap": plus.get("cv_average_precision"),
                        "delta_ap": float(plus.get("cv_average_precision")) - float(base.get("cv_average_precision")),
                    }
                )
    return out


def rate(rows: List[Dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row.get(key)) for row in rows])) if rows else 0.0


def numeric_mean(rows: List[Dict[str, Any]], key: str) -> Any:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [val for val in vals if val is not None]
    return float(np.mean(vals)) if vals else ""


def summary_for_split(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(rows),
        "standard_success_rate": float(np.mean([(to_float(row.get("efficacy")) or 0.0) >= 0.5 for row in rows])) if rows else 0.0,
        "attack_success_rate": rate(rows, "attack_success_bool"),
        "attack_old_resurgence_rate": rate(rows, "attack_old_resurgence_bool"),
        "attack_robust_success_rate": rate(rows, "attack_robust_success_bool"),
        "attack_fragile_success_rate": rate(rows, "attack_fragile_success_bool"),
        "mean_attack_success_rate": numeric_mean(rows, "attack_success_rate"),
        "mean_attack_old_resurgence_rate": numeric_mean(rows, "attack_old_resurgence_rate"),
        "mean_post_attack_margin": numeric_mean(rows, "post_mean_attack_new_minus_old_logprob"),
        "mean_attack_margin_delta": numeric_mean(rows, "attack_margin_delta"),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = label_rows(read_csv(args.error_rows_csv), args.attack_success_threshold, args.attack_margin_threshold)
    pred_rows = prediction_rows(rows, args.bootstrap_iters, args.seed)
    delta_rows = controlled_delta_rows(pred_rows)
    write_csv(out_dir / "attack_editing_rows.csv", rows)
    write_csv(out_dir / "attack_prediction_table.csv", pred_rows)
    write_csv(out_dir / "attack_controlled_deltas.csv", delta_rows)
    attack_splits = analysis_splits(rows)
    summary = {
        "error_rows_csv": args.error_rows_csv,
        "thresholds": {
            "attack_success_threshold": args.attack_success_threshold,
            "attack_margin_threshold": args.attack_margin_threshold,
        },
        "splits": {name: summary_for_split(split) for name, split in attack_splits.items()},
        "top_prediction_rows": sorted(
            [row for row in pred_rows if row.get("available") and row.get("cv_auroc") not in ("", None)],
            key=lambda row: float(row["cv_auroc"]),
            reverse=True,
        )[:30],
        "controlled_deltas": delta_rows,
    }
    (out_dir / "attack_editing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "n_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
