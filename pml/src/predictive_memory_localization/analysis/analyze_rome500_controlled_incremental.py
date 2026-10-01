from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.analysis.analyze_editing_grouped_prediction import (
    OUTCOMES,
    grouped_logistic_report,
    to_float,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled diagnostic-vs-AGOP incremental prediction for ROME editing."
    )
    parser.add_argument("--error-rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def numeric_available(rows: List[Dict[str, Any]], col: str, min_rows: int = 20) -> bool:
    vals = [to_float(row.get(col)) for row in rows]
    vals = [v for v in vals if v is not None]
    return len(vals) >= min_rows and len(set(vals)) > 1


def cols_matching(
    rows: List[Dict[str, Any]],
    *,
    prefix: str = "",
    includes: Sequence[str] = (),
    excludes: Sequence[str] = (),
) -> List[str]:
    cols = sorted({key for row in rows for key in row if key.startswith(prefix)})
    out = []
    for col in cols:
        if includes and not any(token in col for token in includes):
            continue
        if excludes and any(token in col for token in excludes):
            continue
        if numeric_available(rows, col):
            out.append(col)
    return out


def source_one_hot(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = sorted({row.get("source_dataset", "unknown") for row in rows})
    out = []
    for row in rows:
        copied = dict(row)
        source = row.get("source_dataset", "unknown")
        for item in sources:
            copied[f"diagnostic_source_is_{item.replace('/', '__')}"] = 1.0 if source == item else 0.0
        out.append(copied)
    return out


def split_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"all": rows}
    for row in rows:
        out.setdefault(row.get("source_dataset", "unknown"), []).append(row)
    return out


def feature_sets(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    diagnostic_base = [
        "pre_rewrite_logprob",
        "pre_rewrite_mean_logprob",
        "pre_rewrite_num_tokens",
        "pre_old_logprob",
        "pre_old_mean_logprob",
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
        "subject_prompt_pos",
        "subject_prompt_char_ratio",
        "num_paraphrase_prompts",
        "num_locality_prompts",
    ]
    diagnostic = [col for col in diagnostic_base if numeric_available(rows, col)]
    diagnostic_with_source = diagnostic + cols_matching(rows, prefix="diagnostic_source_is_")

    projection_last = cols_matching(rows, prefix="agop_feature_proj_layer_-1_")
    projection_all = cols_matching(rows, prefix="agop_feature_proj_layer_")
    baseline = cols_matching(rows, prefix="baseline_feature_", excludes=["top_saliency_layer"])
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
    agop_topvec = cols_matching(
        rows,
        prefix="agop_feature_agop_layer_",
        includes=["top_vec_abs_cos", "top_vec_cohen", "top_vec_mean", "top_vec_threshold"],
    )
    agop_compact = cols_matching(
        rows,
        prefix="agop_feature_",
        includes=["effective_rank", "spectral_entropy", "top1_ratio", "topk_ratio", "trace", "condition_number"],
        excludes=["proj_layer"],
    )
    agop_neg5 = cols_matching(
        rows,
        prefix="agop_feature_agop_layer_neg5_",
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
    agop_neg13_topvec = cols_matching(
        rows,
        prefix="agop_feature_agop_layer_neg13_",
        includes=["top_vec_abs_cos", "top_vec_cohen", "top_vec_mean", "top_vec_threshold"],
    )

    sets = {
        "diagnostic": diagnostic,
        "diagnostic_with_source": diagnostic_with_source,
        "baseline": baseline,
        "projection_last": projection_last,
        "projection_all": projection_all,
        "agop_spectrum": agop_spectrum,
        "agop_topvec": agop_topvec,
        "agop_compact": agop_compact,
        "agop_neg5_spectrum": agop_neg5,
        "agop_neg13_topvec": agop_neg13_topvec,
        "diagnostic_plus_projection_last": diagnostic + projection_last,
        "diagnostic_plus_agop_compact": diagnostic + agop_compact,
        "diagnostic_plus_agop_spectrum": diagnostic + agop_spectrum,
        "diagnostic_plus_agop_topvec": diagnostic + agop_topvec,
        "diagnostic_source_plus_agop_compact": diagnostic_with_source + agop_compact,
        "diagnostic_source_plus_agop_spectrum": diagnostic_with_source + agop_spectrum,
        "diagnostic_source_plus_projection_last": diagnostic_with_source + projection_last,
        "diagnostic_source_plus_baseline": diagnostic_with_source + baseline,
    }
    return {name: cols for name, cols in sets.items() if cols}


def run_reports(
    rows: List[Dict[str, Any]],
    *,
    bootstrap_iters: int,
    seed: int,
) -> List[Dict[str, Any]]:
    reports = []
    for split_name, split in split_rows(rows).items():
        sets = feature_sets(split)
        for set_name, cols in sets.items():
            for outcome in OUTCOMES:
                report = grouped_logistic_report(
                    split,
                    cols,
                    outcome,
                    bootstrap_iters=bootstrap_iters,
                    seed=seed,
                )
                row: Dict[str, Any] = {
                    "split": split_name,
                    "feature_set": set_name,
                    "outcome": outcome,
                    "n_features": len(cols),
                    "available": bool(report.get("available")),
                    "reason": report.get("reason", ""),
                }
                if report.get("available"):
                    row.update(
                        {
                            "n": report["n"],
                            "positive_rate": report["positive_rate"],
                            "cv_auroc": report["cv_auroc"],
                            "auroc_ci_low": report.get("auroc_ci_low", ""),
                            "auroc_ci_high": report.get("auroc_ci_high", ""),
                            "cv_average_precision": report["cv_average_precision"],
                            "ap_ci_low": report.get("ap_ci_low", ""),
                            "ap_ci_high": report.get("ap_ci_high", ""),
                        }
                    )
                reports.append(row)
    return reports


def delta_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {
        (row.get("split"), row.get("outcome"), row.get("feature_set")): row
        for row in reports
        if row.get("available") and row.get("cv_auroc") not in ("", None)
    }
    comparisons = [
        ("diagnostic_plus_agop_compact", "diagnostic", "agop_compact_over_diagnostic"),
        ("diagnostic_plus_agop_spectrum", "diagnostic", "agop_spectrum_over_diagnostic"),
        ("diagnostic_plus_agop_topvec", "diagnostic", "agop_topvec_over_diagnostic"),
        ("diagnostic_plus_projection_last", "diagnostic", "projection_last_over_diagnostic"),
        ("diagnostic_source_plus_agop_compact", "diagnostic_with_source", "agop_compact_over_diagnostic_source"),
        ("diagnostic_source_plus_agop_spectrum", "diagnostic_with_source", "agop_spectrum_over_diagnostic_source"),
        ("diagnostic_source_plus_projection_last", "diagnostic_with_source", "projection_last_over_diagnostic_source"),
        ("diagnostic_source_plus_baseline", "diagnostic_with_source", "baseline_over_diagnostic_source"),
    ]
    out = []
    splits = sorted({row.get("split") for row in reports})
    outcomes = OUTCOMES
    for split in splits:
        for outcome in outcomes:
            for plus, base, label in comparisons:
                plus_row = by_key.get((split, outcome, plus))
                base_row = by_key.get((split, outcome, base))
                if not plus_row or not base_row:
                    continue
                out.append(
                    {
                        "split": split,
                        "outcome": outcome,
                        "comparison": label,
                        "base_feature_set": base,
                        "plus_feature_set": plus,
                        "base_auroc": base_row["cv_auroc"],
                        "plus_auroc": plus_row["cv_auroc"],
                        "delta_auroc": float(plus_row["cv_auroc"]) - float(base_row["cv_auroc"]),
                        "base_ap": base_row["cv_average_precision"],
                        "plus_ap": plus_row["cv_average_precision"],
                        "delta_ap": float(plus_row["cv_average_precision"]) - float(base_row["cv_average_precision"]),
                        "n": plus_row["n"],
                    }
                )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = source_one_hot(read_csv(args.error_rows_csv))
    reports = run_reports(rows, bootstrap_iters=args.bootstrap_iters, seed=args.seed)
    deltas = delta_rows(reports)
    write_csv(out_dir / "controlled_prediction_table.csv", reports)
    write_csv(out_dir / "controlled_incremental_deltas.csv", deltas)
    summary = {
        "error_rows_csv": args.error_rows_csv,
        "n_rows": len(rows),
        "top_deltas": sorted(deltas, key=lambda row: float(row["delta_auroc"]), reverse=True)[:40],
        "bottom_deltas": sorted(deltas, key=lambda row: float(row["delta_auroc"]))[:20],
    }
    (out_dir / "controlled_incremental_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "n_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
