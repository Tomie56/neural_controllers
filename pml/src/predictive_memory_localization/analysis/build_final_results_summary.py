from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def find_repo_root(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / "pml" / "src" / "predictive_memory_localization").is_dir():
            return parent
    raise RuntimeError(f"Could not infer repository root from {path}")


ROOT = find_repo_root(Path(__file__))
RESULTS = ROOT / "pml" / "results"
OUT_DIR = RESULTS / "final_summary"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f4(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def find_rows(rows: Iterable[Dict[str, Any]], **conds: Any) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        ok = True
        for key, value in conds.items():
            if str(row.get(key)) != str(value):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def top_row(rows: List[Dict[str, Any]], metric: str = "cv_auroc") -> Optional[Dict[str, Any]]:
    valid = []
    for row in rows:
        try:
            valid.append((float(row[metric]), row))
        except (KeyError, TypeError, ValueError):
            continue
    if not valid:
        return None
    return sorted(valid, key=lambda x: x[0], reverse=True)[0][1]


def suppression_rows() -> List[Dict[str, Any]]:
    summary_path = RESULTS / "commonsense_3000_baselines" / "summary" / "summary.json"
    pred_path = RESULTS / "commonsense_3000_baselines" / "prediction_analysis" / "prediction_summary.json"
    summary = read_json(summary_path)
    pred = read_json(pred_path)
    rows: List[Dict[str, Any]] = []

    by = summary["by_method_kind_coef"]
    target_rows = [row for row in by if row["kind"] == "target"]
    neighbor_rows = [row for row in by if row["kind"] == "neighbor"]
    best_target = min(target_rows, key=lambda row: float(row["delta_total_logprob_mean"]))
    best_neighbor_damage = max(neighbor_rows, key=lambda row: float(row["neighbor_damage_rate"]))
    for label, row in [
        ("best_target_drop", best_target),
        ("highest_neighbor_damage", best_neighbor_damage),
    ]:
        if row.get("kind") == "target":
            metric = "target_delta_mean"
            value = row.get("delta_total_logprob_mean")
            secondary_metric = "target_success_rate"
            secondary_value = row.get("target_success_rate")
        else:
            metric = "neighbor_damage_rate"
            value = row.get("neighbor_damage_rate")
            secondary_metric = "neighbor_delta_mean"
            secondary_value = row.get("delta_total_logprob_mean")
        rows.append(
            {
                "section": "activation_suppression_3000",
                "result": label,
                "n_records": summary.get("n_records"),
                "method": row.get("control_method"),
                "coef": row.get("coef"),
                "metric": metric,
                "value": value,
                "secondary_metric": secondary_metric,
                "secondary_value": secondary_value,
                "source_file": str(summary_path.relative_to(ROOT)),
            }
        )

    rows.extend(
        [
            {
                "section": "activation_suppression_3000",
                "result": "target_success_prediction",
                "n_records": pred.get("n_records"),
                "metric": "cv_auroc",
                "value": pred["target_logistic_model"].get("cv_auroc"),
                "secondary_metric": "cv_average_precision",
                "secondary_value": pred["target_logistic_model"].get("cv_average_precision"),
                "source_file": str(pred_path.relative_to(ROOT)),
            },
            {
                "section": "activation_suppression_3000",
                "result": "neighbor_damage_prediction",
                "n_records": pred.get("n_records"),
                "metric": "cv_auroc",
                "value": pred["neighbor_damage_logistic_model"].get("cv_auroc"),
                "secondary_metric": "cv_average_precision",
                "secondary_value": pred["neighbor_damage_logistic_model"].get("cv_average_precision"),
                "source_file": str(pred_path.relative_to(ROOT)),
            },
        ]
    )
    return rows


def agop_suppression_rows() -> List[Dict[str, Any]]:
    table_path = (
        RESULTS
        / "commonsense_500_agop_layer_features"
        / "focused_mean_difference_neg0_5_dedup"
        / "focused_agop_validation_table.csv"
    )
    table = read_csv(table_path)
    rows = []
    picks = [
        ("target", "target_suppression"),
        ("neighbor_damage", "neighbor_damage"),
        ("clean_success", "clean_success"),
    ]
    for outcome, label in picks:
        sub = [row for row in table if row.get("label") == outcome]
        best = top_row(sub, "auroc")
        if best:
            rows.append(
                {
                    "section": "agop_focused_suppression_500",
                    "result": label,
                    "n_records": best.get("n"),
                    "feature_set": best.get("feature_set"),
                    "metric": "auroc",
                    "value": best.get("auroc"),
                    "ci_low": best.get("auroc_ci_low"),
                    "ci_high": best.get("auroc_ci_high"),
                    "secondary_metric": "average_precision",
                    "secondary_value": best.get("ap"),
                    "source_file": str(table_path.relative_to(ROOT)),
                }
            )
    return rows


def agop_geometry_diagnostic_rows() -> List[Dict[str, Any]]:
    specs = [
        (
            RESULTS
            / "commonsense_500_agop_features"
            / "prediction_analysis"
            / "agop_prediction_summary.json",
            "agop_geometry_500",
            "agop_aggregate",
            [
                ("target_success", "target_agop_only_logistic_model"),
                ("neighbor_damage", "neighbor_damage_agop_only_logistic_model"),
                ("target_success_with_method_coef", "target_agop_only_logistic_model_with_method_coef"),
                ("neighbor_damage_with_method_coef", "neighbor_damage_agop_only_logistic_model_with_method_coef"),
            ],
        ),
        (
            RESULTS
            / "commonsense_500_agop_neighbor_overlap"
            / "prediction_analysis"
            / "overlap_prediction_summary.json",
            "agop_geometry_500",
            "agop_neighbor_overlap",
            [
                ("neighbor_damage", "neighbor_overlap_only"),
                ("clean_success", "clean_success_overlap_only"),
                ("neighbor_damage_with_method_coef", "neighbor_overlap_only_with_method_coef"),
                ("clean_success_with_method_coef", "clean_success_overlap_only_with_method_coef"),
            ],
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for path, section, feature_set, keys in specs:
        if not path.exists():
            continue
        summary = read_json(path)
        for result, key in keys:
            model = summary.get(key, {})
            if not model.get("available"):
                continue
            rows.append(
                {
                    "section": section,
                    "result": result,
                    "n_records": model.get("n"),
                    "feature_set": feature_set,
                    "metric": "cv_auroc",
                    "value": model.get("cv_auroc"),
                    "secondary_metric": "cv_average_precision",
                    "secondary_value": model.get("cv_average_precision"),
                    "positive_rate": model.get("positive_rate"),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return rows


def rome500_rows() -> List[Dict[str, Any]]:
    table_path = (
        RESULTS
        / "commonsense_500_agop_editing"
        / "rome"
        / "grouped_prediction_analysis_targeted"
        / "editing_grouped_prediction_table.csv"
    )
    table = read_csv(table_path)
    rows = []
    for outcome in ["efficacy", "rewrite_margin_delta", "rephrase_margin_delta", "locality", "locality_logprob_delta"]:
        sub = [
            row
            for row in table
            if row.get("split") == "all" and row.get("outcome") == outcome and row.get("available") == "True"
        ]
        best = top_row(sub, "cv_auroc")
        if best:
            rows.append(
                {
                    "section": "rome500_targeted",
                    "result": outcome,
                    "n_records": best.get("n"),
                    "feature_set": best.get("feature_set"),
                    "metric": "cv_auroc",
                    "value": best.get("cv_auroc"),
                    "ci_low": best.get("auroc_ci_low"),
                    "ci_high": best.get("auroc_ci_high"),
                    "secondary_metric": "cv_average_precision",
                    "secondary_value": best.get("cv_average_precision"),
                    "source_file": str(table_path.relative_to(ROOT)),
                }
            )
    return rows


def controlled_rows() -> List[Dict[str, Any]]:
    delta_path = (
        RESULTS
        / "commonsense_500_agop_editing"
        / "rome"
        / "controlled_incremental_analysis"
        / "controlled_incremental_deltas.csv"
    )
    deltas = read_csv(delta_path)
    wanted = [
        ("all", "rewrite_margin_delta", "agop_compact_over_diagnostic"),
        ("all", "efficacy", "agop_compact_over_diagnostic"),
        ("tau/commonsense_qa", "rewrite_margin_delta", "agop_compact_over_diagnostic"),
        ("tau/commonsense_qa", "rephrase_margin_delta", "agop_compact_over_diagnostic"),
        ("tau/commonsense_qa", "locality", "agop_compact_over_diagnostic"),
        ("allenai/ai2_arc", "rephrase_margin_delta", "agop_compact_over_diagnostic"),
    ]
    rows = []
    for split, outcome, comparison in wanted:
        match = find_rows(deltas, split=split, outcome=outcome, comparison=comparison)
        if not match:
            continue
        row = match[0]
        rows.append(
            {
                "section": "rome500_controlled_incremental",
                "result": f"{split}:{outcome}:{comparison}",
                "n_records": row.get("n"),
                "metric": "delta_auroc",
                "value": row.get("delta_auroc"),
                "secondary_metric": "delta_ap",
                "secondary_value": row.get("delta_ap"),
                "base_auroc": row.get("base_auroc"),
                "plus_auroc": row.get("plus_auroc"),
                "source_file": str(delta_path.relative_to(ROOT)),
            }
        )
    return rows


def length_rows() -> List[Dict[str, Any]]:
    delta_path = (
        RESULTS
        / "commonsense_500_agop_editing"
        / "rome"
        / "commonsenseqa_length_matched_agop"
        / "length_matched_deltas.csv"
    )
    deltas = read_csv(delta_path)
    wanted = [
        ("all_commonsenseqa", "rewrite_margin_delta", "agop_compact_over_length"),
        ("all_commonsenseqa", "rewrite_margin_delta", "agop_compact_over_difficulty"),
        ("length_1_or_2", "rewrite_margin_delta", "agop_compact_over_length"),
        ("length_1_or_2", "rewrite_margin_delta", "agop_compact_over_difficulty"),
    ]
    rows = []
    for split, outcome, comparison in wanted:
        match = find_rows(deltas, split=split, outcome=outcome, comparison=comparison)
        if not match:
            continue
        row = match[0]
        rows.append(
            {
                "section": "commonsenseqa_length_matched",
                "result": f"{split}:{outcome}:{comparison}",
                "n_records": row.get("n"),
                "metric": "delta_auroc",
                "value": row.get("delta_auroc"),
                "secondary_metric": "delta_ap",
                "secondary_value": row.get("delta_ap"),
                "base_auroc": row.get("base_auroc"),
                "plus_auroc": row.get("plus_auroc"),
                "source_file": str(delta_path.relative_to(ROOT)),
            }
        )
    return rows


def balanced_commonsenseqa_rows() -> List[Dict[str, Any]]:
    rows = []
    base_dir = (
        RESULTS
        / "commonsense_500_agop_editing"
        / "rome"
        / "commonsenseqa_balanced_agop"
    )
    wanted = [
        ("efficacy_cv", "agop_compact_over_difficulty"),
        ("rewrite_margin_delta_cv", "agop_compact_over_length"),
        ("rewrite_margin_delta_cv", "agop_compact_over_difficulty"),
        ("rephrase_margin_delta_cv", "agop_compact_over_length"),
        ("rephrase_margin_delta_cv", "agop_compact_over_difficulty"),
        ("rephrase_margin_delta_cv", "projection_last_over_difficulty"),
        ("efficacy_cv", "projection_last_over_difficulty"),
    ]
    for outcome_dir, comparison in wanted:
        delta_path = base_dir / outcome_dir / "balanced_deltas.csv"
        if not delta_path.exists():
            continue
        outcome = outcome_dir.removesuffix("_cv")
        for row in read_csv(delta_path):
            if row.get("comparison") != comparison or row.get("metric") != "auroc":
                continue
            rows.append(
                {
                    "section": "commonsenseqa_balanced_cv",
                    "result": f"{outcome}:{comparison}",
                    "metric": "delta_mean_auroc",
                    "value": row.get("delta_mean"),
                    "secondary_metric": "plus_mean_auroc",
                    "secondary_value": row.get("plus_mean"),
                    "base_auroc": row.get("base_mean"),
                    "plus_auroc": row.get("plus_mean"),
                    "source_file": str(delta_path.relative_to(ROOT)),
                }
            )
    return rows


def heldout_followup_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_dir = RESULTS / "commonsenseqa_balanced_followup_144" / "rome"

    grouped_summary_path = base_dir / "grouped_prediction_analysis" / "editing_grouped_prediction_summary.json"
    if grouped_summary_path.exists():
        summary = read_json(grouped_summary_path)
        for outcome, stats in summary.get("outcome_summary", {}).items():
            if not isinstance(stats, dict) or "n" not in stats:
                continue
            rows.append(
                {
                    "section": "heldout_commonsenseqa_144_outcomes",
                    "result": outcome,
                    "n_records": stats.get("n"),
                    "metric": "positive_rate",
                    "value": stats.get("positive_rate_threshold_0_5", stats.get("positive_rate")),
                    "secondary_metric": "mean",
                    "secondary_value": stats.get("mean"),
                    "source_file": str(grouped_summary_path.relative_to(ROOT)),
                }
            )

    grouped_table_path = base_dir / "grouped_prediction_analysis" / "editing_grouped_prediction_table.csv"
    if grouped_table_path.exists():
        table = read_csv(grouped_table_path)
        for outcome in ["efficacy", "rewrite_margin_delta", "rephrase_margin_delta", "locality", "locality_logprob_delta"]:
            sub = [
                row
                for row in table
                if row.get("split") == "all" and row.get("outcome") == outcome and row.get("available") == "True"
            ]
            best = top_row(sub, "cv_auroc")
            if best:
                rows.append(
                    {
                        "section": "heldout_commonsenseqa_144_grouped",
                        "result": outcome,
                        "n_records": best.get("n"),
                        "feature_set": best.get("feature_set"),
                        "metric": "cv_auroc",
                        "value": best.get("cv_auroc"),
                        "ci_low": best.get("auroc_ci_low"),
                        "ci_high": best.get("auroc_ci_high"),
                        "secondary_metric": "cv_average_precision",
                        "secondary_value": best.get("cv_average_precision"),
                        "source_file": str(grouped_table_path.relative_to(ROOT)),
                    }
                )

    controlled_path = base_dir / "controlled_incremental_analysis" / "controlled_incremental_deltas.csv"
    if controlled_path.exists():
        deltas = read_csv(controlled_path)
        wanted = [
            ("all", "efficacy", "agop_compact_over_diagnostic"),
            ("all", "rewrite_margin_delta", "agop_topvec_over_diagnostic"),
            ("all", "rewrite_margin_delta", "agop_spectrum_over_diagnostic"),
            ("all", "rewrite_margin_delta", "agop_compact_over_diagnostic"),
            ("all", "rephrase_margin_delta", "agop_topvec_over_diagnostic"),
            ("tau/commonsense_qa", "locality_logprob_delta", "agop_spectrum_over_diagnostic"),
        ]
        for split, outcome, comparison in wanted:
            match = find_rows(deltas, split=split, outcome=outcome, comparison=comparison)
            if not match:
                continue
            row = match[0]
            rows.append(
                {
                    "section": "heldout_commonsenseqa_144_controlled",
                    "result": f"{split}:{outcome}:{comparison}",
                    "n_records": row.get("n"),
                    "metric": "delta_auroc",
                    "value": row.get("delta_auroc"),
                    "secondary_metric": "delta_ap",
                    "secondary_value": row.get("delta_ap"),
                    "base_auroc": row.get("base_auroc"),
                    "plus_auroc": row.get("plus_auroc"),
                    "source_file": str(controlled_path.relative_to(ROOT)),
                }
            )

    balanced_base = base_dir / "balanced_cv_full_rows"
    wanted_balanced = [
        ("efficacy", "agop_compact_over_difficulty"),
        ("rewrite_margin_delta", "agop_compact_over_length"),
        ("rewrite_margin_delta", "agop_spectrum_over_difficulty"),
        ("rewrite_margin_delta", "agop_compact_over_difficulty"),
        ("rephrase_margin_delta", "agop_spectrum_over_difficulty"),
        ("rephrase_margin_delta", "agop_compact_over_difficulty"),
    ]
    for outcome, comparison in wanted_balanced:
        delta_path = balanced_base / outcome / "balanced_deltas.csv"
        if not delta_path.exists():
            continue
        for row in read_csv(delta_path):
            if row.get("comparison") != comparison or row.get("metric") != "auroc":
                continue
            rows.append(
                {
                    "section": "heldout_commonsenseqa_144_balanced_cv",
                    "result": f"{outcome}:{comparison}",
                    "metric": "delta_mean_auroc",
                    "value": row.get("delta_mean"),
                    "secondary_metric": "plus_mean_auroc",
                    "secondary_value": row.get("plus_mean"),
                    "base_auroc": row.get("base_mean"),
                    "plus_auroc": row.get("plus_mean"),
                    "source_file": str(delta_path.relative_to(ROOT)),
                }
            )
    return rows


def agop_direction_suppression_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary_path = (
        RESULTS
        / "commonsenseqa_balanced_followup_144"
        / "agop_direction_suppression"
        / "summary"
        / "summary.json"
    )
    pred_path = (
        RESULTS
        / "commonsenseqa_balanced_followup_144"
        / "agop_direction_suppression"
        / "prediction_analysis"
        / "prediction_summary.json"
    )
    if summary_path.exists():
        summary = read_json(summary_path)
        by = summary.get("by_method_kind_coef", [])
        wanted = [
            ("agop_top1", -1.0, "target"),
            ("agop_top1", -1.0, "neighbor"),
            ("agop_topk_project", -0.1, "target"),
            ("agop_topk_project", -0.1, "neighbor"),
            ("agop_topk_project", -1.0, "target"),
            ("agop_topk_project", -1.0, "neighbor"),
        ]
        for method, coef, kind in wanted:
            match = [
                row
                for row in by
                if row.get("control_method") == method
                and str(row.get("kind")) == kind
                and abs(float(row.get("coef", 999)) - coef) < 1e-9
            ]
            if not match:
                continue
            row = match[0]
            if kind == "target":
                metric = "target_success_rate"
                value = row.get("target_success_rate")
            else:
                metric = "neighbor_damage_rate"
                value = row.get("neighbor_damage_rate")
            rows.append(
                {
                    "section": "agop_direction_suppression_144",
                    "result": f"{method}:{coef}:{kind}",
                    "n_records": summary.get("n_records"),
                    "method": method,
                    "coef": coef,
                    "metric": metric,
                    "value": value,
                    "secondary_metric": "delta_total_logprob_mean",
                    "secondary_value": row.get("delta_total_logprob_mean"),
                    "source_file": str(summary_path.relative_to(ROOT)),
                }
            )
    if pred_path.exists():
        pred = read_json(pred_path)
        for result, key in [
            ("target_prediction", "target_logistic_model"),
            ("neighbor_damage_prediction", "neighbor_damage_logistic_model"),
        ]:
            model = pred.get(key, {})
            if not model.get("available"):
                continue
            rows.append(
                {
                    "section": "agop_direction_suppression_144",
                    "result": result,
                    "n_records": pred.get("n_records"),
                    "metric": "cv_auroc",
                    "value": model.get("cv_auroc"),
                    "secondary_metric": "cv_average_precision",
                    "secondary_value": model.get("cv_average_precision"),
                    "source_file": str(pred_path.relative_to(ROOT)),
                }
            )
    return rows


def agop_neighbor_overlap_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = RESULTS / "commonsenseqa_balanced_followup_144" / "agop_neighbor_overlap"
    summaries = [
        (
            "agop_overlap_direct_suppression_144",
            root / "direct_grouped_prediction_analysis" / "overlap_grouped_prediction_summary.json",
        ),
        (
            "agop_overlap_baseline_suppression_144",
            root / "baseline_grouped_prediction_analysis" / "overlap_grouped_prediction_summary.json",
        ),
    ]
    for section, path in summaries:
        if not path.exists():
            continue
        summary = read_json(path)
        for feature_set in ["pure_overlap", "spectrum", "all"]:
            report = summary.get("feature_sets", {}).get(feature_set, {})
            for result, key in [
                ("neighbor_damage", "neighbor_damage"),
                ("neighbor_damage_with_method_coef", "neighbor_damage_with_method_coef"),
                ("clean_success", "clean_success"),
                ("clean_success_with_method_coef", "clean_success_with_method_coef"),
            ]:
                model = report.get(key, {})
                if not model.get("available"):
                    continue
                rows.append(
                    {
                        "section": section,
                        "result": result,
                        "n_records": summary.get("n_records"),
                        "feature_set": feature_set,
                        "metric": "grouped_cv_auroc",
                        "value": model.get("grouped_cv_auroc"),
                        "secondary_metric": "grouped_cv_average_precision",
                        "secondary_value": model.get("grouped_cv_average_precision"),
                        "source_file": str(path.relative_to(ROOT)),
                    }
                )

        selected = []
        if "direct" in section:
            selected = [
                ("agop_top1", "-1.0"),
                ("agop_topk_project", "-0.1"),
                ("agop_topk_project", "-1.0"),
            ]
        else:
            selected = [
                ("mean_difference", "-0.5"),
                ("logistic", "-1.0"),
                ("linear", "-1.0"),
            ]
        method_specific = summary.get("method_specific", {})
        for method, coef in selected:
            report = method_specific.get(method, {}).get(coef, {})
            if not report:
                continue
            for result, key in [
                ("method_specific_neighbor_damage_pure_overlap", "pure_overlap_neighbor_damage"),
                ("method_specific_neighbor_damage_all", "all_neighbor_damage"),
                ("method_specific_clean_success_pure_overlap", "pure_overlap_clean_success"),
                ("method_specific_clean_success_all", "all_clean_success"),
            ]:
                model = report.get(key, {})
                if not model.get("available"):
                    continue
                rows.append(
                    {
                        "section": section,
                        "result": f"{method}:{coef}:{result}",
                        "n_records": report.get("n"),
                        "method": method,
                        "coef": coef,
                        "metric": "grouped_cv_auroc",
                        "value": model.get("grouped_cv_auroc"),
                        "secondary_metric": "positive_rate",
                        "secondary_value": report.get("neighbor_damage_rate")
                        if "neighbor_damage" in result
                        else report.get("clean_success_rate"),
                        "source_file": str(path.relative_to(ROOT)),
                    }
                )
    return rows


def superficial_editing_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    configs = [
        (
            "rome500_superficial_editing",
            RESULTS / "commonsense_500_agop_editing" / "rome" / "superficial_editing_analysis",
        ),
        (
            "heldout144_superficial_editing",
            RESULTS / "commonsenseqa_balanced_followup_144" / "rome" / "superficial_editing_analysis",
        ),
    ]
    for section, root in configs:
        summary_path = root / "superficial_editing_summary.json"
        pred_path = root / "superficial_prediction_table.csv"
        delta_path = root / "superficial_controlled_deltas.csv"
        if summary_path.exists():
            summary = read_json(summary_path)
            overall = summary.get("overall", {})
            for result, key in [
                ("standard_success_rate", "standard_success_rate"),
                ("fragile_given_success_rate", "fragile_given_success_rate"),
                ("robust_given_success_rate", "robust_given_success_rate"),
                ("weak_post_margin_given_success_rate", "weak_post_margin_given_success_rate"),
                ("weak_rephrase_given_success_rate", "weak_rephrase_given_success_rate"),
                ("locality_damage_given_success_rate", "locality_damage_given_success_rate"),
            ]:
                if key not in overall:
                    continue
                rows.append(
                    {
                        "section": section,
                        "result": result,
                        "n_records": overall.get("n_standard_success")
                        if "given_success" in result
                        else overall.get("n"),
                        "metric": "rate",
                        "value": overall.get(key),
                        "source_file": str(summary_path.relative_to(ROOT)),
                    }
                )
        if pred_path.exists():
            pred_rows = read_csv(pred_path)
            selected = [
                ("all:success_only", "fragile_success_bool", "diagnostic"),
                ("all:success_only", "fragile_success_bool", "agop_topvec_all_layers"),
                ("all:success_only", "fragile_success_bool", "agop_compact_spectrum"),
                ("all:success_only", "robust_success_bool", "diagnostic"),
                ("all:success_only", "locality_damage_bool", "agop_neg13_topvec"),
                ("all:success_only", "locality_damage_bool", "agop_neg5_spectrum"),
                ("all", "fragile_success_bool", "diagnostic"),
                ("all", "fragile_success_bool", "diagnostic_plus_agop_compact"),
            ]
            for split, label, feature_set in selected:
                matches = [
                    row
                    for row in pred_rows
                    if row.get("split") == split
                    and row.get("label") == label
                    and row.get("feature_set") == feature_set
                    and str(row.get("available")) == "True"
                ]
                if not matches:
                    continue
                row = matches[0]
                rows.append(
                    {
                        "section": section,
                        "result": f"{split}:{label}",
                        "n_records": row.get("n"),
                        "feature_set": feature_set,
                        "metric": "cv_auroc",
                        "value": row.get("cv_auroc"),
                        "secondary_metric": "cv_average_precision",
                        "secondary_value": row.get("cv_average_precision"),
                        "source_file": str(pred_path.relative_to(ROOT)),
                    }
                )
        if delta_path.exists():
            delta_rows = read_csv(delta_path)
            selected_deltas = [
                ("all:success_only", "fragile_success_bool", "agop_topvec_over_diagnostic"),
                ("all:success_only", "fragile_success_bool", "agop_compact_over_diagnostic"),
                ("all:success_only", "locality_damage_bool", "agop_spectrum_over_diagnostic"),
                ("all", "fragile_success_bool", "agop_compact_over_diagnostic"),
            ]
            for split, label, comparison in selected_deltas:
                matches = [
                    row
                    for row in delta_rows
                    if row.get("split") == split
                    and row.get("label") == label
                    and row.get("comparison") == comparison
                ]
                if not matches:
                    continue
                row = matches[0]
                rows.append(
                    {
                        "section": section,
                        "result": f"{split}:{label}:{comparison}",
                        "n_records": row.get("n"),
                        "metric": "delta_auroc",
                        "value": row.get("delta_auroc"),
                        "secondary_metric": "delta_ap",
                        "secondary_value": row.get("delta_ap"),
                        "source_file": str(delta_path.relative_to(ROOT)),
                    }
                )
    return rows


def path_geometry_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    configs = [
        (
            RESULTS / "path_geometry_audit",
            [
                ("commonsense_3000_baselines", "mean_difference"),
                ("commonsense_3000_baselines", "logistic"),
                ("commonsense_3000_baselines", "linear"),
                ("commonsense_3000_baselines", "random"),
                ("heldout_144_agop_direct", "agop_top1"),
                ("heldout_144_agop_direct", "agop_topk_project"),
            ],
        ),
        (
            RESULTS / "heldout_100_dense_alpha_baselines" / "path_geometry_audit",
            [
                ("heldout_100_dense_alpha_baselines", "mean_difference"),
                ("heldout_100_dense_alpha_baselines", "logistic"),
                ("heldout_100_dense_alpha_baselines", "linear"),
                ("heldout_100_dense_alpha_baselines", "random"),
            ],
        ),
    ]
    for root, wanted in configs:
        method_path = root / "method_strength_summary.csv"
        if not method_path.exists():
            continue
        table = read_csv(method_path)
        metrics = [
            ("target_success_any_rate", "target_any"),
            ("clean_window_exists_rate", "clean_window_exists"),
            ("path_type_clean-window_rate", "clean_window"),
            ("path_type_damage-first_rate", "damage_first"),
            ("path_type_no-effect_rate", "no_effect"),
            ("path_type_unstable_rate", "unstable"),
            ("path_type_collapse_rate", "collapse"),
            ("capability_measured_rate", "capability_measured"),
        ]
        for dataset, method in wanted:
            match = find_rows(table, dataset_label=dataset, control_method=method)
            if not match:
                continue
            row = match[0]
            for metric, label in metrics:
                if row.get(metric, "") == "":
                    continue
                rows.append(
                    {
                        "section": "path_geometry_audit",
                        "result": f"{dataset}:{method}:{label}",
                        "n_records": row.get("n_records"),
                        "method": method,
                        "metric": metric,
                        "value": row.get(metric),
                        "secondary_metric": "pareto_score_mean",
                        "secondary_value": row.get("pareto_score_mean"),
                        "source_file": str(method_path.relative_to(ROOT)),
                    }
                )

    pred_paths = [
        (
            RESULTS / "path_geometry_audit" / "strength_prediction" / "strength_prediction_summary.json",
            "features_only",
            [
                ("dataset=commonsense_3000_baselines", "baseline", "clean_window_label"),
                ("dataset=commonsense_3000_baselines", "baseline", "usable_path_label"),
                ("dataset=commonsense_3000_baselines", "baseline", "damage_first_label"),
                ("dataset=commonsense_3000_baselines", "baseline", "no_effect_label"),
                ("dataset=heldout_144_agop_direct", "agop_spectrum", "usable_path_label"),
                ("dataset=heldout_144_agop_direct", "agop_spectrum", "damage_first_label"),
                ("dataset=heldout_144_agop_direct", "agop_topvec", "damage_first_label"),
            ],
        ),
        (
            RESULTS / "path_geometry_audit" / "strength_prediction_with_method" / "strength_prediction_summary.json",
            "features_plus_method",
            [
                ("dataset=commonsense_3000_baselines", "baseline", "clean_window_label"),
                ("dataset=commonsense_3000_baselines", "baseline", "usable_path_label"),
                ("dataset=commonsense_3000_baselines", "baseline", "damage_first_label"),
                ("dataset=commonsense_3000_baselines", "baseline", "no_effect_label"),
                ("dataset=heldout_144_agop_direct", "agop_spectrum", "usable_path_label"),
                ("dataset=heldout_144_agop_direct", "agop_spectrum", "damage_first_label"),
                ("dataset=heldout_144_agop_direct", "agop_topvec", "damage_first_label"),
            ],
        ),
        (
            RESULTS
            / "heldout_100_dense_alpha_baselines"
            / "path_geometry_audit"
            / "strength_prediction"
            / "strength_prediction_summary.json",
            "dense_alpha_features_plus_method",
            [
                ("pooled", "baseline", "clean_window_label"),
                ("pooled", "baseline", "usable_path_label"),
                ("pooled", "baseline", "damage_first_label"),
                ("pooled", "baseline", "no_effect_label"),
            ],
        ),
    ]
    for pred_path, label, wanted_models in pred_paths:
        if not pred_path.exists():
            continue
        summary = read_json(pred_path)
        groups = summary.get("model_summaries", {})
        for group, feature_group, outcome in wanted_models:
            model = (
                groups.get(group, {})
                .get("feature_groups", {})
                .get(feature_group, {})
                .get("classification", {})
                .get(outcome, {})
            )
            if not model.get("available"):
                continue
            rows.append(
                {
                    "section": "path_geometry_prediction",
                    "result": f"{label}:{group}:{outcome}",
                    "n_records": model.get("n"),
                    "feature_set": feature_group,
                    "metric": "cv_auroc",
                    "value": model.get("cv_auroc"),
                    "secondary_metric": "cv_average_precision",
                    "secondary_value": model.get("cv_average_precision"),
                    "source_file": str(pred_path.relative_to(ROOT)),
                }
            )
    return rows


def representative_capability_probe_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = RESULTS / "path_representative_50_capability_onset_probe"
    table_path = root / "path_geometry_audit" / "representative_outcomes" / "selection_reason_summary.csv"
    status_path = root / "status.json"
    if not table_path.exists():
        return rows

    table = read_csv(table_path)
    wanted_reasons = [
        "agop_topk_damage_risk",
        "agop_topk_clean_window",
        "agop_top1_clean_window",
        "no_effect",
        "onset_borderline",
    ]
    metrics = [
        ("target_success_any_rate", "target_any"),
        ("neighbor_damage_any_rate", "neighbor_damage_any"),
        ("capability_damage_any_rate", "capability_damage_any"),
        ("path_type_clean-window_rate", "clean_window"),
        ("path_type_damage-first_rate", "damage_first"),
        ("path_type_capability-first_rate", "capability_first"),
        ("capability_onset_strength_median", "capability_onset_median"),
        ("capability_minus_target_onset_median", "capability_minus_target_median"),
    ]
    for reason in wanted_reasons:
        for method in ["mean_difference", "logistic"]:
            match = find_rows(table, selection_reason=reason, control_method=method)
            if not match:
                continue
            row = match[0]
            for metric, label in metrics:
                if row.get(metric, "") == "":
                    continue
                rows.append(
                    {
                        "section": "path_representative_capability_probe",
                        "result": f"{reason}:{method}:{label}",
                        "n_records": row.get("n_records"),
                        "method": method,
                        "metric": metric,
                        "value": row.get(metric),
                        "secondary_metric": "has_capability_eval_rate",
                        "secondary_value": row.get("has_capability_eval_rate"),
                        "source_file": str(table_path.relative_to(ROOT)),
                    }
                )
    if status_path.exists():
        status = read_json(status_path)
        if status.get("state") not in {"complete_with_analysis", "scoring_complete_analysis_missing"}:
            rows.append(
                {
                    "section": "path_representative_capability_probe",
                    "result": "status",
                    "metric": "state",
                    "value": status.get("state"),
                    "source_file": str(status_path.relative_to(ROOT)),
                }
            )
    return rows


def minimum_strength_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    table_path = (
        RESULTS
        / "heldout_100_dense_alpha_baselines"
        / "minimum_effective_strength"
        / "minimum_strength_method_summary.csv"
    )
    if not table_path.exists():
        return rows
    table = read_csv(table_path)
    metrics = [
        ("target_onset_median", "target_onset_median"),
        ("best_clean_strength_median_all_clean", "best_clean_strength_median"),
        ("strict_clean_path_rate", "strict_clean_path_rate"),
        ("target_onset_median_strict_clean", "strict_clean_onset_median"),
        ("damage_onset_median_damage_first", "damage_first_damage_onset_median"),
        ("unstable_rate", "unstable_rate"),
    ]
    for method in ["mean_difference", "logistic", "linear", "random"]:
        match = find_rows(table, control_method=method)
        if not match:
            continue
        row = match[0]
        for metric, label in metrics:
            if row.get(metric, "") == "":
                continue
            rows.append(
                {
                    "section": "minimum_effective_strength",
                    "result": f"{method}:{label}",
                    "n_records": row.get("n_records"),
                    "method": method,
                    "metric": metric,
                    "value": row.get(metric),
                    "secondary_metric": "clean_window_exists_rate",
                    "secondary_value": row.get("clean_window_exists_rate"),
                    "source_file": str(table_path.relative_to(ROOT)),
                }
            )
    return rows


def path_prediction_feature_audit_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    table_path = RESULTS / "paper_tables" / "feature_audit.csv"
    if not table_path.exists():
        table_path = RESULTS / "path_geometry_audit" / "feature_audit" / "path_prediction_top_coefficients.csv"
    if not table_path.exists():
        return rows
    table = read_csv(table_path)
    wanted = [
        ("pooled", "baseline", "clean_window_label", "1", "path_geometry"),
        ("pooled", "baseline", "usable_path_label", "1", "path_geometry"),
        ("pooled", "baseline", "damage_first_label", "1", "path_geometry"),
        ("pooled", "baseline", "no_effect_label", "1", "path_geometry"),
        ("heldout_144_agop_direct", "agop_spectrum", "usable_path_label", "1", "path_geometry_with_method"),
        ("heldout_144_agop_direct", "agop_spectrum", "damage_first_label", "1", "path_geometry_with_method"),
        ("heldout_144_agop_direct", "agop_topvec", "damage_first_label", "1", "path_geometry_with_method"),
        ("pooled", "baseline", "clean_window_label", "1", "heldout_100_dense_alpha_baselines"),
        ("pooled", "baseline", "usable_path_label", "1", "heldout_100_dense_alpha_baselines"),
        ("pooled", "baseline", "damage_first_label", "1", "heldout_100_dense_alpha_baselines"),
        ("pooled", "baseline", "no_effect_label", "1", "heldout_100_dense_alpha_baselines"),
    ]
    for dataset, feature_group, outcome, rank, summary_source in wanted:
        match = [
            row
            for row in table
            if str(row.get("dataset")) == dataset
            and str(row.get("feature_group")) == feature_group
            and str(row.get("outcome")) == outcome
            and str(row.get("rank")) == rank
            and str(row.get("summary_source", row.get("source", ""))) == summary_source
        ]
        if not match:
            continue
        row = match[0]
        rows.append(
            {
                "section": "path_prediction_feature_audit",
                "result": f"{summary_source}:{dataset}:{feature_group}:{outcome}:top{rank}",
                "feature_set": row.get("feature_family"),
                "metric": "coefficient",
                "value": row.get("coefficient"),
                "secondary_metric": "feature",
                "secondary_value": row.get("feature"),
                "cv_auroc": row.get("cv_auroc"),
                "source_file": str(table_path.relative_to(ROOT)),
            }
        )
    return rows


def path_editing_transfer_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = RESULTS / "commonsenseqa_balanced_followup_144" / "rome" / "path_editing_transfer"
    pred_path = root / "path_editing_prediction_rows.csv"
    delta_path = root / "path_editing_controlled_deltas.csv"
    group_path = root / "path_type_editing_group_summary.csv"
    stability_delta_path = root / "stability" / "stability_controlled_deltas.csv"
    if pred_path.exists():
        table = read_csv(pred_path)
        wanted = [
            ("standard_success_bool", "agop_top1"),
            ("robust_success_bool", "agop_top1"),
            ("fragile_success_bool", "both"),
            ("locality_damage_bool", "both"),
            ("weak_post_margin_bool", "agop_top1"),
        ]
        for outcome, feature_set in wanted:
            match = find_rows(table, outcome=outcome, feature_set=feature_set, available="True")
            if not match:
                continue
            row = match[0]
            rows.append(
                {
                    "section": "path_editing_transfer",
                    "result": outcome,
                    "n_records": row.get("n"),
                    "feature_set": feature_set,
                    "metric": "cv_auroc",
                    "value": row.get("cv_auroc"),
                    "secondary_metric": "cv_average_precision",
                    "secondary_value": row.get("cv_average_precision"),
                    "positive_rate": row.get("positive_rate"),
                    "source_file": str(pred_path.relative_to(ROOT)),
                }
            )
    if delta_path.exists():
        table = read_csv(delta_path)
        wanted_deltas = [
            ("fragile_success_bool", "diagnostic+agop_top1_over_diagnostic"),
            ("fragile_success_bool", "diagnostic+both_over_diagnostic"),
            ("robust_success_bool", "diagnostic+agop_top1_over_diagnostic"),
            ("standard_success_bool", "diagnostic+agop_top1_over_diagnostic"),
            ("locality_damage_bool", "diagnostic+both_over_diagnostic"),
        ]
        for outcome, comparison in wanted_deltas:
            match = find_rows(table, outcome=outcome, comparison=comparison)
            if not match:
                continue
            row = match[0]
            rows.append(
                {
                    "section": "path_editing_transfer_controlled",
                    "result": f"{outcome}:{comparison}",
                    "n_records": row.get("n"),
                    "metric": "delta_auroc",
                    "value": row.get("delta_auroc"),
                    "secondary_metric": "delta_ap",
                    "secondary_value": row.get("delta_ap"),
                    "base_auroc": row.get("base_auroc"),
                    "plus_auroc": row.get("plus_auroc"),
                    "source_file": str(delta_path.relative_to(ROOT)),
                }
            )
    if group_path.exists():
        table = read_csv(group_path)
        wanted_groups = [
            ("agop_top1", "clean-window", "standard_success_bool_rate"),
            ("agop_top1", "no-effect", "standard_success_bool_rate"),
            ("agop_top1", "damage-first", "fragile_success_bool_rate"),
            ("agop_topk_project", "collapse", "fragile_success_bool_rate"),
            ("agop_topk_project", "clean-window", "robust_success_bool_rate"),
        ]
        for method, path_type, metric in wanted_groups:
            match = find_rows(table, control_method=method, path_type=path_type)
            if not match or match[0].get(metric, "") == "":
                continue
            row = match[0]
            rows.append(
                {
                    "section": "path_editing_transfer_group",
                    "result": f"{method}:{path_type}:{metric}",
                    "n_records": row.get("n_records"),
                    "method": method,
                    "metric": metric,
                    "value": row.get(metric),
                    "secondary_metric": "rewrite_margin_delta_mean",
                    "secondary_value": row.get("rewrite_margin_delta_mean"),
                    "source_file": str(group_path.relative_to(ROOT)),
                }
            )
    if stability_delta_path.exists():
        table = read_csv(stability_delta_path)
        wanted_stability = [
            ("fragile_success_bool", "diagnostic+agop_top1_over_diagnostic", "auroc"),
            ("fragile_success_bool", "diagnostic+agop_top1_over_diagnostic", "ap"),
            ("fragile_success_bool", "diagnostic+both_over_diagnostic", "auroc"),
            ("robust_success_bool", "diagnostic+agop_top1_over_diagnostic", "auroc"),
            ("locality_damage_bool", "diagnostic+both_over_diagnostic", "auroc"),
        ]
        for outcome, comparison, metric in wanted_stability:
            match = find_rows(table, outcome=outcome, comparison=comparison, metric=metric)
            if not match:
                continue
            row = match[0]
            rows.append(
                {
                    "section": "path_editing_transfer_stability",
                    "result": f"{outcome}:{comparison}:{metric}",
                    "metric": "delta_observed",
                    "value": row.get("delta_observed"),
                    "secondary_metric": "delta_95ci",
                    "secondary_value": f"[{f4(row.get('delta_ci_low'))},{f4(row.get('delta_ci_high'))}]",
                    "base_auroc": row.get("base_observed") if metric == "auroc" else "",
                    "plus_auroc": row.get("plus_observed") if metric == "auroc" else "",
                    "p_bootstrap_delta_le_zero": row.get("p_bootstrap_delta_le_zero"),
                    "p_permutation_two_sided": row.get("p_permutation_two_sided"),
                    "source_file": str(stability_delta_path.relative_to(ROOT)),
                }
            )
    return rows


def path_editing_case_audit_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = RESULTS / "commonsenseqa_balanced_followup_144" / "rome" / "path_editing_transfer" / "case_audit"
    bucket_path = root / "path_editing_case_audit_buckets.csv"
    if not bucket_path.exists():
        return rows
    table = read_csv(bucket_path)
    wanted = [
        ("risk_0", "fragile_success_rate"),
        ("risk_5_plus", "fragile_success_rate"),
        ("risk_5_plus", "standard_success_rate"),
        ("clean_0", "robust_success_rate"),
        ("clean_3_plus", "robust_success_rate"),
        ("clean_3_plus", "fragile_success_rate"),
    ]
    for bucket, metric in wanted:
        match = find_rows(table, bucket=bucket)
        if not match or match[0].get(metric, "") == "":
            continue
        row = match[0]
        rows.append(
            {
                "section": "path_editing_case_audit",
                "result": f"{bucket}:{metric}",
                "n_records": row.get("n_records"),
                "metric": metric,
                "value": row.get(metric),
                "secondary_metric": "rewrite_margin_delta_mean",
                "secondary_value": row.get("rewrite_margin_delta_mean"),
                "source_file": str(bucket_path.relative_to(ROOT)),
            }
        )
    return rows


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def build_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Predictive Memory Localization Final Results Summary",
        "",
        "Generated from current result files. Paths are relative to `/data/neural_controllers`.",
        "",
        "## Main Quantitative Results",
        "",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.get("section", ""),
                row.get("result", ""),
                row.get("n_records", ""),
                row.get("feature_set", row.get("method", "")),
                row.get("metric", ""),
                f4(row.get("value")),
                row.get("secondary_metric", ""),
                f4(row.get("secondary_value")),
            ]
        )
    lines.append(
        md_table(
            ["Section", "Result", "n", "Method/Features", "Metric", "Value", "Secondary", "Secondary Value"],
            table_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Learned activation-suppression directions outperform random for target logprob reduction, but stronger suppression damages neighbors.",
            "- Baseline localization features predict target suppression moderately and neighbor damage weakly.",
            "- AGOP/RFM per-layer features are useful for focused suppression prediction, especially clean suppression / neighbor risk under `mean_difference -0.5`.",
            "- AGOP/RFM aggregate and target-neighbor overlap features give moderate diagnostic prediction for target leverage, neighbor damage, and clean success on the 500-record diagnostic set; this supports treating AGOP as predictive geometry rather than a direct clean-steering method.",
            "- ROME 500 shows outcome-specific predictability: pooled efficacy is mostly explained by edit difficulty, while AGOP compact spectrum has source-specific margin signal.",
            "- Controlled analyses show AGOP compact adds rewrite/rephrase margin information on CommonsenseQA beyond diagnostic features, but not general efficacy/locality.",
            "- CommonsenseQA length-matched analysis supports that AGOP compact rewrite-margin signal is not only a target-length artifact, though exact-length strata are underpowered.",
            "- CommonsenseQA balanced-CV follow-up confirms positive AGOP compact deltas for rewrite/rephrase margins, but not for efficacy; projection/baseline features can match AGOP on rephrase margin.",
            "- A held-out balanced CommonsenseQA follow-up narrows the AGOP claim: difficulty still dominates efficacy; AGOP layer/top-vector/spectrum signals help some rewrite/locality-style outcomes, while compact spectrum does not replicate as a universal improvement.",
            "- Direct AGOP intervention is a boundary result: top-k AGOP projection strongly suppresses targets but damages neighbors heavily, while AGOP top-1 is gentler but does not beat simple mean-difference/logistic baselines on clean tradeoff.",
            "- Path-geometry audit strengthens this boundary result: target-any success and clean-window intervention diverge sharply, with AGOP top-k showing high target leverage but frequent damage-first / collapse paths.",
            "- Held-out 100 dense-alpha audit shows that denser strength grids expose path instability: learned directions keep higher target-any rates than random, but strict clean-window path types remain rare, so minimum effective strength must be conditioned on path stability rather than target onset alone.",
            "- Path-conditioned minimum-strength analysis separates first target onset from usable strength: learned directions often reach target onset around alpha 0.1, but strict clean paths are rare and damage-first paths often show neighbor damage onset around alpha 0.05.",
            "- Path-to-editing transfer on the held-out 144 suggests activation path geometry weakly predicts standard ROME efficacy, but adds controlled signal for fragile / superficial editing risk.",
            "- Bootstrap stability checks sharpen that claim: fragile-risk controlled deltas stay positive but their 95% intervals cross zero, so this is a promising diagnostic signal rather than a settled effect; robust/locality deltas are not improved by adding path geometry.",
            "- Case audit shows the same pattern qualitatively: high path-risk buckets have more fragile edits than risk-free buckets, but there are many high-risk robust counterexamples, so path geometry should be used as a risk flag rather than a deterministic edit-success predictor.",
            "- Feature audit suggests path-type models rely on saliency entropy/concentration and direction agreement for baseline directions, and on AGOP spectrum/top-k structure for AGOP path-risk diagnostics.",
            "- Current localization features predict path type better than exact onset strength; the next capability-onset probe should measure whether unrelated QA behavior degrades before target onset.",
            "- Target-neighbor AGOP overlap weakly predicts side effects by itself and becomes strong only when method/coef is included, so overlap is best read as a risk diagnostic rather than a standalone locality guarantee.",
            "- Superficial-editing analysis shows that roughly a quarter to a third of standard-success ROME edits are fragile under margin/rephrase/locality checks; AGOP does not reliably improve success-only fragile-edit prediction over diagnostics.",
            "- A paper-facing result matrix is available at `PML_PAPER_RESULTS_MATRIX_中文.md`; it maps research questions to evidence, figures, claims, and current gaps.",
            "- Paper-ready tables are generated under `pml/results/paper_tables/`.",
            "",
            "## Recommended Paper Claim",
            "",
            "> Predictive Memory Localization is a diagnostic framework for testing when internal memory signals predict interventions. The most informative object is not a localization heatmap but a strength/path curve: target onset, damage onset, capability onset, and the existence of a clean intervention window. AGOP/RFM spectral geometry provides incremental prediction for selected edit-margin and path-risk outcomes, but edit efficacy is dominated by difficulty diagnostics and AGOP does not universally predict locality.",
            "",
            "## Source Files",
            "",
        ]
    )
    for path in sorted({str(row.get("source_file")) for row in rows if row.get("source_file")}):
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    rows.extend(suppression_rows())
    rows.extend(agop_suppression_rows())
    rows.extend(agop_geometry_diagnostic_rows())
    rows.extend(rome500_rows())
    rows.extend(controlled_rows())
    rows.extend(length_rows())
    rows.extend(balanced_commonsenseqa_rows())
    rows.extend(heldout_followup_rows())
    rows.extend(agop_direction_suppression_rows())
    rows.extend(agop_neighbor_overlap_rows())
    rows.extend(path_geometry_rows())
    rows.extend(representative_capability_probe_rows())
    rows.extend(minimum_strength_rows())
    rows.extend(path_prediction_feature_audit_rows())
    rows.extend(path_editing_transfer_rows())
    rows.extend(path_editing_case_audit_rows())
    rows.extend(superficial_editing_rows())
    write_csv(OUT_DIR / "final_results_summary.csv", rows)
    (OUT_DIR / "FINAL_RESULTS_SUMMARY.md").write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "n_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
