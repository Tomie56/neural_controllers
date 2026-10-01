from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def find_repo_root(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / "pml" / "src" / "predictive_memory_localization").is_dir():
            return parent
    raise RuntimeError(f"Could not infer repository root from {path}")


ROOT = find_repo_root(Path(__file__))
RESULTS = ROOT / "pml" / "results"
OUT_DIR = RESULTS / "paper_tables"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def find_row(rows: Iterable[Dict[str, Any]], **conds: Any) -> Optional[Dict[str, Any]]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in conds.items()):
            return row
    return None


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def path_summary_table() -> List[Dict[str, Any]]:
    path = RESULTS / "path_geometry_audit" / "method_strength_summary.csv"
    table = read_csv(path)
    rows = []
    configs = [
        (
            path,
            table,
            "commonsense_3000_baselines",
            ["mean_difference", "logistic", "linear", "random"],
        ),
        (
            path,
            table,
            "heldout_144_agop_direct",
            ["agop_top1", "agop_topk_project"],
        ),
    ]
    dense_path = (
        RESULTS
        / "heldout_100_dense_alpha_baselines"
        / "path_geometry_audit"
        / "method_strength_summary.csv"
    )
    if dense_path.exists():
        configs.append(
            (
                dense_path,
                read_csv(dense_path),
                "heldout_100_dense_alpha_baselines",
                ["mean_difference", "logistic", "linear", "random"],
            )
        )

    for source_path, source_table, dataset, methods in configs:
        for method in methods:
            row = find_row(source_table, dataset_label=dataset, control_method=method)
            if not row:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": row.get("n_records"),
                    "target_any": f4(row.get("target_success_any_rate")),
                    "neighbor_damage_any": f4(row.get("neighbor_damage_any_rate")),
                    "clean_window": f4(row.get("path_type_clean-window_rate")),
                    "damage_first": f4(row.get("path_type_damage-first_rate")),
                    "no_effect": f4(row.get("path_type_no-effect_rate")),
                    "collapse": f4(row.get("path_type_collapse_rate")),
                    "target_onset_median": f4(row.get("target_onset_strength_median")),
                    "damage_onset_median": f4(row.get("damage_onset_strength_median")),
                    "source_file": str(source_path.relative_to(ROOT)),
                }
            )
    return rows


def path_prediction_table() -> List[Dict[str, Any]]:
    rows = []
    configs = [
        (
            RESULTS / "path_geometry_audit" / "strength_prediction" / "strength_prediction_summary.json",
            "pooled",
            "commonsense_3000_baselines",
            "baseline",
            ["clean_window_label", "usable_path_label", "damage_first_label", "no_effect_label"],
        ),
        (
            RESULTS / "path_geometry_audit" / "strength_prediction_with_method" / "strength_prediction_summary.json",
            "pooled",
            "heldout_144_agop_direct",
            "agop_spectrum",
            ["usable_path_label", "damage_first_label"],
        ),
        (
            RESULTS / "path_geometry_audit" / "strength_prediction_with_method" / "strength_prediction_summary.json",
            "pooled",
            "heldout_144_agop_direct",
            "agop_topvec",
            ["damage_first_label"],
        ),
        (
            RESULTS
            / "heldout_100_dense_alpha_baselines"
            / "path_geometry_audit"
            / "strength_prediction"
            / "strength_prediction_summary.json",
            "pooled",
            "heldout_100_dense_alpha_baselines",
            "baseline",
            ["clean_window_label", "usable_path_label", "damage_first_label", "no_effect_label"],
        ),
    ]
    for path, group_key, dataset, feature_group, labels in configs:
        if not path.exists():
            continue
        summary = read_json(path)
        model_summaries = summary.get("model_summaries", {})
        group = model_summaries.get(f"dataset={dataset}") or model_summaries.get(group_key) or {}
        feature_groups = group.get("feature_groups", {})
        feature_summary = feature_groups.get(feature_group, {})
        classification = feature_summary.get("classification", {})
        for label in labels:
            row = classification.get(label)
            if not row:
                continue
            rows.append(
                {
                    "setting": "features_plus_method" if read_json(path).get("include_method") else "features_only",
                    "dataset": dataset,
                    "feature_group": feature_group,
                    "outcome": label,
                    "n": row.get("n"),
                    "positive_rate": f4(row.get("positive_rate")),
                    "cv_auroc": f4(row.get("cv_auroc")),
                    "cv_ap": f4(row.get("cv_average_precision")),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return rows


def editing_transfer_table() -> List[Dict[str, Any]]:
    root = RESULTS / "commonsenseqa_balanced_followup_144" / "rome" / "path_editing_transfer"
    pred = read_csv(root / "path_editing_prediction_rows.csv")
    delta = read_csv(root / "path_editing_controlled_deltas.csv")
    stability = read_csv(root / "stability" / "stability_controlled_deltas.csv")
    rows = []
    wanted_pred = [
        ("standard_success_bool", "agop_top1"),
        ("robust_success_bool", "agop_top1"),
        ("fragile_success_bool", "both"),
        ("locality_damage_bool", "both"),
    ]
    for outcome, feature_set in wanted_pred:
        row = find_row(pred, outcome=outcome, feature_set=feature_set, available="True")
        if row:
            rows.append(
                {
                    "block": "path_only",
                    "outcome": outcome,
                    "comparison_or_features": feature_set,
                    "n": row.get("n"),
                    "auroc": f4(row.get("cv_auroc")),
                    "ap": f4(row.get("cv_average_precision")),
                    "delta_auroc": "",
                    "delta_ap": "",
                    "delta_ci": "",
                    "source_file": str((root / "path_editing_prediction_rows.csv").relative_to(ROOT)),
                }
            )
    wanted_delta = [
        ("fragile_success_bool", "diagnostic+agop_top1_over_diagnostic"),
        ("fragile_success_bool", "diagnostic+both_over_diagnostic"),
        ("standard_success_bool", "diagnostic+agop_top1_over_diagnostic"),
        ("robust_success_bool", "diagnostic+agop_top1_over_diagnostic"),
        ("locality_damage_bool", "diagnostic+both_over_diagnostic"),
    ]
    for outcome, comparison in wanted_delta:
        row = find_row(delta, outcome=outcome, comparison=comparison)
        if row:
            rows.append(
                {
                    "block": "controlled_delta",
                    "outcome": outcome,
                    "comparison_or_features": comparison,
                    "n": row.get("n"),
                    "auroc": f4(row.get("plus_auroc")),
                    "ap": f4(row.get("plus_ap")),
                    "delta_auroc": f4(row.get("delta_auroc")),
                    "delta_ap": f4(row.get("delta_ap")),
                    "delta_ci": "",
                    "source_file": str((root / "path_editing_controlled_deltas.csv").relative_to(ROOT)),
                }
            )
    wanted_stability = [
        ("fragile_success_bool", "diagnostic+agop_top1_over_diagnostic", "auroc"),
        ("fragile_success_bool", "diagnostic+agop_top1_over_diagnostic", "ap"),
        ("robust_success_bool", "diagnostic+agop_top1_over_diagnostic", "auroc"),
        ("locality_damage_bool", "diagnostic+both_over_diagnostic", "auroc"),
    ]
    for outcome, comparison, metric in wanted_stability:
        row = find_row(stability, outcome=outcome, comparison=comparison, metric=metric)
        if row:
            rows.append(
                {
                    "block": "stability_delta",
                    "outcome": outcome,
                    "comparison_or_features": f"{comparison}:{metric}",
                    "n": "",
                    "auroc": "",
                    "ap": "",
                    "delta_auroc": f4(row.get("delta_observed")) if metric == "auroc" else "",
                    "delta_ap": f4(row.get("delta_observed")) if metric == "ap" else "",
                    "delta_ci": f"[{f4(row.get('delta_ci_low'))}, {f4(row.get('delta_ci_high'))}]",
                    "source_file": str((root / "stability" / "stability_controlled_deltas.csv").relative_to(ROOT)),
                }
            )
    return rows


def case_audit_table() -> List[Dict[str, Any]]:
    path = (
        RESULTS
        / "commonsenseqa_balanced_followup_144"
        / "rome"
        / "path_editing_transfer"
        / "case_audit"
        / "path_editing_case_audit_buckets.csv"
    )
    table = read_csv(path)
    rows = []
    for bucket in ["risk_0", "risk_5_plus", "clean_0", "clean_3_plus"]:
        row = find_row(table, bucket=bucket)
        if not row:
            continue
        rows.append(
            {
                "bucket": bucket,
                "n": row.get("n_records"),
                "standard_success": f4(row.get("standard_success_rate")),
                "robust_success": f4(row.get("robust_success_rate")),
                "fragile_success": f4(row.get("fragile_success_rate")),
                "locality_damage": f4(row.get("locality_damage_rate")),
                "rewrite_margin_mean": f4(row.get("rewrite_margin_delta_mean")),
                "source_file": str(path.relative_to(ROOT)),
            }
        )
    return rows


def minimum_strength_table() -> List[Dict[str, Any]]:
    path = (
        RESULTS
        / "heldout_100_dense_alpha_baselines"
        / "minimum_effective_strength"
        / "minimum_strength_method_summary.csv"
    )
    if not path.exists():
        return []
    table = read_csv(path)
    rows = []
    for method in ["mean_difference", "logistic", "linear", "random"]:
        row = find_row(table, control_method=method)
        if not row:
            continue
        rows.append(
            {
                "dataset": row.get("dataset_label"),
                "method": method,
                "n": row.get("n_records"),
                "target_any": f4(row.get("target_any_rate")),
                "clean_exists": f4(row.get("clean_window_exists_rate")),
                "strict_clean_path": f4(row.get("strict_clean_path_rate")),
                "damage_first": f4(row.get("damage_first_rate")),
                "unstable": f4(row.get("unstable_rate")),
                "target_onset_med": f4(row.get("target_onset_median")),
                "best_clean_med": f4(row.get("best_clean_strength_median_all_clean")),
                "strict_clean_onset_med": f4(row.get("target_onset_median_strict_clean")),
                "damage_first_damage_onset_med": f4(row.get("damage_onset_median_damage_first")),
                "source_file": str(path.relative_to(ROOT)),
            }
        )
    return rows


def agop_geometry_diagnostics_table() -> List[Dict[str, Any]]:
    specs = [
        (
            RESULTS
            / "commonsense_500_agop_features"
            / "prediction_analysis"
            / "agop_prediction_summary.json",
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
            "agop_neighbor_overlap",
            [
                ("neighbor_damage", "neighbor_overlap_only"),
                ("clean_success", "clean_success_overlap_only"),
                ("neighbor_damage_with_method_coef", "neighbor_overlap_only_with_method_coef"),
                ("clean_success_with_method_coef", "clean_success_overlap_only_with_method_coef"),
            ],
        ),
    ]
    rows = []
    for path, feature_family, keys in specs:
        if not path.exists():
            continue
        summary = read_json(path)
        for outcome, key in keys:
            model = summary.get(key, {})
            if not model.get("available"):
                continue
            rows.append(
                {
                    "dataset": "commonsense_500",
                    "feature_family": feature_family,
                    "outcome": outcome,
                    "n": model.get("n"),
                    "positive_rate": f4(model.get("positive_rate")),
                    "cv_auroc": f4(model.get("cv_auroc")),
                    "cv_ap": f4(model.get("cv_average_precision")),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return rows


def figure_index_table() -> List[Dict[str, Any]]:
    entries = [
        (
            "Figure 1",
            "Endpoint success vs clean intervention",
            [
                "pml/results/path_geometry_audit/plots/commonsense_3000_baselines_target_success_rate.png",
                "pml/results/path_geometry_audit/plots/commonsense_3000_baselines_clean_success_rate.png",
                "pml/results/path_geometry_audit/plots/commonsense_3000_baselines_path_type_distribution.png",
            ],
        ),
        (
            "Figure 2",
            "AGOP top-k boundary result",
            [
                "pml/results/path_geometry_audit/plots/heldout_144_agop_direct_target_success_rate.png",
                "pml/results/path_geometry_audit/plots/heldout_144_agop_direct_neighbor_damage_rate.png",
                "pml/results/path_geometry_audit/plots/heldout_144_agop_direct_path_type_distribution.png",
            ],
        ),
        (
            "Figure 3",
            "Onset geometry",
            [
                "pml/results/path_geometry_audit/plots/commonsense_3000_baselines_target_vs_damage_onset.png",
                "pml/results/path_geometry_audit/plots/commonsense_3000_baselines_damage_minus_target_onset_hist.png",
                "pml/results/path_geometry_audit/plots/heldout_144_agop_direct_target_vs_damage_onset.png",
                "pml/results/path_geometry_audit/plots/heldout_144_agop_direct_damage_minus_target_onset_hist.png",
            ],
        ),
        (
            "Figure 4",
            "Dense-alpha path stability",
            [
                "pml/results/heldout_100_dense_alpha_baselines/path_geometry_audit/plots/heldout_100_dense_alpha_baselines_target_success_rate.png",
                "pml/results/heldout_100_dense_alpha_baselines/path_geometry_audit/plots/heldout_100_dense_alpha_baselines_clean_success_rate.png",
                "pml/results/heldout_100_dense_alpha_baselines/path_geometry_audit/plots/heldout_100_dense_alpha_baselines_path_type_distribution.png",
            ],
        ),
        (
            "Table 4",
            "Path-conditioned minimum effective strength",
            [
                "pml/results/heldout_100_dense_alpha_baselines/minimum_effective_strength/minimum_strength_method_summary.csv",
                "pml/results/heldout_100_dense_alpha_baselines/minimum_effective_strength/MINIMUM_EFFECTIVE_STRENGTH.md",
            ],
        ),
        (
            "Table 5",
            "Path-to-editing transfer",
            [
                "pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/path_editing_controlled_deltas.csv",
                "pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/stability/stability_controlled_deltas.csv",
            ],
        ),
        (
            "Table 6",
            "Case audit / failure taxonomy",
            [
                "pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/case_audit/path_editing_case_audit_buckets.csv",
                "pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer/case_audit/path_editing_case_audit_cases.csv",
            ],
        ),
        (
            "Table 7",
            "AGOP/RFM diagnostic geometry",
            [
                "pml/results/paper_tables/agop_geometry_diagnostics.csv",
                "pml/results/commonsense_500_agop_features/prediction_analysis/agop_prediction_summary.json",
                "pml/results/commonsense_500_agop_neighbor_overlap/prediction_analysis/overlap_prediction_summary.json",
            ],
        ),
    ]
    rows = []
    for label, purpose, paths in entries:
        missing = [path for path in paths if not (ROOT / path).exists()]
        rows.append(
            {
                "label": label,
                "purpose": purpose,
                "files": "\n".join(paths),
                "status": "ready" if not missing else "missing",
                "missing": "\n".join(missing),
            }
        )
    return rows


def feature_audit_table() -> List[Dict[str, Any]]:
    path = RESULTS / "path_geometry_audit" / "feature_audit" / "path_prediction_top_coefficients.csv"
    if not path.exists():
        return []
    table = read_csv(path)
    wanted = [
        ("pooled", "baseline", "clean_window_label", False, "path_geometry"),
        ("pooled", "baseline", "usable_path_label", False, "path_geometry"),
        ("pooled", "baseline", "damage_first_label", False, "path_geometry"),
        ("pooled", "baseline", "no_effect_label", False, "path_geometry"),
        ("heldout_144_agop_direct", "agop_spectrum", "usable_path_label", True, "path_geometry_with_method"),
        ("heldout_144_agop_direct", "agop_spectrum", "damage_first_label", True, "path_geometry_with_method"),
        ("heldout_144_agop_direct", "agop_topvec", "damage_first_label", True, "path_geometry_with_method"),
        ("pooled", "baseline", "clean_window_label", True, "heldout_100_dense_alpha_baselines"),
        ("pooled", "baseline", "usable_path_label", True, "heldout_100_dense_alpha_baselines"),
        ("pooled", "baseline", "damage_first_label", True, "heldout_100_dense_alpha_baselines"),
        ("pooled", "baseline", "no_effect_label", True, "heldout_100_dense_alpha_baselines"),
    ]
    rows = []
    for dataset, feature_group, outcome, include_method, summary_source in wanted:
        subset = [
            row
            for row in table
            if row.get("dataset") == dataset
            and row.get("feature_group") == feature_group
            and row.get("outcome") == outcome
            and str(row.get("include_method")) == str(include_method)
            and row.get("summary_source", "") == summary_source
            and int(row.get("rank", "999")) <= 3
        ]
        for row in sorted(subset, key=lambda r: int(r.get("rank", "999"))):
            rows.append(
                {
                    "source": summary_source,
                    "dataset": dataset,
                    "feature_group": feature_group,
                    "outcome": outcome,
                    "include_method": include_method,
                    "rank": row.get("rank"),
                    "feature": row.get("feature"),
                    "feature_family": row.get("feature_family"),
                    "coefficient": row.get("coefficient"),
                    "cv_auroc": row.get("cv_auroc"),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return rows


def write_markdown(tables: Dict[str, List[Dict[str, Any]]]) -> None:
    lines = [
        "# PML Paper Tables",
        "",
        "Generated from current result files. These are paper-facing tables; source CSVs remain the authoritative artifacts.",
        "",
    ]
    specs = [
        ("Path Summary", "path_summary", ["dataset", "method", "n", "target_any", "neighbor_damage_any", "clean_window", "damage_first", "no_effect", "collapse"]),
        ("Path Prediction", "path_prediction", ["setting", "dataset", "feature_group", "outcome", "n", "positive_rate", "cv_auroc", "cv_ap"]),
        ("Minimum Strength", "minimum_strength", ["dataset", "method", "n", "target_any", "clean_exists", "strict_clean_path", "damage_first", "unstable", "target_onset_med", "best_clean_med", "strict_clean_onset_med", "damage_first_damage_onset_med"]),
        ("AGOP Geometry Diagnostics", "agop_geometry_diagnostics", ["dataset", "feature_family", "outcome", "n", "positive_rate", "cv_auroc", "cv_ap"]),
        ("Editing Transfer", "editing_transfer", ["block", "outcome", "comparison_or_features", "n", "auroc", "ap", "delta_auroc", "delta_ap", "delta_ci"]),
        ("Case Audit", "case_audit", ["bucket", "n", "standard_success", "robust_success", "fragile_success", "locality_damage", "rewrite_margin_mean"]),
        ("Feature Audit", "feature_audit", ["source", "dataset", "feature_group", "outcome", "rank", "feature", "feature_family", "coefficient", "cv_auroc"]),
        ("Figure Index", "figure_index", ["label", "purpose", "status"]),
    ]
    for title, key, columns in specs:
        lines.extend([f"## {title}", ""])
        lines.append(markdown_table(columns, [[row.get(col, "") for col in columns] for row in tables[key]]))
        lines.append("")
    (OUT_DIR / "PML_PAPER_TABLES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "path_summary": path_summary_table(),
        "path_prediction": path_prediction_table(),
        "minimum_strength": minimum_strength_table(),
        "agop_geometry_diagnostics": agop_geometry_diagnostics_table(),
        "editing_transfer": editing_transfer_table(),
        "case_audit": case_audit_table(),
        "feature_audit": feature_audit_table(),
        "figure_index": figure_index_table(),
    }
    for name, rows in tables.items():
        write_csv(OUT_DIR / f"{name}.csv", rows)
    write_markdown(tables)
    summary = {
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "tables": {name: len(rows) for name, rows in tables.items()},
    }
    (OUT_DIR / "paper_tables_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
