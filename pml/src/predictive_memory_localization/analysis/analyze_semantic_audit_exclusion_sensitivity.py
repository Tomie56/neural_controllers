from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset import (
    atomic_write_csv,
    atomic_write_json,
)
from predictive_memory_localization.analysis.build_layer_pilot_exclusion_dataset import (
    paired_outcome_comparisons,
)
from predictive_memory_localization.analysis.run_activation_space_pml_main_prediction import (
    outcome_summary,
    run_tasks,
    summarize_results,
)


DEFAULT_ROOT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b"
)
DEFAULT_AUDIT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "benchmark_semantic_audit_20260722/semantic_audit_judgments.csv"
)
TARGETS = [
    "later_target_any_path",
    "later_clean_any_path",
    "later_neighbor_damage_any_path",
    "later_capability_damage_any_path",
]
SCENARIOS = {
    "exclude_fail": {"fail"},
    "exclude_nonpass": {"minor", "fail"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute PML outcomes and record-held-out prediction after semantic-audit exclusions."
    )
    parser.add_argument(
        "--source-dataset-dir",
        default=str(DEFAULT_ROOT / "prediction_dataset_strict_later"),
    )
    parser.add_argument("--audit-judgments", default=str(DEFAULT_AUDIT))
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_ROOT / "semantic_audit_exclusion_sensitivity"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--rf-n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame, columns: List[str]) -> str:
    subset = frame[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in subset.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_filtered_dataset(
    source: pd.DataFrame,
    source_manifest: Dict[str, Any],
    audit: pd.DataFrame,
    excluded_labels: set[str],
    scenario_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, set[str]]:
    excluded_ids = set(
        audit.loc[audit["overall"].astype(str).str.lower().isin(excluded_labels), "record_id"]
        .astype(str)
        .tolist()
    )
    source_ids = set(source["record_id"].astype(str))
    missing = excluded_ids - source_ids
    if missing:
        raise ValueError(f"Audited records missing from prediction dataset: {sorted(missing)}")
    filtered = source[~source["record_id"].astype(str).isin(excluded_ids)].copy()
    filtered = filtered.reset_index(drop=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = scenario_dir / "pml_path_prediction_dataset.csv"
    manifest_path = scenario_dir / "prediction_dataset_manifest.json"
    manifest = {
        **source_manifest,
        "source_dataset_path": str(
            (Path(args.source_dataset_dir) / "pml_path_prediction_dataset.csv").resolve()
        ),
        "semantic_audit_path": str(Path(args.audit_judgments).resolve()),
        "semantic_exclusion_labels": sorted(excluded_labels),
        "semantic_excluded_record_ids": sorted(excluded_ids),
        "n_excluded_records": len(excluded_ids),
        "n_records": int(filtered["record_id"].nunique()),
        "n_rows": int(len(filtered)),
    }
    atomic_write_csv(dataset_path, filtered)
    atomic_write_json(manifest_path, manifest)
    return filtered, excluded_ids


def run_prediction(
    frame: pd.DataFrame,
    manifest: Dict[str, Any],
    analysis_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results_path = analysis_dir / "fold_results.jsonl"
    if args.overwrite:
        results_path.unlink(missing_ok=True)
        (analysis_dir / "prediction_summary.csv").unlink(missing_ok=True)
        (analysis_dir / "outcome_summary.csv").unlink(missing_ok=True)
    task_args = Namespace(
        cohorts=["all"],
        splits=["record"],
        targets=TARGETS,
        feature_sets=["B+M+L+R"],
        models=["random_forest"],
        n_splits=args.n_splits,
        rf_n_jobs=args.rf_n_jobs,
        seed=args.seed,
        min_positive=20,
    )
    run_tasks(frame, manifest, task_args, results_path)
    summary = summarize_results(results_path, analysis_dir, args.seed)
    outcome_summary(frame, analysis_dir, TARGETS)
    return summary


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dataset_dir)
    source = pd.read_csv(source_dir / "pml_path_prediction_dataset.csv")
    source["record_id"] = source["record_id"].astype(str)
    if "method" in source.columns:
        source = source[~source["method"].eq("matched_norm_random")].reset_index(drop=True)
    source_manifest = json.loads(
        (source_dir / "prediction_dataset_manifest.json").read_text(encoding="utf-8")
    )
    audit = pd.read_csv(args.audit_judgments)
    audit["record_id"] = audit["record_id"].astype(str)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_path = (
        DEFAULT_ROOT
        / "prediction_analysis_strict_later_weak_response_baselines"
        / "prediction_summary.csv"
    )
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[
        baseline["cohort"].eq("all")
        & baseline["split"].eq("record")
        & baseline["feature_set"].eq("B+M+L+R")
        & baseline["model"].eq("random_forest")
        & baseline["target"].isin(TARGETS)
    ][["target", "positive_rate_mean", "auroc_mean", "average_precision_mean"]].copy()
    baseline = baseline.rename(
        columns={
            "positive_rate_mean": "full_prevalence",
            "auroc_mean": "full_auroc",
            "average_precision_mean": "full_ap",
        }
    )

    prediction_rows: List[pd.DataFrame] = []
    outcome_rows: List[pd.DataFrame] = []
    scenario_manifest: Dict[str, Any] = {}
    for scenario, labels in SCENARIOS.items():
        scenario_dir = output_root / scenario / "prediction_dataset"
        filtered, excluded_ids = build_filtered_dataset(
            source, source_manifest, audit, labels, scenario_dir, args
        )
        manifest = json.loads(
            (scenario_dir / "prediction_dataset_manifest.json").read_text(encoding="utf-8")
        )
        prediction = run_prediction(
            filtered, manifest, output_root / scenario / "prediction_analysis", args
        )
        prediction = prediction[
            prediction["cohort"].eq("all")
            & prediction["split"].eq("record")
            & prediction["feature_set"].eq("B+M+L+R")
            & prediction["model"].eq("random_forest")
            & prediction["target"].isin(TARGETS)
        ][["target", "positive_rate_mean", "auroc_mean", "average_precision_mean"]].copy()
        prediction.insert(0, "scenario", scenario)
        prediction.insert(1, "n_records", int(filtered["record_id"].nunique()))
        prediction_rows.append(prediction)

        comparisons = paired_outcome_comparisons(
            filtered,
            random_method="random",
            reps=args.bootstrap_reps,
            seed=args.seed,
        )
        comparisons.insert(0, "scenario", scenario)
        comparisons.to_csv(output_root / scenario / "paired_outcome_vs_random.csv", index=False)
        outcome_rows.append(comparisons)
        scenario_manifest[scenario] = {
            "excluded_labels": sorted(labels),
            "n_excluded_records": len(excluded_ids),
            "n_remaining_records": int(filtered["record_id"].nunique()),
        }

    prediction = pd.concat(prediction_rows, ignore_index=True).merge(
        baseline, on="target", how="left", validate="many_to_one"
    )
    prediction["delta_auroc_vs_full"] = prediction["auroc_mean"] - prediction["full_auroc"]
    prediction["delta_ap_vs_full"] = (
        prediction["average_precision_mean"] - prediction["full_ap"]
    )
    outcomes = pd.concat(outcome_rows, ignore_index=True)
    prediction.to_csv(output_root / "semantic_exclusion_prediction_summary.csv", index=False)
    outcomes.to_csv(output_root / "semantic_exclusion_outcome_summary.csv", index=False)

    focus_outcomes = outcomes[
        outcomes["layer"].eq(-21)
        & outcomes["target"].isin(["later_target_any_path", "later_clean_any_path"])
    ].copy()
    focus_outcomes["target"] = focus_outcomes["target"].map(
        {
            "later_target_any_path": "Target-any",
            "later_clean_any_path": "Clean-any",
        }
    )
    prediction_display = prediction.copy()
    prediction_display["target"] = prediction_display["target"].map(
        {
            "later_target_any_path": "Target-any",
            "later_clean_any_path": "Clean-any",
            "later_neighbor_damage_any_path": "Neighbor damage",
            "later_capability_damage_any_path": "Capability damage",
        }
    )
    report = [
        "# Semantic-Audit Exclusion Sensitivity",
        "",
        "The audit uses dataset-stratified independent human-expert judgments on 108 records. "
        "This analysis removes the audited records labeled fail, and then all records labeled "
        "minor or fail, without altering any stored GPU trajectory.",
        "",
        "## Record-held-out complete predictor",
        "",
        markdown_table(
            prediction_display,
            [
                "scenario",
                "n_records",
                "target",
                "positive_rate_mean",
                "auroc_mean",
                "average_precision_mean",
                "delta_auroc_vs_full",
                "delta_ap_vs_full",
            ],
        ),
        "",
        "## Learned-minus-random outcomes at block 7 (implementation layer -21)",
        "",
        markdown_table(
            focus_outcomes,
            [
                "scenario",
                "method",
                "target",
                "n_records",
                "difference",
                "ci_low",
                "ci_high",
            ],
        ),
        "",
        "The exclusion sets are sensitivity analyses over known audited records, not estimates "
        "of the full benchmark error rate.",
        "",
    ]
    (output_root / "SEMANTIC_AUDIT_EXCLUSION_SENSITIVITY.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    atomic_write_json(
        output_root / "semantic_audit_exclusion_manifest.json",
        {
            "source_dataset_dir": str(source_dir.resolve()),
            "audit_judgments": str(Path(args.audit_judgments).resolve()),
            "source_records": int(source["record_id"].nunique()),
            "audit_records": int(audit["record_id"].nunique()),
            "scenarios": scenario_manifest,
            "targets": TARGETS,
            "feature_set": "B+M+L+R",
            "model": "random_forest",
            "split": "record",
            "seed": args.seed,
        },
    )
    print((output_root / "SEMANTIC_AUDIT_EXCLUSION_SENSITIVITY.md").read_text())


if __name__ == "__main__":
    main()
