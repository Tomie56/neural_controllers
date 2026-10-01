from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "target_endpoint_correct_rate",
    "target_endpoint_gain_rate",
    "target_endpoint_damage_from_base_rate",
    "target_wrong_contrast_rate",
    "neighbor_endpoint_correct_rate",
    "neighbor_endpoint_damage_from_base_rate",
    "capability_endpoint_correct_rate",
    "capability_endpoint_damage_from_base_rate",
    "collapse_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate endpoint-validation paths into paper-ready CSV summaries.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--selection-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    result_paths = sorted(root_dir.glob("*/layer_*/per_record_endpoint.csv"))
    if not result_paths:
        raise FileNotFoundError(f"No per_record_endpoint.csv files under {root_dir}")

    frames = [pd.read_csv(path) for path in result_paths if path.stat().st_size > 0]
    if not frames:
        raise ValueError(f"Endpoint result files are empty under {root_dir}")
    rows = pd.concat(frames, ignore_index=True)
    selection = pd.read_csv(args.selection_csv)
    selection_columns = [
        column
        for column in [
            "record_id",
            "selection_bucket",
            "target_any_path",
            "clean_any_path",
            "damage_any_path",
            "no_effect_path",
            "has_neighbor_damage_path",
            "has_capability_damage_path",
        ]
        if column in selection.columns
    ]
    rows["record_id"] = rows["record_id"].astype(str)
    selection["record_id"] = selection["record_id"].astype(str)
    rows = rows.merge(selection[selection_columns], on="record_id", how="left", validate="many_to_one")

    available_metrics = [metric for metric in METRICS if metric in rows.columns]
    group_columns = ["selection_bucket", "control_method", "layer", "alpha"]
    summary = (
        rows.groupby(group_columns, dropna=False)
        .agg(n_records=("record_id", "nunique"), **{metric: (metric, "mean") for metric in available_metrics})
        .reset_index()
    )
    overall = (
        rows.groupby(["control_method", "layer", "alpha"], dropna=False)
        .agg(n_records=("record_id", "nunique"), **{metric: (metric, "mean") for metric in available_metrics})
        .reset_index()
    )

    all_rows_path = root_dir / "endpoint_validation_all_rows.csv"
    bucket_summary_path = root_dir / "endpoint_validation_bucket_summary.csv"
    overall_summary_path = root_dir / "endpoint_validation_overall_summary.csv"
    manifest_path = root_dir / "endpoint_validation_analysis_manifest.json"
    report_path = root_dir / "ENDPOINT_VALIDATION_REPORT.md"
    rows.to_csv(all_rows_path, index=False)
    summary.to_csv(bucket_summary_path, index=False)
    overall.to_csv(overall_summary_path, index=False)

    baseline = overall[overall["alpha"].astype(float).abs() < 1e-12]
    manifest = {
        "root_dir": str(root_dir),
        "n_result_paths": len(result_paths),
        "n_records": int(rows["record_id"].nunique()),
        "n_rows": len(rows),
        "methods": sorted(rows["control_method"].dropna().astype(str).unique().tolist()),
        "layers": sorted(rows["layer"].dropna().astype(str).unique().tolist()),
        "alphas": sorted(rows["alpha"].dropna().astype(float).unique().tolist()),
        "baseline_target_endpoint_correct_rate": float(baseline["target_endpoint_correct_rate"].mean())
        if not baseline.empty
        else None,
        "all_rows_csv": str(all_rows_path),
        "bucket_summary_csv": str(bucket_summary_path),
        "overall_summary_csv": str(overall_summary_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# Endpoint Validation Report",
        "",
        f"- Records: `{manifest['n_records']}`",
        f"- Result paths: `{manifest['n_result_paths']}`",
        f"- Methods: `{manifest['methods']}`",
        f"- Layers: `{manifest['layers']}`",
        f"- Alphas: `{manifest['alphas']}`",
        f"- Mean alpha=0 target endpoint accuracy: `{manifest['baseline_target_endpoint_correct_rate']}`",
        "",
        "Detailed results are stored in the bucket and overall summary CSV files.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
