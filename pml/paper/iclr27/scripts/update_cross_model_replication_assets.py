#!/usr/bin/env python3
"""Track dimension-corrected RMS replications without publishing partial claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parents[2]
RESULT_BASE = REPO_ROOT / "pml/results/fresh_multidomain_3000/stage_activation_pml"
RMS_ROOT = RESULT_BASE / "replication_500_residual_rms_v2"
SCALE_MANIFEST = (
    RESULT_BASE
    / "scale_calibration_residual_rms_v2/residual_rms_reference_scale.json"
)
DATA_PATH = PAPER_ROOT / "data/cross_model_replication_summary.csv"
REPORT_PATH = PAPER_ROOT / "data/CROSS_MODEL_REPLICATION_REPORT.md"
TABLE_PATH = PAPER_ROOT / "tables/cross_model_replication.tex"
GENERATED_PATHS = [
    PAPER_ROOT / "sections/cross_model_results_generated.tex",
    PAPER_ROOT / "sections/cross_model_abstract_generated.tex",
    PAPER_ROOT / "sections/cross_model_discussion_generated.tex",
    PAPER_ROOT / "sections/cross_model_limitations_generated.tex",
    PAPER_ROOT / "sections/cross_model_conclusion_generated.tex",
]

METHODS = ["random", "mean_difference", "logistic", "rfm_agop_top1"]
MODEL_SPECS = [
    ("qwen3_1_7b", "Qwen3-1.7B-Base", 28),
    ("qwen3_5_2b_base", "Qwen3.5-2B-Base", 24),
    ("ministral_3_3b_base_2512", "Ministral-3-3B-Base", 26),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def load_scale_manifest() -> dict:
    if not SCALE_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing scale manifest: {SCALE_MANIFEST}")
    payload = json.loads(SCALE_MANIFEST.read_text(encoding="utf-8"))
    expected = "residual_rms_reference_matching_dimension_corrected_v2"
    if payload.get("protocol") != expected:
        raise ValueError(
            f"Unexpected RMS protocol: {payload.get('protocol')!r}; expected {expected!r}"
        )
    return payload


def summary_rows() -> pd.DataFrame:
    manifest = load_scale_manifest()
    rows = []
    for model_key, model_label, depth in MODEL_SPECS:
        model_root = RMS_ROOT / model_key
        summary_path = (
            model_root
            / "outcome_analysis_later_strength/strict_later_outcome_summary.csv"
        )
        summary = pd.read_csv(summary_path) if summary_path.is_file() else None
        layer_rows = list(manifest["models"][model_key]["layers"].values())
        layer_rows.sort(key=lambda row: int(row["reference_layer"]))
        for scale_row in layer_rows:
            layer = int(scale_row["target_layer"])
            display_layer = depth + layer
            for method in METHODS:
                matching = None
                if summary is not None:
                    method_rows = summary[summary["scope"].eq("method_layer")].copy()
                    method_rows["layer_numeric"] = pd.to_numeric(
                        method_rows["layer"], errors="coerce"
                    )
                    matching = method_rows[
                        method_rows["method"].eq(method)
                        & method_rows["layer_numeric"].eq(layer)
                    ]
                complete = matching is not None and len(matching) == 1
                row = {
                    "model_key": model_key,
                    "model_label": model_label,
                    "records": 500,
                    "layer": layer,
                    "display_layer": display_layer,
                    "reference_layer": int(scale_row["reference_layer"]),
                    "method": method,
                    "control_scale": float(scale_row["control_scale"]),
                    "summary_path": str(summary_path),
                    "complete": bool(complete),
                    "protocol": manifest["protocol"],
                }
                if complete:
                    for column, value in matching.iloc[0].items():
                        if str(column).endswith("_rate"):
                            row[str(column)] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def render_report(frame: pd.DataFrame) -> str:
    complete = int(frame["complete"].sum())
    total = int(len(frame))
    lines = [
        "# Dimension-Corrected RMS Replication Report",
        "",
        "Protocol: `residual_rms_reference_matching_dimension_corrected_v2`.",
        "Partial paths are execution status only and are never rendered into manuscript claims.",
        "",
        f"Complete method-layer paths: **{complete}/{total}**.",
        "",
        "| Model | Layer | Method | Scale | Complete |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['model_label']} | {int(row['display_layer'])} | "
            f"{row['method']} | {float(row['control_scale']):.6f} | "
            f"{'yes' if bool(row['complete']) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript Gate",
            "",
            "Cross-model Abstract, Discussion, Limitations, Conclusion, and result-table "
            "cells remain empty until all 24 paths are complete and paired comparisons "
            "have been regenerated from the frozen 500-record cohort.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    frame = summary_rows()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    REPORT_PATH.write_text(render_report(frame), encoding="utf-8")
    TABLE_PATH.write_text(
        "% RMS-matched cross-model result rows are rendered only after all 24 paths complete.\n",
        encoding="utf-8",
    )
    for path in GENERATED_PATHS:
        path.write_text("", encoding="utf-8")

    complete = int(frame["complete"].sum())
    total = int(len(frame))
    print(f"Dimension-corrected RMS paths complete: {complete}/{total}")
    print(f"Wrote {DATA_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print("Kept all manuscript cross-model result fragments empty")
    if args.require_complete and complete != total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
