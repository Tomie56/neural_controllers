from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset import atomic_write_csv


DEFAULT_ROOT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine prevalence, AUROC, AP, and uncertainty for supplemental analyses."
    )
    parser.add_argument("--main-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT / "cpu_supplemental_evidence"))
    return parser.parse_args()


def learned_rows(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["split"].eq("record")].copy()
    return pd.DataFrame(
        {
            "source": source,
            "cohort": frame["cohort"],
            "split": frame["split"],
            "target": frame["target"],
            "variant": frame["feature_set"],
            "model": frame["model"],
            "prevalence": frame["positive_rate_mean"],
            "auroc": frame["auroc_mean"],
            "auroc_ci_low": frame["auroc_ci_low"],
            "auroc_ci_high": frame["auroc_ci_high"],
            "average_precision": frame["average_precision_mean"],
            "ap_ci_low": frame["average_precision_ci_low"],
            "ap_ci_high": frame["average_precision_ci_high"],
        }
    )


def heuristic_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["split"].eq("record")].copy()
    return pd.DataFrame(
        {
            "source": "full_3000_training_free",
            "cohort": frame["cohort"],
            "split": frame["split"],
            "target": frame["target"],
            "variant": frame["baseline"],
            "model": "none",
            "prevalence": frame["positive_rate_mean"],
            "auroc": frame["auroc_mean"],
            "auroc_ci_low": frame["auroc_ci_low"],
            "auroc_ci_high": frame["auroc_ci_high"],
            "average_precision": frame["average_precision_mean"],
            "ap_ci_low": frame["average_precision_ci_low"],
            "ap_ci_high": frame["average_precision_ci_high"],
        }
    )


def main() -> None:
    args = parse_args()
    root = Path(args.main_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: List[pd.DataFrame] = [
        learned_rows(
            root / "prediction_analysis_strict_later_excluding_pilot500/prediction_summary.csv",
            "pilot500_exclusion_2500",
        ),
        learned_rows(
            root / "prediction_analysis_strict_later_weak_response_baselines/prediction_summary.csv",
            "full_3000_learned",
        ),
        heuristic_rows(
            root / "weak_response_simple_baselines/weak_response_baseline_summary.csv"
        ),
    ]
    combined = pd.concat(sources, ignore_index=True).sort_values(
        ["source", "cohort", "target", "variant", "model"]
    )
    atomic_write_csv(output_dir / "cpu_supplemental_prediction_metrics.csv", combined)
    focus = combined[
        combined["cohort"].eq("all")
        & combined["target"].isin(
            ["later_target_any_path", "later_clean_any_path", "later_neighbor_damage_any_path"]
        )
    ]
    lines = [
        "# CPU Supplemental Prediction Evidence",
        "",
        "Every row reports label prevalence, AUROC, average precision, and fold-bootstrap 95% confidence intervals.",
        "",
        "```csv",
        focus.to_csv(index=False).strip(),
        "```",
        "",
        "The complete table is stored in `cpu_supplemental_prediction_metrics.csv`.",
        "",
    ]
    (output_dir / "CPU_SUPPLEMENTAL_EVIDENCE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
