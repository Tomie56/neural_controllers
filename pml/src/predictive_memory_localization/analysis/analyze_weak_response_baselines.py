from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset import (
    atomic_write_csv,
    atomic_write_json,
)


DEFAULT_ROOT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b"
)
TARGETS = [
    "later_suppression_path",
    "later_enhancement_path",
    "later_target_any_path",
    "later_neighbor_damage_any_path",
    "later_capability_damage_any_path",
    "later_clean_suppression_path",
    "later_clean_enhancement_path",
    "later_clean_any_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate training-free weak-response baselines for strict later outcomes."
    )
    parser.add_argument(
        "--dataset-dir", default=str(DEFAULT_ROOT / "prediction_dataset_strict_later")
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_ROOT / "weak_response_simple_baselines")
    )
    parser.add_argument("--splits", nargs="+", default=["record", "dataset", "domain"])
    parser.add_argument("--cohorts", nargs="+", default=["all", "learned", "rfm"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cohort_frame(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == "all":
        return frame[~frame["method"].eq("matched_norm_random")]
    if cohort == "learned":
        return frame[~frame["method"].isin(["random", "matched_norm_random"])]
    if cohort == "rfm":
        return frame[frame["method"].astype(str).str.startswith("rfm_agop_")]
    raise ValueError(f"Unknown cohort: {cohort}")


def split_groups(frame: pd.DataFrame, split: str) -> pd.Series:
    if split == "record":
        return frame["record_id"].astype(str)
    if split in {"dataset", "domain"}:
        return frame[split].astype(str)
    raise ValueError(f"Unknown split: {split}")


def evidence_components(frame: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, pd.Series]:
    target_negative = -pd.to_numeric(frame["early_target_delta_negative"], errors="coerce") / thresholds["target"]
    target_positive = pd.to_numeric(frame["early_target_delta_positive"], errors="coerce") / thresholds["target"]
    neighbor_negative = -pd.to_numeric(frame["early_neighbor_delta_negative"], errors="coerce") / thresholds["neighbor"]
    neighbor_positive = -pd.to_numeric(frame["early_neighbor_delta_positive"], errors="coerce") / thresholds["neighbor"]
    capability_negative = -pd.to_numeric(frame["early_capability_delta_negative"], errors="coerce") / thresholds["capability"]
    capability_positive = -pd.to_numeric(frame["early_capability_delta_positive"], errors="coerce") / thresholds["capability"]
    collateral_negative = np.maximum(neighbor_negative, capability_negative)
    collateral_positive = np.maximum(neighbor_positive, capability_positive)
    selective_negative = target_negative - np.maximum(collateral_negative, 0.0)
    selective_positive = target_positive - np.maximum(collateral_positive, 0.0)
    return {
        "target_negative": target_negative,
        "target_positive": target_positive,
        "target_any": np.maximum(target_negative, target_positive),
        "neighbor_any": np.maximum(neighbor_negative, neighbor_positive),
        "capability_any": np.maximum(capability_negative, capability_positive),
        "selective_negative": selective_negative,
        "selective_positive": selective_positive,
        "selective_any": np.maximum(selective_negative, selective_positive),
    }


def scores_for_target(
    frame: pd.DataFrame, target: str, thresholds: Dict[str, float]
) -> Dict[str, tuple[np.ndarray, float]]:
    component = evidence_components(frame, thresholds)
    target_map = {
        "later_suppression_path": "target_negative",
        "later_enhancement_path": "target_positive",
        "later_target_any_path": "target_any",
        "later_neighbor_damage_any_path": "neighbor_any",
        "later_capability_damage_any_path": "capability_any",
        "later_clean_suppression_path": "selective_negative",
        "later_clean_enhancement_path": "selective_positive",
        "later_clean_any_path": "selective_any",
    }
    primary = np.asarray(component[target_map[target]], dtype=float)
    results = {
        "signed_weak_response": (primary, 1.0),
        "weak_threshold_persistence": ((primary >= 1.0).astype(float), 0.5),
    }
    if target.startswith("later_clean_"):
        direction = {
            "later_clean_suppression_path": "target_negative",
            "later_clean_enhancement_path": "target_positive",
            "later_clean_any_path": "target_any",
        }[target]
        results["target_only_weak_response"] = (
            np.asarray(component[direction], dtype=float),
            1.0,
        )
    return results


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    finite = np.isfinite(scores)
    y_true = y_true[finite]
    scores = scores[finite]
    predictions = (scores >= threshold).astype(int)
    return {
        "n_test": int(len(y_true)),
        "test_positive_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def bootstrap_mean_ci(values: Sequence[float], reps: int, seed: int) -> tuple[float, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(reps, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize(folds: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    group_columns = ["cohort", "split", "target", "baseline"]
    rows: List[Dict[str, object]] = []
    for group_index, (keys, group) in enumerate(folds.groupby(group_columns, dropna=False)):
        row: Dict[str, object] = dict(zip(group_columns, keys))
        row["n_folds"] = len(group)
        row["n_test_total"] = int(group["n_test"].sum())
        row["positive_rate_mean"] = float(group["test_positive_rate"].mean())
        for metric_name in ["auroc", "average_precision", "balanced_accuracy", "f1"]:
            values = group[metric_name].astype(float).tolist()
            row[f"{metric_name}_mean"] = float(np.mean(values))
            low, high = bootstrap_mean_ci(values, reps, seed + group_index)
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "weak_response_baseline_folds.csv"
    summary_path = output_dir / "weak_response_baseline_summary.csv"
    report_path = output_dir / "WEAK_RESPONSE_BASELINE_REPORT.md"
    manifest_path = output_dir / "weak_response_baseline_manifest.json"
    if all(path.exists() for path in [fold_path, summary_path, report_path, manifest_path]) and not args.overwrite:
        print(manifest_path.read_text(encoding="utf-8"))
        return

    frame = pd.read_csv(dataset_dir / "pml_path_prediction_dataset.csv")
    manifest = json.loads((dataset_dir / "prediction_dataset_manifest.json").read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in manifest["thresholds"].items()}
    rows: List[Dict[str, object]] = []
    for cohort in args.cohorts:
        cohort_data = cohort_frame(frame, cohort).reset_index(drop=True)
        for split in args.splits:
            groups = split_groups(cohort_data, split)
            n_splits = min(args.n_splits, groups.nunique())
            splitter = GroupKFold(n_splits=n_splits)
            for fold, (_, test_indices) in enumerate(splitter.split(cohort_data, groups=groups)):
                test = cohort_data.iloc[test_indices]
                for target in TARGETS:
                    y_true = test[target].astype(int).to_numpy()
                    if np.unique(y_true).size < 2:
                        continue
                    for baseline, (scores, threshold) in scores_for_target(
                        test, target, thresholds
                    ).items():
                        rows.append(
                            {
                                "cohort": cohort,
                                "split": split,
                                "target": target,
                                "baseline": baseline,
                                "fold": fold,
                                **metrics(y_true, scores, threshold),
                            }
                        )
    folds = pd.DataFrame(rows)
    summary = summarize(folds, args.bootstrap_reps, args.seed)
    atomic_write_csv(fold_path, folds)
    atomic_write_csv(summary_path, summary)
    run_manifest = {
        "dataset_dir": str(dataset_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_rows": int(len(frame)),
        "n_records": int(frame["record_id"].nunique()),
        "thresholds": thresholds,
        "splits": args.splits,
        "cohorts": args.cohorts,
        "n_splits": args.n_splits,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    atomic_write_json(manifest_path, run_manifest)

    focus = summary[
        summary["split"].eq("record") & summary["baseline"].eq("signed_weak_response")
    ][
        [
            "cohort",
            "target",
            "positive_rate_mean",
            "auroc_mean",
            "auroc_ci_low",
            "auroc_ci_high",
            "average_precision_mean",
        ]
    ]
    report = [
        "# Training-Free Weak-Response Baselines",
        "",
        "Scores are threshold-normalized using the frozen random-direction null thresholds.",
        "The signed baseline ranks later outcomes using only the matching $|alpha|=0.1$ response.",
        "Clean-path scores subtract the larger weak neighbor/capability damage signal.",
        "",
        markdown_table(focus),
        "",
        "Full split/cohort/baseline results are stored in `weak_response_baseline_summary.csv`.",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
