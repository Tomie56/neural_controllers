from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


DIRECTIONS = ("suppression", "enhancement")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a strength selector by held-out dense-grid policy replay."
    )
    parser.add_argument("--dense-train-dataset-dir", required=True)
    parser.add_argument("--validation-dataset-dir", required=True)
    parser.add_argument("--selector-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selector-task", default="multifidelity__MAR__hist_gbdt")
    parser.add_argument("--dense-selector-task", default="dense_only__MAR__hist_gbdt")
    parser.add_argument("--probe-count", type=float, default=2.0)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=113)
    return parser.parse_args()


def read_alpha_dataset(directory: Path) -> pd.DataFrame:
    path = directory / "alpha_level_selector_dataset.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["direction"].isin(DIRECTIONS)].copy()
    for column in ["is_clean", "target_success", "neighbor_damage", "capability_damage"]:
        frame[column] = frame[column].astype(str).str.lower().isin(["true", "1"])
    return frame


def summarize_outcomes(
    frame: pd.DataFrame,
    *,
    policy: str,
    direction: str,
    expected_evaluations: float,
    dense_scan_evaluations: int,
) -> Dict[str, Any]:
    return {
        "policy": policy,
        "direction": direction,
        "n_paths": int(len(frame)),
        "clean_rate": float(frame["is_clean"].mean()),
        "target_success_rate": float(frame["target_success"].mean()),
        "neighbor_damage_rate": float(frame["neighbor_damage"].mean()),
        "capability_damage_rate": float(frame["capability_damage"].mean()),
        "utility_mean": float(frame["utility"].mean()),
        "abstention_rate": float(frame["abstained"].mean()),
        "expected_evaluations": float(expected_evaluations),
        "evaluation_reduction_vs_dense": 1.0 - float(expected_evaluations) / dense_scan_evaluations,
    }


def selected_to_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "selected_clean": "is_clean",
            "selected_target_success": "target_success",
            "selected_neighbor_damage": "neighbor_damage",
            "selected_capability_damage": "capability_damage",
            "selected_utility": "utility",
        }
    )


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_policy_comparisons(
    policy_frames: Dict[tuple[str, str], pd.DataFrame],
    policies: Iterable[str],
    baselines: Iterable[str],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    keys = ["record_id", "method", "layer"]
    metrics = [
        "utility",
        "is_clean",
        "target_success",
        "neighbor_damage",
        "capability_damage",
    ]
    rows: list[Dict[str, Any]] = []
    comparison_index = 0
    for policy in policies:
        for baseline in baselines:
            for direction in DIRECTIONS:
                left = policy_frames[(policy, direction)][keys + metrics].copy()
                right = policy_frames[(baseline, direction)][keys + metrics].copy()
                merged = left.merge(right, on=keys, suffixes=("_policy", "_baseline"), validate="one_to_one")
                if merged.empty:
                    raise ValueError(
                        f"No paired policy rows for {policy} versus {baseline}, direction={direction}"
                    )
                for metric in metrics:
                    policy_values = merged[f"{metric}_policy"].astype(float)
                    baseline_values = merged[f"{metric}_baseline"].astype(float)
                    merged["difference"] = policy_values - baseline_values
                    record_differences = merged.groupby("record_id")["difference"].mean().to_numpy()
                    low, high = bootstrap_mean_ci(
                        record_differences,
                        reps,
                        seed + comparison_index,
                    )
                    rows.append(
                        {
                            "policy": policy,
                            "baseline": baseline,
                            "direction": direction,
                            "metric": metric,
                            "n_records": int(merged["record_id"].nunique()),
                            "n_paths": int(len(merged)),
                            "policy_mean": float(policy_values.mean()),
                            "baseline_mean": float(baseline_values.mean()),
                            "difference": float(merged["difference"].mean()),
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
                    comparison_index += 1
    return pd.DataFrame(rows)


def add_selector_rows(
    rows: list[Dict[str, Any]],
    selected: pd.DataFrame,
    *,
    task: str,
    label: str,
    probe_count: float,
    dense_scan_evaluations: int,
) -> None:
    task_rows = selected[selected["task"] == task].copy()
    if task_rows.empty:
        raise ValueError(f"Selector task not found: {task}")
    task_rows = selected_to_outcomes(task_rows)
    for direction in DIRECTIONS:
        direction_rows = task_rows[task_rows["direction"] == direction].copy()
        action_rate = 1.0 - float(direction_rows["abstained"].mean())
        expected_evaluations = probe_count + action_rate
        rows.append(
            summarize_outcomes(
                direction_rows,
                policy=label,
                direction=direction,
                expected_evaluations=expected_evaluations,
                dense_scan_evaluations=dense_scan_evaluations,
            )
        )


def format_pair(frame: pd.DataFrame, metric: str) -> str:
    values = []
    for direction in DIRECTIONS:
        value = float(frame[frame["direction"] == direction].iloc[0][metric])
        values.append("--" if math.isnan(value) else f"{value:.4f}")
    return " / ".join(values)


def write_report(path: Path, summary: pd.DataFrame, fixed_alphas: Dict[str, float]) -> None:
    lines = [
        "# Held-Out Strength-Selection Policy Replay",
        "",
        "The fixed coefficient is selected only on the 500-record dense training set. "
        "The 100-record dense validation set is then used only to reveal the measured "
        "outcome at the coefficient chosen by each policy.",
        "",
        f"- train-selected fixed suppression coefficient: `{fixed_alphas['suppression']:g}`",
        f"- train-selected fixed enhancement coefficient: `{fixed_alphas['enhancement']:g}`",
        "- learned-policy cost: two weak probes plus one final intervention when not abstaining",
        "- dense-scan reference cost: 26 nonzero coefficients",
        "",
        "| policy | clean S/E | utility S/E | neighbor damage S/E | abstention S/E | evals S/E |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, group in summary.groupby("policy", sort=False):
        lines.append(
            f"| {policy} | {format_pair(group, 'clean_rate')} | "
            f"{format_pair(group, 'utility_mean')} | "
            f"{format_pair(group, 'neighbor_damage_rate')} | "
            f"{format_pair(group, 'abstention_rate')} | "
            f"{format_pair(group, 'expected_evaluations')} |"
        )
    lines.extend(
        [
            "",
            "This is offline closed-loop policy replay over measured dense-grid outcomes, "
            "not online endpoint generation. It tests whether the selector improves the "
            "actual intervention decision without exposing held-out response curves during selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_report(path: Path, comparisons: pd.DataFrame) -> None:
    utility = comparisons[comparisons["metric"].eq("utility")].copy()
    lines = [
        "# Paired Strength-Policy Comparisons",
        "",
        "Confidence intervals use paired bootstrap resampling over held-out record IDs; all method-layer paths for a record remain together.",
        "",
        "| policy | baseline | direction | utility delta | 95% CI | records | paths |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in utility.itertuples(index=False):
        lines.append(
            f"| {row.policy} | {row.baseline} | {row.direction} | "
            f"{row.difference:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] | "
            f"{row.n_records} | {row.n_paths} |"
        )
    lines.extend(
        [
            "",
            "The accompanying CSV also reports paired changes in clean success, target success, neighbor damage, and capability damage.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dense_train = read_alpha_dataset(Path(args.dense_train_dataset_dir))
    validation = read_alpha_dataset(Path(args.validation_dataset_dir))
    selector_root = Path(args.selector_output_dir)
    selected = pd.read_csv(selector_root / "selected_alpha_rows.csv")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nonzero_alphas = sorted(alpha for alpha in validation["alpha"].unique() if alpha != 0.0)
    dense_scan_evaluations = len(nonzero_alphas)
    rows: list[Dict[str, Any]] = []
    fixed_alphas: Dict[str, float] = {}
    policy_frames: Dict[tuple[str, str], pd.DataFrame] = {}

    for direction in DIRECTIONS:
        train_direction = dense_train[dense_train["direction"] == direction]
        alpha_utility = train_direction.groupby("alpha", as_index=False)["utility"].mean()
        alpha_utility = alpha_utility[alpha_utility["alpha"] != 0.0]
        best_alpha = float(alpha_utility.sort_values("utility", ascending=False).iloc[0]["alpha"])
        fixed_alphas[direction] = best_alpha
        fixed_rows = validation[
            (validation["direction"] == direction) & (validation["alpha"] == best_alpha)
        ].copy()
        fixed_rows["abstained"] = False
        policy_frames[("Train-tuned fixed alpha", direction)] = fixed_rows
        rows.append(
            summarize_outcomes(
                fixed_rows,
                policy="Train-tuned fixed alpha",
                direction=direction,
                expected_evaluations=1.0,
                dense_scan_evaluations=dense_scan_evaluations,
            )
        )

        zero_rows = validation[validation["direction"] == direction][
            ["record_id", "method", "layer"]
        ].drop_duplicates()
        zero_rows = zero_rows.assign(
            is_clean=False,
            target_success=False,
            neighbor_damage=False,
            capability_damage=False,
            utility=0.0,
            abstained=True,
        )
        policy_frames[("No intervention", direction)] = zero_rows
        rows.append(
            summarize_outcomes(
                zero_rows,
                policy="No intervention",
                direction=direction,
                expected_evaluations=0.0,
                dense_scan_evaluations=dense_scan_evaluations,
            )
        )

    add_selector_rows(
        rows,
        selected,
        task=args.dense_selector_task,
        label="Dense-only M+A+R selector",
        probe_count=args.probe_count,
        dense_scan_evaluations=dense_scan_evaluations,
    )
    for task, label in [
        (args.dense_selector_task, "Dense-only M+A+R selector"),
        (args.selector_task, "Multi-fidelity M+A+R selector"),
    ]:
        task_rows = selected_to_outcomes(selected[selected["task"] == task].copy())
        for direction in DIRECTIONS:
            policy_frames[(label, direction)] = task_rows[
                task_rows["direction"] == direction
            ].copy()
    add_selector_rows(
        rows,
        selected,
        task=args.selector_task,
        label="Multi-fidelity M+A+R selector",
        probe_count=args.probe_count,
        dense_scan_evaluations=dense_scan_evaluations,
    )

    oracle = selected[selected["task"] == args.selector_task].copy()
    for direction in DIRECTIONS:
        direction_rows = oracle[oracle["direction"] == direction].copy()
        oracle_rows = pd.DataFrame(
            {
                "is_clean": direction_rows["oracle_clean"],
                "target_success": float("nan"),
                "neighbor_damage": float("nan"),
                "capability_damage": float("nan"),
                "utility": direction_rows["oracle_utility"],
                "abstained": direction_rows["oracle_alpha"] == 0.0,
            }
        )
        rows.append(
            summarize_outcomes(
                oracle_rows,
                policy="Dense oracle",
                direction=direction,
                expected_evaluations=float(dense_scan_evaluations),
                dense_scan_evaluations=dense_scan_evaluations,
            )
        )

    summary = pd.DataFrame(rows)
    policy_order = [
        "No intervention",
        "Train-tuned fixed alpha",
        "Dense-only M+A+R selector",
        "Multi-fidelity M+A+R selector",
        "Dense oracle",
    ]
    summary["policy"] = pd.Categorical(summary["policy"], policy_order, ordered=True)
    summary = summary.sort_values(["policy", "direction"]).reset_index(drop=True)
    summary["policy"] = summary["policy"].astype(str)
    summary.to_csv(output_dir / "strength_policy_evaluation.csv", index=False)
    comparisons = paired_policy_comparisons(
        policy_frames,
        policies=["Dense-only M+A+R selector", "Multi-fidelity M+A+R selector"],
        baselines=["No intervention", "Train-tuned fixed alpha"],
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    comparisons.to_csv(output_dir / "strength_policy_paired_comparisons.csv", index=False)
    write_paired_report(output_dir / "STRENGTH_POLICY_PAIRED_REPORT.md", comparisons)

    manifest = {
        "dense_train_dataset_dir": str(Path(args.dense_train_dataset_dir).resolve()),
        "validation_dataset_dir": str(Path(args.validation_dataset_dir).resolve()),
        "selector_output_dir": str(selector_root.resolve()),
        "selector_task": args.selector_task,
        "dense_selector_task": args.dense_selector_task,
        "fixed_alphas_selected_on_training": fixed_alphas,
        "probe_count": args.probe_count,
        "dense_scan_nonzero_coefficients": dense_scan_evaluations,
        "validation_records": int(validation["record_id"].nunique()),
        "validation_paths": int(
            validation[["record_id", "method", "layer"]].drop_duplicates().shape[0]
        ),
        "paired_bootstrap_unit": "record_id",
        "paired_bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    (output_dir / "strength_policy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "STRENGTH_POLICY_EVALUATION.md", summary, fixed_alphas)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
