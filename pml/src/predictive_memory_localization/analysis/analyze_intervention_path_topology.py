from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset import (
    atomic_write_csv,
    atomic_write_json,
    discover_runs,
)


DEFAULT_ROOT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b"
)
DIRECTIONS = ("suppression", "enhancement")
SUMMARY_METRICS = [
    "target_exists",
    "damage_exists",
    "clean_window_exists",
    "target_first",
    "damage_first",
    "simultaneous_onset",
    "target_only",
    "damage_only",
    "neither_target_nor_damage",
    "instability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize onset ordering, observed clean windows, and path instability."
    )
    parser.add_argument("--stage-root", default=str(DEFAULT_ROOT / "outcomes"))
    parser.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_ROOT / "prediction_dataset_strict_later/prediction_dataset_manifest.json"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT / "path_topology_analysis"))
    parser.add_argument("--random-method", default="random")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def first_true_strength(strengths: np.ndarray, values: np.ndarray) -> float:
    selected = strengths[values]
    return float(selected.min()) if len(selected) else float("nan")


def has_reversal(values: np.ndarray) -> bool:
    true_indices = np.flatnonzero(values)
    return bool(len(true_indices) and np.any(~values[true_indices[0] :]))


def largest_contiguous_window(
    strengths: np.ndarray, clean: np.ndarray
) -> tuple[float, float, float, int]:
    best_start = float("nan")
    best_end = float("nan")
    best_span = float("nan")
    best_count = 0
    start: int | None = None
    for index, is_clean in enumerate(clean):
        if is_clean and start is None:
            start = index
        at_end = index == len(clean) - 1
        if start is not None and ((not is_clean) or at_end):
            end = index if is_clean and at_end else index - 1
            count = end - start + 1
            span = float(strengths[end] - strengths[start])
            if count > best_count or (count == best_count and span > best_span):
                best_start = float(strengths[start])
                best_end = float(strengths[end])
                best_span = span
                best_count = count
            start = None
    return best_start, best_end, best_span, best_count


def direction_row(
    group: pd.DataFrame,
    direction: str,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    sign = -1 if direction == "suppression" else 1
    selected = group[np.sign(group["alpha"]) == sign].copy()
    selected["abs_alpha"] = selected["alpha"].abs()
    selected = selected.sort_values("abs_alpha")
    strengths = selected["abs_alpha"].to_numpy(dtype=float)
    target_delta = selected["target_mean_delta_margin"].to_numpy(dtype=float)
    neighbor_delta = selected["neighbor_mean_delta_margin"].to_numpy(dtype=float)
    capability_delta = selected["capability_mean_delta_margin"].to_numpy(dtype=float)
    target = (
        target_delta <= -thresholds["target"]
        if direction == "suppression"
        else target_delta >= thresholds["target"]
    )
    neighbor_damage = neighbor_delta <= -thresholds["neighbor"]
    capability_damage = capability_delta <= -thresholds["capability"]
    damage = neighbor_damage | capability_damage
    clean = target & ~damage
    target_onset = first_true_strength(strengths, target)
    damage_onset = first_true_strength(strengths, damage)
    if np.isfinite(target_onset) and np.isfinite(damage_onset):
        if target_onset < damage_onset:
            ordering = "target_first"
        elif damage_onset < target_onset:
            ordering = "damage_first"
        else:
            ordering = "simultaneous"
    elif np.isfinite(target_onset):
        ordering = "target_only"
    elif np.isfinite(damage_onset):
        ordering = "damage_only"
    else:
        ordering = "neither"
    clean_min, clean_max, clean_span, clean_contiguous_count = largest_contiguous_window(
        strengths, clean
    )
    target_instability = has_reversal(target)
    damage_instability = has_reversal(damage)
    first = selected.iloc[0]
    return {
        "record_id": str(first["record_id"]),
        "method": str(first["control_method"]),
        "layer": int(first["layer"]),
        "dataset": str(first["dataset"]),
        "domain": str(first["domain"]),
        "direction": direction,
        "n_nonzero_strengths": int(len(strengths)),
        "target_exists": bool(target.any()),
        "damage_exists": bool(damage.any()),
        "clean_window_exists": bool(clean.any()),
        "target_onset_strength": target_onset,
        "damage_onset_strength": damage_onset,
        "onset_ordering": ordering,
        "target_first": ordering == "target_first",
        "damage_first": ordering == "damage_first",
        "simultaneous_onset": ordering == "simultaneous",
        "target_only": ordering == "target_only",
        "damage_only": ordering == "damage_only",
        "neither_target_nor_damage": ordering == "neither",
        "clean_strength_count": int(clean.sum()),
        "largest_contiguous_clean_count": clean_contiguous_count,
        "clean_window_min_strength": clean_min,
        "clean_window_max_strength": clean_max,
        "clean_window_observed_span": clean_span,
        "target_instability": target_instability,
        "damage_instability": damage_instability,
        "instability": target_instability or damage_instability,
    }


def bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: List[Dict[str, Any]] = []
    for (method, layer, direction), group in rows.groupby(["method", "layer", "direction"]):
        summary: Dict[str, Any] = {
            "method": method,
            "layer": layer,
            "direction": direction,
            "n_paths": len(group),
        }
        for metric in SUMMARY_METRICS:
            summary[f"{metric}_rate"] = float(group[metric].mean())
        summary["target_onset_mean"] = float(group["target_onset_strength"].mean())
        summary["damage_onset_mean"] = float(group["damage_onset_strength"].mean())
        summary["clean_window_observed_span_mean"] = float(
            group["clean_window_observed_span"].mean()
        )
        summary["clean_strength_count_mean"] = float(group["clean_strength_count"].mean())
        summaries.append(summary)
    return pd.DataFrame(summaries).sort_values(["method", "layer", "direction"])


def paired_vs_random(
    rows: pd.DataFrame,
    random_method: str,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    output: List[Dict[str, Any]] = []
    learned_methods = sorted(
        set(rows["method"]) - {random_method, "matched_norm_random"}
    )
    comparison_index = 0
    for layer in sorted(rows["layer"].unique()):
        for direction in DIRECTIONS:
            random = rows[
                rows["method"].eq(random_method)
                & rows["layer"].eq(layer)
                & rows["direction"].eq(direction)
            ].set_index("record_id")
            for method in learned_methods:
                learned = rows[
                    rows["method"].eq(method)
                    & rows["layer"].eq(layer)
                    & rows["direction"].eq(direction)
                ].set_index("record_id")
                shared = learned.index.intersection(random.index)
                for metric in SUMMARY_METRICS:
                    differences = (
                        learned.loc[shared, metric].astype(float).to_numpy()
                        - random.loc[shared, metric].astype(float).to_numpy()
                    )
                    low, high = bootstrap_ci(differences, reps, seed + comparison_index)
                    output.append(
                        {
                            "method": method,
                            "layer": layer,
                            "direction": direction,
                            "metric": metric,
                            "n_records": len(shared),
                            "learned_rate": float(learned.loc[shared, metric].mean()),
                            "random_rate": float(random.loc[shared, metric].mean()),
                            "difference": float(differences.mean()),
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
                    comparison_index += 1
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": output_dir / "path_topology_rows.csv",
        "summary": output_dir / "path_topology_summary.csv",
        "paired": output_dir / "path_topology_paired_vs_random.csv",
        "manifest": output_dir / "path_topology_manifest.json",
        "report": output_dir / "PATH_TOPOLOGY_REPORT.md",
    }
    if all(path.exists() for path in paths.values()) and not args.overwrite:
        print(paths["manifest"].read_text(encoding="utf-8"))
        return
    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in dataset_manifest["thresholds"].items()}
    topology_rows: List[Dict[str, Any]] = []
    for method, layer, layer_dir in discover_runs(Path(args.stage_root)):
        frame = pd.read_csv(layer_dir / "per_record.csv")
        frame["layer"] = layer
        for _, group in frame.groupby("record_id", sort=False):
            for direction in DIRECTIONS:
                topology_rows.append(direction_row(group, direction, thresholds))
    rows = pd.DataFrame(topology_rows)
    summary = summarize(rows)
    paired = paired_vs_random(rows, args.random_method, args.bootstrap_reps, args.seed)
    atomic_write_csv(paths["rows"], rows)
    atomic_write_csv(paths["summary"], summary)
    atomic_write_csv(paths["paired"], paired)
    manifest = {
        "stage_root": str(Path(args.stage_root).resolve()),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "thresholds": thresholds,
        "n_rows": len(rows),
        "n_records": int(rows["record_id"].nunique()),
        "methods": sorted(rows["method"].unique()),
        "layers": sorted(int(value) for value in rows["layer"].unique()),
        "strength_grid": sorted(
            float(value)
            for value in pd.concat(
                [pd.read_csv(path / "per_record.csv", usecols=["alpha"]) for _, _, path in discover_runs(Path(args.stage_root))]
            )["alpha"].abs().unique()
            if value > 0
        ),
        "clean_window_width_definition": "Span of the largest contiguous run of clean points on the measured |alpha| grid.",
        "instability_definition": "A target-success or damage indicator reverts at a larger measured |alpha| after first becoming true.",
        "paired_bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    atomic_write_json(paths["manifest"], manifest)
    focus = summary[
        summary["method"].isin(["random", "mean_difference", "logistic", "rfm_agop_top1"])
    ]
    report = [
        "# Intervention-Path Topology Analysis",
        "",
        "This analysis recomputes path topology from the full measured coefficient grid.",
        "Window width is an observed-grid span, not a claim about an unmeasured continuous interval.",
        "",
        focus.to_csv(index=False),
        "Full row-level and paired learned-minus-random results are provided as CSV files.",
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
