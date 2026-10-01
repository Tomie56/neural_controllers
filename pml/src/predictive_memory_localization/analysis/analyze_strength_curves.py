from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit intervention strength curves from suppression per_record.csv files. "
            "The script treats alpha/coefficient scans as candidate intervention paths "
            "and computes onset/window/path-type metrics."
        )
    )
    parser.add_argument(
        "--per-record-csv",
        nargs="+",
        required=True,
        help="One or more summary/per_record.csv files from suppression runs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for path rows, method summaries, and optional plots.",
    )
    parser.add_argument(
        "--label",
        nargs="*",
        default=None,
        help=(
            "Optional labels for each input CSV. If omitted, labels are inferred from "
            "the parent result directory."
        ),
    )
    parser.add_argument(
        "--target-success-threshold",
        type=float,
        default=-0.05,
        help="Target mean delta threshold for target onset.",
    )
    parser.add_argument(
        "--neighbor-damage-threshold",
        type=float,
        default=-0.05,
        help="Neighbor mean delta threshold for damage onset.",
    )
    parser.add_argument(
        "--capability-damage-threshold",
        type=float,
        default=-0.05,
        help="Capability mean delta threshold for capability degradation onset.",
    )
    parser.add_argument(
        "--min-clean-alpha-points",
        type=int,
        default=1,
        help="Minimum number of clean alpha points required for clean-window path type.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write PNG plots if matplotlib is available.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_label(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 3 and parts[-2] == "summary":
        return parts[-3]
    return path.parent.name


def load_rows(paths: List[str], labels: Optional[List[str]]) -> List[Dict[str, Any]]:
    if labels is not None and len(labels) != len(paths):
        raise ValueError("--label count must match --per-record-csv count")
    out: List[Dict[str, Any]] = []
    for idx, raw_path in enumerate(paths):
        path = Path(raw_path)
        dataset_label = labels[idx] if labels is not None else infer_label(path)
        for row in read_csv(path):
            row = dict(row)
            row["dataset_label"] = dataset_label
            row["input_csv"] = str(path)
            row["coef_float"] = to_float(row.get("coef"), 0.0)
            row["target_mean_delta_float"] = to_float(row.get("target_mean_delta"), 0.0)
            row["neighbor_mean_delta_float"] = to_float(row.get("neighbor_mean_delta"), 0.0)
            row["capability_mean_delta_float"] = to_float(row.get("capability_mean_delta"))
            row["has_capability_eval_bool"] = (to_float(row.get("n_capability_eval"), 0.0) or 0.0) > 0
            row["target_success_bool"] = to_bool(row.get("target_success"))
            row["neighbor_damaged_bool"] = to_bool(row.get("neighbor_damaged"))
            row["capability_damaged_bool"] = to_bool(row.get("capability_damaged"))
            out.append(row)
    return out


def alpha_strength(coef: float) -> float:
    return abs(coef)


def sorted_path_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (alpha_strength(float(r["coef_float"])), float(r["coef_float"])))


def trapz(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i - 1] + ys[i]) * (xs[i] - xs[i - 1])
    return total


def monotonicity_error(values: List[float], expected_nonincreasing: bool = True) -> float:
    if len(values) < 2:
        return 0.0
    error = 0.0
    for prev, cur in zip(values, values[1:]):
        diff = cur - prev
        if expected_nonincreasing:
            error += max(0.0, diff)
        else:
            error += max(0.0, -diff)
    return float(error)


def smoothness(values: List[float]) -> float:
    if len(values) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(values) - 1):
        total += abs(values[i + 1] - 2 * values[i] + values[i - 1])
    return float(total)


def first_onset(rows: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
    for row in rows:
        if bool(row.get(key)):
            coef = float(row["coef_float"])
            return coef, alpha_strength(coef)
    return None, None


def clean_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if bool(row.get("target_success_bool"))
        and not bool(row.get("neighbor_damaged_bool"))
        and not bool(row.get("capability_damaged_bool"))
    ]


def longest_consecutive_width(clean_strengths: List[float], all_strengths: List[float]) -> float:
    if not clean_strengths:
        return 0.0
    clean = set(clean_strengths)
    best = 0.0
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    for strength in sorted(all_strengths):
        if strength in clean:
            if current_start is None:
                current_start = strength
            current_end = strength
        else:
            if current_start is not None and current_end is not None:
                best = max(best, current_end - current_start)
            current_start = None
            current_end = None
    if current_start is not None and current_end is not None:
        best = max(best, current_end - current_start)
    return float(best)


def classify_path(path: Dict[str, Any], min_clean_alpha_points: int) -> str:
    if path["target_onset_strength"] is None:
        return "no-effect"
    target = float(path["target_onset_strength"])
    damage = path["damage_onset_strength"]
    capability = path.get("capability_onset_strength")
    if capability is not None and float(capability) <= target:
        return "capability-first"
    if damage is not None and float(damage) <= target:
        return "damage-first"
    if path["collapse_flag"]:
        return "collapse"
    if path["target_monotonicity_error"] > 0.1:
        return "unstable"
    if path["clean_alpha_count"] >= min_clean_alpha_points:
        if path["clean_window_width"] > 0 or min_clean_alpha_points <= 1:
            return "clean-window"
    if path["clean_alpha_count"] > 0:
        return "narrow-window"
    return "target-only-late-damage"


def compute_path_metrics(
    rows: List[Dict[str, Any]],
    target_threshold: float,
    damage_threshold: float,
    capability_threshold: float,
    min_clean_alpha_points: int,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("dataset_label", "")),
            str(row.get("control_method", "")),
            str(row.get("id", "")),
        )
        row["target_success_bool"] = (
            to_float(row.get("target_mean_delta"), 0.0) <= target_threshold
        )
        row["neighbor_damaged_bool"] = (
            to_float(row.get("neighbor_mean_delta"), 0.0) <= damage_threshold
        )
        has_capability = bool(row.get("has_capability_eval_bool"))
        capability_delta = to_float(row.get("capability_mean_delta")) if has_capability else None
        row["capability_damaged_bool"] = (
            capability_delta is not None and capability_delta <= capability_threshold
        )
        groups[key].append(row)

    path_rows: List[Dict[str, Any]] = []
    for (dataset_label, method, record_id), group in sorted(groups.items()):
        path_rows_sorted = sorted_path_rows(group)
        strengths = [alpha_strength(float(row["coef_float"])) for row in path_rows_sorted]
        target_deltas = [float(row["target_mean_delta_float"]) for row in path_rows_sorted]
        neighbor_deltas = [float(row["neighbor_mean_delta_float"]) for row in path_rows_sorted]
        capability_pairs = [
            (alpha_strength(float(row["coef_float"])), float(row["capability_mean_delta_float"]))
            for row in path_rows_sorted
            if bool(row.get("has_capability_eval_bool"))
            and to_float(row.get("capability_mean_delta_float")) is not None
        ]
        capability_deltas = [value for _, value in capability_pairs]
        target_successes = [bool(row["target_success_bool"]) for row in path_rows_sorted]
        neighbor_damages = [bool(row["neighbor_damaged_bool"]) for row in path_rows_sorted]
        capability_damages = [
            bool(row["capability_damaged_bool"])
            for row in path_rows_sorted
            if bool(row.get("has_capability_eval_bool"))
        ]

        target_onset_coef, target_onset_strength = first_onset(path_rows_sorted, "target_success_bool")
        damage_onset_coef, damage_onset_strength = first_onset(path_rows_sorted, "neighbor_damaged_bool")
        capability_onset_coef, capability_onset_strength = first_onset(path_rows_sorted, "capability_damaged_bool")
        clean = clean_points(path_rows_sorted)
        clean_strengths = [alpha_strength(float(row["coef_float"])) for row in clean]
        best_clean = min(clean, key=lambda row: alpha_strength(float(row["coef_float"]))) if clean else None

        max_target_drop = max([max(0.0, -v) for v in target_deltas], default=0.0)
        max_neighbor_drop = max([max(0.0, -v) for v in neighbor_deltas], default=0.0)
        collapse_flag = bool(max_target_drop >= abs(target_threshold) and max_neighbor_drop >= abs(damage_threshold) * 3)
        row0 = path_rows_sorted[0]
        feature_values = {
            key: row0.get(key)
            for key in row0
            if key.startswith("feature_")
        }

        target_auc = trapz(strengths, [max(0.0, -v) for v in target_deltas])
        neighbor_damage_auc = trapz(strengths, [max(0.0, -v) for v in neighbor_deltas])
        capability_damage_auc = trapz(
            [strength for strength, _ in capability_pairs],
            [max(0.0, -value) for _, value in capability_pairs],
        )
        path = {
            **feature_values,
            "dataset_label": dataset_label,
            "control_method": method,
            "id": record_id,
            "concept": row0.get("concept", record_id),
            "n_alpha": len(path_rows_sorted),
            "alpha_values": "|".join(str(row["coef_float"]) for row in path_rows_sorted),
            "strength_values": "|".join(f"{s:g}" for s in strengths),
            "target_onset_coef": target_onset_coef,
            "target_onset_strength": target_onset_strength,
            "damage_onset_coef": damage_onset_coef,
            "damage_onset_strength": damage_onset_strength,
            "damage_minus_target_onset": (
                float(damage_onset_strength) - float(target_onset_strength)
                if damage_onset_strength is not None and target_onset_strength is not None
                else None
            ),
            "capability_onset_coef": capability_onset_coef,
            "capability_onset_strength": capability_onset_strength,
            "capability_minus_target_onset": (
                float(capability_onset_strength) - float(target_onset_strength)
                if capability_onset_strength is not None and target_onset_strength is not None
                else None
            ),
            "clean_window_exists": bool(clean),
            "clean_alpha_count": len(clean),
            "clean_window_width": longest_consecutive_width(clean_strengths, strengths),
            "best_clean_coef": to_float(best_clean.get("coef")) if best_clean else None,
            "best_clean_strength": alpha_strength(float(best_clean["coef_float"])) if best_clean else None,
            "target_auc": target_auc,
            "neighbor_damage_auc": neighbor_damage_auc,
            "capability_damage_auc": capability_damage_auc,
            "pareto_score": target_auc - neighbor_damage_auc,
            "pareto_score_with_capability": target_auc - neighbor_damage_auc - capability_damage_auc,
            "target_min_delta": min(target_deltas) if target_deltas else 0.0,
            "neighbor_min_delta": min(neighbor_deltas) if neighbor_deltas else 0.0,
            "capability_min_delta": min(capability_deltas) if capability_deltas else None,
            "target_success_any": any(target_successes),
            "neighbor_damage_any": any(neighbor_damages),
            "has_capability_eval": bool(capability_pairs),
            "capability_damage_any": any(capability_damages) if capability_damages else None,
            "target_success_rate_over_alpha": mean([float(x) for x in target_successes]) if target_successes else 0.0,
            "neighbor_damage_rate_over_alpha": mean([float(x) for x in neighbor_damages]) if neighbor_damages else 0.0,
            "capability_damage_rate_over_alpha": mean([float(x) for x in capability_damages]) if capability_damages else None,
            "target_monotonicity_error": monotonicity_error(target_deltas, expected_nonincreasing=True),
            "neighbor_monotonicity_error": monotonicity_error(neighbor_deltas, expected_nonincreasing=True),
            "capability_monotonicity_error": monotonicity_error(capability_deltas, expected_nonincreasing=True),
            "target_smoothness": smoothness(target_deltas),
            "neighbor_smoothness": smoothness(neighbor_deltas),
            "capability_smoothness": smoothness(capability_deltas),
            "collapse_flag": collapse_flag,
        }
        path["path_type"] = classify_path(path, min_clean_alpha_points)
        path_rows.append(path)
    return path_rows


def finite_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def summarize_paths(path_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        groups[(str(row["dataset_label"]), str(row["control_method"]))].append(row)

    method_rows: List[Dict[str, Any]] = []
    path_type_rows: List[Dict[str, Any]] = []
    for (dataset_label, method), group in sorted(groups.items()):
        n = len(group)
        type_counts = Counter(str(row["path_type"]) for row in group)
        for path_type, count in sorted(type_counts.items()):
            path_type_rows.append(
                {
                    "dataset_label": dataset_label,
                    "control_method": method,
                    "path_type": path_type,
                    "n": count,
                    "rate": count / n if n else 0.0,
                }
            )

        def frac(key: str) -> float:
            return mean([1.0 if row.get(key) else 0.0 for row in group]) if group else 0.0

        capability_measured = [row for row in group if row.get("has_capability_eval")]

        summary = {
            "dataset_label": dataset_label,
            "control_method": method,
            "n_records": n,
            "target_success_any_rate": frac("target_success_any"),
            "neighbor_damage_any_rate": frac("neighbor_damage_any"),
            "capability_measured_rate": len(capability_measured) / n if n else 0.0,
            "capability_damage_any_rate": (
                mean([1.0 if row.get("capability_damage_any") else 0.0 for row in capability_measured])
                if capability_measured
                else None
            ),
            "clean_window_exists_rate": frac("clean_window_exists"),
            "collapse_rate": frac("collapse_flag"),
            "clean_window_width_mean": mean(finite_values(group, "clean_window_width")) if group else 0.0,
            "clean_alpha_count_mean": mean(finite_values(group, "clean_alpha_count")) if group else 0.0,
            "target_auc_mean": mean(finite_values(group, "target_auc")) if group else 0.0,
            "neighbor_damage_auc_mean": mean(finite_values(group, "neighbor_damage_auc")) if group else 0.0,
            "capability_damage_auc_mean": mean(finite_values(group, "capability_damage_auc")) if group else 0.0,
            "pareto_score_mean": mean(finite_values(group, "pareto_score")) if group else 0.0,
            "pareto_score_with_capability_mean": mean(finite_values(group, "pareto_score_with_capability")) if group else 0.0,
            "target_monotonicity_error_mean": mean(finite_values(group, "target_monotonicity_error")) if group else 0.0,
            "target_smoothness_mean": mean(finite_values(group, "target_smoothness")) if group else 0.0,
        }
        for path_type, count in sorted(type_counts.items()):
            summary[f"path_type_{path_type}_rate"] = count / n if n else 0.0
        for key in [
            "target_onset_strength",
            "damage_onset_strength",
            "capability_onset_strength",
            "damage_minus_target_onset",
            "capability_minus_target_onset",
        ]:
            values = finite_values(group, key)
            summary[f"{key}_mean"] = mean(values) if values else None
            summary[f"{key}_median"] = median(values) if values else None
            summary[f"{key}_n"] = len(values)
        method_rows.append(summary)
    return method_rows, path_type_rows


def summarize_curve_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("dataset_label", "")),
                str(row.get("control_method", "")),
                alpha_strength(float(row["coef_float"])),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (dataset_label, method, strength), group in sorted(groups.items()):
        target_deltas = [float(r["target_mean_delta_float"]) for r in group]
        neighbor_deltas = [float(r["neighbor_mean_delta_float"]) for r in group]
        capability_deltas = [
            float(r["capability_mean_delta_float"])
            for r in group
            if to_float(r.get("capability_mean_delta_float")) is not None
        ]
        target_success = [1.0 if r["target_success_bool"] else 0.0 for r in group]
        neighbor_damage = [1.0 if r["neighbor_damaged_bool"] else 0.0 for r in group]
        capability_damage = [
            1.0 if r["capability_damaged_bool"] else 0.0
            for r in group
            if r.get("has_capability_eval_bool")
        ]
        clean_success = [
            1.0
            if r["target_success_bool"]
            and not r["neighbor_damaged_bool"]
            and not r["capability_damaged_bool"]
            else 0.0
            for r in group
        ]
        out.append(
            {
                "dataset_label": dataset_label,
                "control_method": method,
                "strength": strength,
                "n": len(group),
                "target_delta_mean": mean(target_deltas),
                "target_delta_median": median(target_deltas),
                "neighbor_delta_mean": mean(neighbor_deltas),
                "neighbor_delta_median": median(neighbor_deltas),
                "capability_delta_mean": mean(capability_deltas) if capability_deltas else None,
                "capability_delta_median": median(capability_deltas) if capability_deltas else None,
                "target_success_rate": mean(target_success),
                "neighbor_damage_rate": mean(neighbor_damage),
                "capability_damage_rate": mean(capability_damage) if capability_damage else None,
                "clean_success_rate": mean(clean_success),
            }
        )
    return out


def write_plots(
    output_dir: Path,
    curve_rows: List[Dict[str, Any]],
    path_type_rows: List[Dict[str, Any]],
    path_rows: List[Dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[str(row["dataset_label"])].append(row)

    metrics = [
        ("target_delta_mean", "Target mean delta"),
        ("neighbor_delta_mean", "Neighbor mean delta"),
        ("capability_delta_mean", "Capability mean delta"),
        ("target_success_rate", "Target success rate"),
        ("neighbor_damage_rate", "Neighbor damage rate"),
        ("capability_damage_rate", "Capability damage rate"),
        ("clean_success_rate", "Clean success rate"),
    ]
    for dataset_label, rows in grouped.items():
        methods = sorted({str(row["control_method"]) for row in rows})
        for metric, title in metrics:
            plt.figure(figsize=(7, 4.5))
            for method in methods:
                method_rows = sorted(
                    [
                        row
                        for row in rows
                        if row["control_method"] == method and row.get(metric) not in {"", None}
                    ],
                    key=lambda row: float(row["strength"]),
                )
                if not method_rows:
                    continue
                plt.plot(
                    [float(row["strength"]) for row in method_rows],
                    [float(row[metric]) for row in method_rows],
                    marker="o",
                    label=method,
                )
            if not plt.gca().has_data():
                plt.close()
                continue
            plt.xlabel("Intervention strength |alpha|")
            plt.ylabel(title)
            plt.title(f"{dataset_label}: {title}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            safe_metric = metric.replace("/", "_")
            plt.savefig(plot_dir / f"{dataset_label}_{safe_metric}.png", dpi=160)
            plt.close()

    path_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in path_type_rows:
        path_grouped[str(row["dataset_label"])].append(row)
    for dataset_label, rows in path_grouped.items():
        methods = sorted({str(row["control_method"]) for row in rows})
        path_types = sorted({str(row["path_type"]) for row in rows})
        bottoms = [0.0 for _ in methods]
        plt.figure(figsize=(8, 4.8))
        for path_type in path_types:
            values = []
            for method in methods:
                match = [
                    row
                    for row in rows
                    if row["control_method"] == method and row["path_type"] == path_type
                ]
                values.append(float(match[0]["rate"]) if match else 0.0)
            plt.bar(methods, values, bottom=bottoms, label=path_type)
            bottoms = [b + v for b, v in zip(bottoms, values)]
        plt.ylabel("Rate")
        plt.title(f"{dataset_label}: path type distribution")
        plt.xticks(rotation=25, ha="right")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{dataset_label}_path_type_distribution.png", dpi=160)
        plt.close()

    onset_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        onset_grouped[str(row["dataset_label"])].append(row)

    for dataset_label, rows in onset_grouped.items():
        methods = sorted({str(row["control_method"]) for row in rows})

        plt.figure(figsize=(6.5, 5.5))
        plotted = False
        for method in methods:
            method_rows = [
                row
                for row in rows
                if row["control_method"] == method
                and to_float(row.get("target_onset_strength")) is not None
                and to_float(row.get("damage_onset_strength")) is not None
            ]
            if not method_rows:
                continue
            plotted = True
            plt.scatter(
                [float(row["target_onset_strength"]) for row in method_rows],
                [float(row["damage_onset_strength"]) for row in method_rows],
                s=10,
                alpha=0.35,
                label=method,
            )
        if plotted:
            finite_target = finite_values(rows, "target_onset_strength")
            finite_damage = finite_values(rows, "damage_onset_strength")
            max_axis = max(finite_target + finite_damage + [1.0])
            plt.plot([0, max_axis], [0, max_axis], linestyle="--", color="black", linewidth=1)
            plt.xlabel("Target onset strength")
            plt.ylabel("Neighbor damage onset strength")
            plt.title(f"{dataset_label}: target onset vs neighbor damage onset")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{dataset_label}_target_vs_damage_onset.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4.5))
        plotted = False
        for method in methods:
            values = [
                float(row["clean_window_width"])
                for row in rows
                if row["control_method"] == method
                and to_float(row.get("clean_window_width")) is not None
                and float(row["clean_window_width"]) > 0
            ]
            if not values:
                continue
            plotted = True
            plt.hist(values, bins=20, alpha=0.45, label=method)
        if plotted:
            plt.xlabel("Clean window width")
            plt.ylabel("Number of records")
            plt.title(f"{dataset_label}: nonzero clean window width")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{dataset_label}_clean_window_width_hist.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4.5))
        plotted = False
        for method in methods:
            values = [
                float(row["damage_minus_target_onset"])
                for row in rows
                if row["control_method"] == method
                and to_float(row.get("damage_minus_target_onset")) is not None
            ]
            if not values:
                continue
            plotted = True
            plt.hist(values, bins=24, alpha=0.45, label=method)
        if plotted:
            plt.axvline(0.0, linestyle="--", color="black", linewidth=1)
            plt.xlabel("Damage onset - target onset")
            plt.ylabel("Number of records")
            plt.title(f"{dataset_label}: onset gap distribution")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{dataset_label}_damage_minus_target_onset_hist.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.per_record_csv, args.label)
    path_rows = compute_path_metrics(
        rows,
        target_threshold=args.target_success_threshold,
        damage_threshold=args.neighbor_damage_threshold,
        capability_threshold=args.capability_damage_threshold,
        min_clean_alpha_points=args.min_clean_alpha_points,
    )
    method_rows, path_type_rows = summarize_paths(path_rows)
    curve_rows = summarize_curve_points(rows)

    write_csv(output_dir / "strength_path_rows.csv", path_rows)
    write_csv(output_dir / "method_strength_summary.csv", method_rows)
    write_csv(output_dir / "path_type_distribution.csv", path_type_rows)
    write_csv(output_dir / "method_strength_curve_points.csv", curve_rows)

    summary = {
        "inputs": args.per_record_csv,
        "n_input_rows": len(rows),
        "n_paths": len(path_rows),
        "target_success_threshold": args.target_success_threshold,
        "neighbor_damage_threshold": args.neighbor_damage_threshold,
        "capability_damage_threshold": args.capability_damage_threshold,
        "method_summary": method_rows,
        "path_type_distribution": path_type_rows,
    }
    (output_dir / "strength_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.plot:
        write_plots(output_dir, curve_rows, path_type_rows, path_rows)

    print(f"Wrote path rows to {output_dir / 'strength_path_rows.csv'}")
    print(f"Wrote method summary to {output_dir / 'method_strength_summary.csv'}")
    print(f"Wrote curve points to {output_dir / 'method_strength_curve_points.csv'}")
    print(f"Wrote summary to {output_dir / 'strength_curve_summary.json'}")


if __name__ == "__main__":
    main()
