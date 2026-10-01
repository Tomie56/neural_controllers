from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze path-conditioned minimum effective strength from strength_path_rows.csv. "
            "This is a report-only analysis; it does not run model scoring."
        )
    )
    parser.add_argument("--path-rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-label", default=None)
    return parser.parse_args()


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


def to_float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def frac(rows: List[Dict[str, Any]], key: str) -> float:
    return mean([1.0 if to_bool(row.get(key)) else 0.0 for row in rows]) if rows else 0.0


def rate(rows: List[Dict[str, Any]], key: str, value: str) -> float:
    return mean([1.0 if str(row.get(key)) == value else 0.0 for row in rows]) if rows else 0.0


def summarize_method(rows: List[Dict[str, Any]], dataset_label: str, method: str) -> Dict[str, Any]:
    target_onsets = finite(rows, "target_onset_strength")
    best_clean = finite(rows, "best_clean_strength")
    clean_width = finite(rows, "clean_window_width")
    clean_rows = [row for row in rows if to_bool(row.get("clean_window_exists"))]
    strict_clean_rows = [row for row in rows if str(row.get("path_type")) == "clean-window"]
    damage_first_rows = [row for row in rows if str(row.get("path_type")) == "damage-first"]
    unstable_rows = [row for row in rows if str(row.get("path_type")) == "unstable"]

    def med(values: List[float]) -> Optional[float]:
        return median(values) if values else None

    return {
        "dataset_label": dataset_label,
        "control_method": method,
        "n_records": len(rows),
        "target_any_rate": frac(rows, "target_success_any"),
        "neighbor_damage_any_rate": frac(rows, "neighbor_damage_any"),
        "clean_window_exists_rate": frac(rows, "clean_window_exists"),
        "strict_clean_path_rate": rate(rows, "path_type", "clean-window"),
        "damage_first_rate": rate(rows, "path_type", "damage-first"),
        "unstable_rate": rate(rows, "path_type", "unstable"),
        "no_effect_rate": rate(rows, "path_type", "no-effect"),
        "collapse_rate": rate(rows, "path_type", "collapse"),
        "target_onset_median": med(target_onsets),
        "target_onset_mean": mean(target_onsets) if target_onsets else None,
        "best_clean_strength_median_all_clean": med(best_clean),
        "best_clean_strength_mean_all_clean": mean(best_clean) if best_clean else None,
        "best_clean_strength_median_strict_clean": med(finite(strict_clean_rows, "best_clean_strength")),
        "target_onset_median_strict_clean": med(finite(strict_clean_rows, "target_onset_strength")),
        "target_onset_median_damage_first": med(finite(damage_first_rows, "target_onset_strength")),
        "damage_onset_median_damage_first": med(finite(damage_first_rows, "damage_onset_strength")),
        "target_onset_median_unstable": med(finite(unstable_rows, "target_onset_strength")),
        "clean_window_width_median_all": med(clean_width),
        "clean_window_width_median_clean_only": med(finite(clean_rows, "clean_window_width")),
        "clean_alpha_count_mean": mean(finite(rows, "clean_alpha_count")) if rows else None,
        "target_monotonicity_error_mean": mean(finite(rows, "target_monotonicity_error")) if rows else None,
        "target_smoothness_mean": mean(finite(rows, "target_smoothness")) if rows else None,
        "capability_measured_rate": frac(rows, "has_capability_eval"),
    }


def f4(value: Any) -> str:
    value = to_float(value)
    return f"{value:.4f}" if value is not None else ""


def build_markdown(
    method_rows: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> str:
    lines = [
        "# Path-Conditioned Minimum Effective Strength",
        "",
        "This report separates first target onset from usable intervention strength. It is generated from existing path rows and does not rerun model scoring.",
        "",
        "## Inputs",
        "",
        f"- path rows: `{report['path_rows_csv']}`",
        f"- dataset_label: `{report.get('dataset_label') or ''}`",
        f"- rows: `{report['n_rows']}`",
        "",
        "## Method Summary",
        "",
        "| Method | n | target any | clean exists | strict clean path | damage-first | unstable | target onset med | best clean med | strict clean onset med | damage-first damage onset med |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            "| {method} | {n} | {target_any} | {clean_exists} | {strict_clean} | {damage_first} | {unstable} | {target_onset} | {best_clean} | {strict_onset} | {damage_onset} |".format(
                method=row["control_method"],
                n=row["n_records"],
                target_any=f4(row["target_any_rate"]),
                clean_exists=f4(row["clean_window_exists_rate"]),
                strict_clean=f4(row["strict_clean_path_rate"]),
                damage_first=f4(row["damage_first_rate"]),
                unstable=f4(row["unstable_rate"]),
                target_onset=f4(row["target_onset_median"]),
                best_clean=f4(row["best_clean_strength_median_all_clean"]),
                strict_onset=f4(row["target_onset_median_strict_clean"]),
                damage_onset=f4(row["damage_onset_median_damage_first"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `target onset` is the first alpha where target logprob crosses the suppression threshold.",
            "- `best clean` is the smallest alpha where target suppression succeeds without neighbor/capability damage.",
            "- `strict clean path` is more conservative: the whole path must avoid earlier damage-first/collapse/instability classifications.",
            "- If clean exists is high but strict clean path is low, the direction can hit a clean point but the path is not reliable enough to call the intervention naturally controllable.",
            "- Capability columns are meaningful only when `capability_measured_rate > 0`; otherwise this report only covers target and neighbor behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.path_rows_csv))
    if args.dataset_label:
        rows = [row for row in rows if row.get("dataset_label") == args.dataset_label]

    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset_label", "")), str(row.get("control_method", "")))].append(row)

    method_rows = [
        summarize_method(group_rows, dataset, method)
        for (dataset, method), group_rows in sorted(groups.items())
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "minimum_strength_method_summary.csv", method_rows)

    report = {
        "path_rows_csv": args.path_rows_csv,
        "dataset_label": args.dataset_label,
        "n_rows": len(rows),
        "n_methods": len(method_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "minimum_strength_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "MINIMUM_EFFECTIVE_STRENGTH.md").write_text(
        build_markdown(method_rows, report),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
