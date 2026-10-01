from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize path/capability-onset outcomes for a representative subset, "
            "grouped by the subset selection reason."
        )
    )
    parser.add_argument("--path-rows-csv", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
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
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def frac(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    if not rows:
        return None
    return mean([1.0 if to_bool(row.get(key)) else 0.0 for row in rows])


def rate_for_value(rows: List[Dict[str, Any]], key: str, value: str) -> Optional[float]:
    if not rows:
        return None
    return mean([1.0 if str(row.get(key, "")) == value else 0.0 for row in rows])


def summarize_group(rows: List[Dict[str, Any]], group_key: Tuple[str, str]) -> Dict[str, Any]:
    selection_reason, method = group_key
    capability_rows = [row for row in rows if to_bool(row.get("has_capability_eval"))]
    out: Dict[str, Any] = {
        "selection_reason": selection_reason,
        "control_method": method,
        "n_records": len(rows),
        "target_success_any_rate": frac(rows, "target_success_any"),
        "neighbor_damage_any_rate": frac(rows, "neighbor_damage_any"),
        "clean_window_exists_rate": frac(rows, "clean_window_exists"),
        "collapse_flag_rate": frac(rows, "collapse_flag"),
        "has_capability_eval_rate": len(capability_rows) / len(rows) if rows else None,
        "capability_damage_any_rate": frac(capability_rows, "capability_damage_any") if capability_rows else None,
    }
    for path_type in ["clean-window", "damage-first", "capability-first", "no-effect", "collapse", "unstable"]:
        out[f"path_type_{path_type}_rate"] = rate_for_value(rows, "path_type", path_type)
    for key in [
        "target_onset_strength",
        "damage_onset_strength",
        "capability_onset_strength",
        "damage_minus_target_onset",
        "capability_minus_target_onset",
        "clean_window_width",
        "target_auc",
        "neighbor_damage_auc",
        "capability_damage_auc",
        "pareto_score",
        "pareto_score_with_capability",
    ]:
        values = finite_values(rows, key)
        out[f"{key}_n"] = len(values)
        out[f"{key}_mean"] = mean(values) if values else None
        out[f"{key}_median"] = median(values) if values else None
    return out


def add_labels(path_rows: List[Dict[str, str]], diagnostics_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    reason_by_id = {row["id"]: row.get("selection_reason", "") for row in diagnostics_rows if row.get("id")}
    concept_by_id = {row["id"]: row.get("concept", "") for row in diagnostics_rows if row.get("id")}
    out: List[Dict[str, Any]] = []
    for row in path_rows:
        rid = str(row.get("id", ""))
        if rid not in reason_by_id:
            continue
        item: Dict[str, Any] = dict(row)
        item["selection_reason"] = reason_by_id[rid]
        item["diagnostic_concept"] = concept_by_id.get(rid, "")
        out.append(item)
    return out


def build_markdown(summary_rows: List[Dict[str, Any]], report: Dict[str, Any]) -> str:
    lines = [
        "# Representative Capability Probe Outcome Summary",
        "",
        "This report is generated from path rows and the representative-subset diagnostics table.",
        "",
        "## Inputs",
        "",
        f"- path rows: `{report['path_rows_csv']}`",
        f"- diagnostics: `{report['diagnostics_csv']}`",
        f"- dataset_label: `{report.get('dataset_label') or ''}`",
        f"- matched path rows: `{report['matched_path_rows']}`",
        f"- matched ids: `{report['matched_ids']}`",
        "",
        "## Group Summary",
        "",
        "| Selection reason | Method | n | target any | damage any | capability any | clean-window | damage-first | capability-first | no-effect | target onset med | damage onset med | capability onset med |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        has_capability = (row.get("has_capability_eval_rate") or 0.0) > 0.0
        lines.append(
            "| {selection_reason} | {control_method} | {n_records} | {target_success_any_rate:.4f} | "
            "{neighbor_damage_any_rate:.4f} | {capability_damage_any_rate} | {path_type_clean-window_rate:.4f} | "
            "{path_type_damage-first_rate:.4f} | {path_type_capability-first_rate} | "
            "{path_type_no-effect_rate:.4f} | {target_onset_strength_median} | "
            "{damage_onset_strength_median} | {capability_onset_strength_median} |".format(
                **{
                    **row,
                    "capability_damage_any_rate": (
                        f"{row['capability_damage_any_rate']:.4f}"
                        if row.get("capability_damage_any_rate") is not None
                        else ""
                    ),
                    "path_type_capability-first_rate": (
                        f"{row['path_type_capability-first_rate']:.4f}"
                        if has_capability
                        else ""
                    ),
                    "target_onset_strength_median": (
                        f"{row['target_onset_strength_median']:.4f}"
                        if row.get("target_onset_strength_median") is not None
                        else ""
                    ),
                    "damage_onset_strength_median": (
                        f"{row['damage_onset_strength_median']:.4f}"
                        if row.get("damage_onset_strength_median") is not None
                        else ""
                    ),
                    "capability_onset_strength_median": (
                        f"{row['capability_onset_strength_median']:.4f}"
                        if row.get("capability_onset_strength_median") is not None
                        else ""
                    ),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Empty capability columns mean the path rows did not contain capability probes; do not read that as zero capability damage.",
            "- `capability-first` is only meaningful when `has_capability_eval_rate` is nonzero.",
            "- This subset is intentionally path-representative, not distribution-representative.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    path_rows = read_csv(Path(args.path_rows_csv))
    if args.dataset_label:
        path_rows = [row for row in path_rows if row.get("dataset_label") == args.dataset_label]
    diagnostics_rows = read_csv(Path(args.diagnostics_csv))
    labeled = add_labels(path_rows, diagnostics_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in labeled:
        groups[(str(row.get("selection_reason", "")), str(row.get("control_method", "")))].append(row)

    summary_rows = [summarize_group(rows, key) for key, rows in sorted(groups.items())]
    write_csv(output_dir / "selection_reason_summary.csv", summary_rows)
    write_csv(output_dir / "labeled_path_rows.csv", labeled)

    report = {
        "path_rows_csv": args.path_rows_csv,
        "diagnostics_csv": args.diagnostics_csv,
        "dataset_label": args.dataset_label,
        "input_path_rows": len(path_rows),
        "diagnostics_rows": len(diagnostics_rows),
        "matched_path_rows": len(labeled),
        "matched_ids": len({row.get("id", "") for row in labeled}),
        "selection_reason_counts": dict(Counter(row.get("selection_reason", "") for row in diagnostics_rows)),
        "control_methods": sorted({row.get("control_method", "") for row in labeled}),
    }
    (output_dir / "representative_outcome_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "REPRESENTATIVE_OUTCOME_SUMMARY.md").write_text(
        build_markdown(summary_rows, report),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
