from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl, validate_suppression_record, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a path-type representative JSONL subset from existing strength-path rows. "
            "The output is intended for dense-alpha / capability-onset probes."
        )
    )
    parser.add_argument("--path-rows-csv", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--dataset-label", default="heldout_144_agop_direct")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
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
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sort_key(row: Dict[str, Any], fields: Iterable[Tuple[str, bool]]) -> Tuple[Any, ...]:
    out: List[Any] = []
    for key, reverse in fields:
        value = to_float(row.get(key), 0.0)
        if value is None:
            value = 0.0
        out.append(-value if reverse else value)
    return tuple(out)


def add_candidates(
    selected_ids: List[str],
    selected_reasons: Dict[str, str],
    candidates: List[Dict[str, Any]],
    quota: int,
    reason: str,
) -> None:
    for row in candidates:
        rid = str(row.get("id", ""))
        if not rid or rid in selected_reasons:
            continue
        selected_ids.append(rid)
        selected_reasons[rid] = reason
        if sum(1 for item in selected_ids if selected_reasons[item] == reason) >= quota:
            break


def representative_rows(path_rows: List[Dict[str, Any]], max_records: int, seed: int) -> Tuple[List[str], Dict[str, str]]:
    rng = random.Random(seed)
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        by_id[str(row.get("id", ""))].append(row)

    selected_ids: List[str] = []
    selected_reasons: Dict[str, str] = {}

    topk_rows = [row for row in path_rows if row.get("control_method") == "agop_topk_project"]
    top1_rows = [row for row in path_rows if row.get("control_method") == "agop_top1"]

    high_risk = [
        row
        for row in topk_rows
        if row.get("path_type") in {"damage-first", "collapse"} or to_bool(row.get("collapse_flag"))
    ]
    high_risk = sorted(
        high_risk,
        key=lambda row: sort_key(row, [("neighbor_damage_auc", True), ("target_auc", True)]),
    )

    topk_clean = sorted(
        [row for row in topk_rows if row.get("path_type") == "clean-window"],
        key=lambda row: sort_key(row, [("clean_window_width", True), ("damage_minus_target_onset", True)]),
    )

    top1_clean = sorted(
        [row for row in top1_rows if row.get("path_type") == "clean-window"],
        key=lambda row: sort_key(row, [("clean_window_width", True), ("damage_minus_target_onset", True)]),
    )

    no_effect = sorted(
        [row for row in path_rows if row.get("path_type") == "no-effect"],
        key=lambda row: sort_key(row, [("target_auc", False), ("neighbor_damage_auc", False)]),
    )

    finite_gap = [
        row
        for row in path_rows
        if to_float(row.get("damage_minus_target_onset")) is not None
        and row.get("path_type") not in {"no-effect"}
    ]
    onset_borderline = sorted(
        finite_gap,
        key=lambda row: (
            abs(float(row["damage_minus_target_onset"])),
            str(row.get("control_method", "")),
            str(row.get("id", "")),
        ),
    )

    # Quotas intentionally sum below max_records; the remainder is a deterministic
    # fill that preserves diversity without hard-coding every future path type.
    quotas = [
        ("agop_topk_damage_risk", high_risk, min(16, max_records)),
        ("agop_topk_clean_window", topk_clean, min(8, max_records)),
        ("agop_top1_clean_window", top1_clean, min(10, max_records)),
        ("no_effect", no_effect, min(8, max_records)),
        ("onset_borderline", onset_borderline, min(8, max_records)),
    ]
    for reason, candidates, quota in quotas:
        if len(selected_ids) >= max_records:
            break
        add_candidates(selected_ids, selected_reasons, candidates, min(quota, max_records - len(selected_ids)), reason)

    if len(selected_ids) < max_records:
        remaining_ids = [rid for rid in by_id if rid and rid not in selected_reasons]
        rng.shuffle(remaining_ids)
        for rid in remaining_ids[: max_records - len(selected_ids)]:
            selected_ids.append(rid)
            selected_reasons[rid] = "diversity_fill"

    return selected_ids[:max_records], selected_reasons


def make_diagnostic_rows(
    selected_ids: List[str],
    selected_reasons: Dict[str, str],
    path_rows: List[Dict[str, Any]],
    jsonl_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id_method: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in path_rows:
        by_id_method[(str(row.get("id", "")), str(row.get("control_method", "")))] = row

    out = []
    methods = sorted({str(row.get("control_method", "")) for row in path_rows})
    for rank, rid in enumerate(selected_ids, start=1):
        record = jsonl_by_id.get(rid, {})
        base = {
            "rank": rank,
            "id": rid,
            "concept": record.get("concept", ""),
            "subject": record.get("subject", ""),
            "source_dataset": record.get("source", {}).get("dataset", ""),
            "selection_reason": selected_reasons.get(rid, ""),
            "n_eval": len(record.get("eval", [])),
            "n_neighbors": len(record.get("neighbors", [])),
            "n_capability": len(record.get("capability", [])),
        }
        for method in methods:
            row = by_id_method.get((rid, method), {})
            prefix = method.replace("-", "_")
            base[f"{prefix}_path_type"] = row.get("path_type", "")
            base[f"{prefix}_target_onset"] = row.get("target_onset_strength", "")
            base[f"{prefix}_damage_onset"] = row.get("damage_onset_strength", "")
            base[f"{prefix}_damage_minus_target"] = row.get("damage_minus_target_onset", "")
            base[f"{prefix}_clean_width"] = row.get("clean_window_width", "")
            base[f"{prefix}_target_any"] = row.get("target_success_any", "")
            base[f"{prefix}_damage_any"] = row.get("neighbor_damage_any", "")
            base[f"{prefix}_collapse_flag"] = row.get("collapse_flag", "")
        out.append(base)
    return out


def main() -> None:
    args = parse_args()
    path_rows = [
        row
        for row in read_csv(Path(args.path_rows_csv))
        if str(row.get("dataset_label", "")) == args.dataset_label
    ]
    if not path_rows:
        raise ValueError(f"No path rows found for dataset_label={args.dataset_label!r}")

    input_records = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(args.input_jsonl):
        validate_suppression_record(row)
        rid = str(row.get("id", ""))
        input_records.append(row)
        if rid and rid not in by_id:
            by_id[rid] = row

    selected_ids, selected_reasons = representative_rows(path_rows, args.max_records, args.seed)
    missing = [rid for rid in selected_ids if rid not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} selected ids are absent from input JSONL; examples: {missing[:10]}"
        )

    selected_id_set = set(selected_ids)
    selected_records = [row for row in input_records if str(row.get("id", "")) in selected_id_set]
    selected_records = sorted(selected_records, key=lambda row: selected_ids.index(str(row.get("id", ""))))
    write_jsonl(args.output_jsonl, selected_records)

    diagnostics = make_diagnostic_rows(selected_ids, selected_reasons, path_rows, by_id)
    if args.output_csv:
        write_csv(Path(args.output_csv), diagnostics)

    report = {
        "path_rows_csv": args.path_rows_csv,
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "dataset_label": args.dataset_label,
        "seed": args.seed,
        "max_records": args.max_records,
        "path_rows": len(path_rows),
        "input_records": len(input_records),
        "selected_records": len(selected_records),
        "selection_reason_counts": dict(Counter(selected_reasons[rid] for rid in selected_ids)),
        "path_type_counts_by_method": {
            method: dict(Counter(row.get("path_type", "") for row in path_rows if row.get("control_method") == method))
            for method in sorted({row.get("control_method", "") for row in path_rows})
        },
        "selected_ids": selected_ids,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
