from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl, validate_suppression_record, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified suppression-data subset.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-count",
        action="append",
        default=[],
        help="Dataset=count, e.g. allenai/openbookqa=200. Repeat for stratified sampling.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Fallback uniform sample size.")
    return parser.parse_args()


def parse_source_counts(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --source-count {item!r}; expected dataset=count")
        key, value = item.rsplit("=", 1)
        out[key] = int(value)
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = list(load_jsonl(args.input_jsonl))
    valid_rows = []
    for row in rows:
        validate_suppression_record(row)
        valid_rows.append(row)

    source_counts = parse_source_counts(args.source_count)
    if source_counts:
        by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in valid_rows:
            by_source[str(row.get("source", {}).get("dataset", ""))].append(row)
        selected: List[Dict[str, Any]] = []
        for source, count in source_counts.items():
            candidates = list(by_source.get(source, []))
            if len(candidates) < count:
                raise ValueError(f"Source {source!r} has only {len(candidates)} rows, requested {count}")
            rng.shuffle(candidates)
            selected.extend(candidates[:count])
        rng.shuffle(selected)
    else:
        if args.max_records is None:
            raise ValueError("Provide either --source-count or --max-records")
        selected = list(valid_rows)
        rng.shuffle(selected)
        selected = selected[: args.max_records]

    write_jsonl(args.output_jsonl, selected)
    report = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "seed": args.seed,
        "input_rows": len(rows),
        "valid_input_rows": len(valid_rows),
        "selected_rows": len(selected),
        "source_dataset_counts": dict(Counter(str(row.get("source", {}).get("dataset", "")) for row in selected)),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
