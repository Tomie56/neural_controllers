from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl, validate_suppression_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate suppression-data JSONL.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    return {
        "min": min(values),
        "median": float(median(values)),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    rows = list(load_jsonl(args.input_jsonl))
    errors = []
    ids = []
    source_ids = []
    source_datasets = []
    train_pos_counts = []
    train_neg_counts = []
    eval_counts = []
    neighbor_counts = []
    subject_count = 0
    subject_in_first_eval_count = 0

    for idx, row in enumerate(rows):
        try:
            validate_suppression_record(row)
        except Exception as exc:
            errors.append({"index": idx, "id": row.get("id"), "error": repr(exc)})
        ids.append(str(row.get("id", "")))
        source = row.get("source", {})
        source_ids.append(str(source.get("source_id", "")))
        source_datasets.append(str(source.get("dataset", "")))
        train_pos_counts.append(len(row.get("train_positive", [])))
        train_neg_counts.append(len(row.get("train_negative", [])))
        eval_counts.append(len(row.get("eval", [])))
        neighbor_counts.append(len(row.get("neighbors", [])))
        subject = str(row.get("subject", "")).strip()
        if subject:
            subject_count += 1
            eval_items = row.get("eval", [])
            if eval_items and subject.lower() in str(eval_items[0].get("prompt", "")).lower():
                subject_in_first_eval_count += 1

    report = {
        "input_jsonl": args.input_jsonl,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "unique_source_ids": len(set(source_ids)),
        "duplicate_ids": [item for item, count in Counter(ids).items() if item and count > 1][:50],
        "duplicate_source_ids": [item for item, count in Counter(source_ids).items() if item and count > 1][:50],
        "source_dataset_counts": dict(Counter(source_datasets)),
        "train_positive_count": stats(train_pos_counts),
        "train_negative_count": stats(train_neg_counts),
        "eval_count": stats(eval_counts),
        "neighbor_count": stats(neighbor_counts),
        "subject_count": subject_count,
        "subject_in_first_eval_count": subject_in_first_eval_count,
        "subject_in_first_eval_rate": (
            subject_in_first_eval_count / subject_count if subject_count else 0.0
        ),
        "num_errors": len(errors),
        "errors": errors[:50],
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
