from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.data.build_editing_tasks_from_suppression import build_task
from predictive_memory_localization.common import load_jsonl, validate_suppression_record, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CommonsenseQA-only balanced follow-up subset for editing prediction. "
            "The default bins balance target_new length: 1, 2, and >=3 words."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--source-dataset", default="tau/commonsense_qa")
    parser.add_argument("--per-bin", type=int, default=50)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--allow-short-bin",
        action="store_true",
        help="Use all available rows if a length bin has fewer than --per-bin candidates.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Do not exclude ids from --exclude-jsonl files.",
    )
    parser.add_argument(
        "--new-target-source",
        choices=["wrong_answer_used", "first_negative_suffix"],
        default="wrong_answer_used",
    )
    parser.add_argument("--require-subject-in-prompt", action="store_true")
    return parser.parse_args()


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or "")


def load_ids(paths: Iterable[str]) -> Set[str]:
    ids: Set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            rid = row_id(row)
            if rid:
                ids.add(rid)
    return ids


def words(text: Any) -> int:
    return len(str(text or "").strip().split())


def target_new_text(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if metadata.get("wrong_answer_used"):
        return str(metadata["wrong_answer_used"]).strip()
    negatives = [str(x) for x in record.get("train_negative", [])]
    if not negatives:
        return ""
    return negatives[0].strip()


def target_new_bin(record: Dict[str, Any]) -> str:
    n = words(target_new_text(record))
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "ge3"


def difficulty_proxy(record: Dict[str, Any]) -> Dict[str, Any]:
    eval_items = record.get("eval", [])
    first_eval = eval_items[0] if eval_items else {}
    target_new = target_new_text(record)
    target_old = str(first_eval.get("target", "")).strip()
    prompt = str(first_eval.get("prompt", "")).strip()
    subject = str(record.get("subject", "")).strip()
    return {
        "target_new_words": words(target_new),
        "target_old_words": words(target_old),
        "prompt_words": words(prompt),
        "subject_words": words(subject),
        "num_paraphrase_prompts": max(len(eval_items) - 1, 0),
        "num_locality_prompts": len(record.get("neighbors", [])),
        "target_new_chars": len(target_new),
        "target_old_chars": len(target_old),
        "prompt_chars": len(prompt),
        "subject_chars": len(subject),
    }


def interleave_by_prompt_length(rows: List[Dict[str, Any]], rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Sample across short/medium/long prompt tertiles to avoid one easy length cluster."""
    shuffled = list(rows)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda row: difficulty_proxy(row)["prompt_words"])
    buckets = [[], [], []]
    for i, row in enumerate(shuffled):
        buckets[min(2, int(i * 3 / max(len(shuffled), 1)))].append(row)
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < count and any(buckets):
        bucket = buckets[cursor % len(buckets)]
        if bucket:
            selected.append(bucket.pop(0))
        cursor += 1
    return selected


def build_report(
    *,
    args: argparse.Namespace,
    input_rows: int,
    excluded_ids: Set[str],
    candidates: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_bin = defaultdict(list)
    for row in selected:
        by_bin[target_new_bin(row)].append(row)
    return {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "tasks_jsonl": args.tasks_jsonl,
        "source_dataset": args.source_dataset,
        "seed": args.seed,
        "per_bin": args.per_bin,
        "include_existing": args.include_existing,
        "exclude_jsonl": args.exclude_jsonl,
        "input_rows": input_rows,
        "excluded_ids": len(excluded_ids),
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "tasks": len(tasks),
        "task_failures": len(failures),
        "candidate_bins": dict(Counter(target_new_bin(row) for row in candidates)),
        "selected_bins": dict(Counter(target_new_bin(row) for row in selected)),
        "selected_source_counts": dict(Counter(str(row.get("source", {}).get("dataset", "")) for row in selected)),
        "difficulty_proxy_by_bin": {
            bin_name: {
                key: sum(float(difficulty_proxy(row)[key]) for row in rows) / len(rows)
                for key in [
                    "target_new_words",
                    "target_old_words",
                    "prompt_words",
                    "num_paraphrase_prompts",
                    "num_locality_prompts",
                ]
            }
            for bin_name, rows in sorted(by_bin.items())
            if rows
        },
        "failure_examples": failures[:20],
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    all_rows = list(load_jsonl(args.input_jsonl))
    excluded_ids = set() if args.include_existing else load_ids(args.exclude_jsonl)

    candidates: List[Dict[str, Any]] = []
    for row in all_rows:
        validate_suppression_record(row)
        if str(row.get("source", {}).get("dataset", "")) != args.source_dataset:
            continue
        if row_id(row) in excluded_ids:
            continue
        candidates.append(row)

    by_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bin[target_new_bin(row)].append(row)

    selected: List[Dict[str, Any]] = []
    for bin_name in ["1", "2", "ge3"]:
        rows = list(by_bin.get(bin_name, []))
        if len(rows) < args.per_bin and not args.allow_short_bin:
            raise ValueError(
                f"Bin {bin_name!r} has only {len(rows)} candidates after exclusions; "
                f"requested {args.per_bin}. Use --allow-short-bin or --include-existing."
            )
        selected.extend(interleave_by_prompt_length(rows, rng, min(args.per_bin, len(rows))))
    rng.shuffle(selected)

    tasks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for row in selected:
        try:
            tasks.append(
                build_task(
                    row,
                    new_target_source=args.new_target_source,
                    require_subject_in_prompt=args.require_subject_in_prompt,
                )
            )
        except Exception as exc:
            failures.append({"id": row.get("id"), "error": repr(exc)})

    write_jsonl(args.output_jsonl, selected)
    write_jsonl(args.tasks_jsonl, tasks)
    report = build_report(
        args=args,
        input_rows=len(all_rows),
        excluded_ids=excluded_ids,
        candidates=candidates,
        selected=selected,
        tasks=tasks,
        failures=failures,
    )
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
