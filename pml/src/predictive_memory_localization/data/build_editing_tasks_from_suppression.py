from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ROME/MEMIT-style editing tasks from suppression records."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--new-target-source",
        choices=["wrong_answer_used", "first_negative_suffix"],
        default="wrong_answer_used",
        help="How to choose the counterfactual target_new for editing.",
    )
    parser.add_argument(
        "--require-subject-in-prompt",
        action="store_true",
        help="Only keep tasks whose subject phrase appears in the primary editing prompt.",
    )
    return parser.parse_args()


def strip_leading_space(text: str) -> str:
    return str(text).strip()


def infer_new_target(record: Dict[str, Any], source: str) -> str:
    metadata = record.get("metadata", {})
    if source == "wrong_answer_used" and metadata.get("wrong_answer_used"):
        return strip_leading_space(metadata["wrong_answer_used"])

    positives = [str(x) for x in record.get("train_positive", [])]
    negatives = [str(x) for x in record.get("train_negative", [])]
    if positives and negatives:
        pos = positives[0]
        neg = negatives[0]
        prefix_len = 0
        for left, right in zip(pos, neg):
            if left != right:
                break
            prefix_len += 1
        suffix = neg[prefix_len:].strip(" .")
        if suffix:
            return suffix

    raise ValueError(f"Could not infer target_new for record {record.get('id')}")


def infer_subject(record: Dict[str, Any], prompt: str) -> str:
    candidates = [
        record.get("subject"),
        record.get("metadata", {}).get("subject"),
        record.get("concept"),
        record.get("id"),
    ]
    for candidate in candidates:
        subject = str(candidate or "").strip()
        if subject:
            return subject
    raise ValueError(f"Could not infer subject for record {record.get('id')}")


def build_task(
    record: Dict[str, Any],
    new_target_source: str,
    require_subject_in_prompt: bool,
) -> Dict[str, Any]:
    eval_items = list(record.get("eval", []))
    if not eval_items:
        raise ValueError(f"Record {record.get('id')} has no eval prompts")

    primary = eval_items[0]
    prompt = str(primary["prompt"])
    ground_truth = strip_leading_space(primary["target"])
    target_new = infer_new_target(record, new_target_source)
    subject = infer_subject(record, prompt)
    subject_in_prompt = subject.lower() in prompt.lower()
    if require_subject_in_prompt and not subject_in_prompt:
        raise ValueError(f"Subject {subject!r} is not in primary prompt")
    paraphrase_prompts = [str(item["prompt"]) for item in eval_items[1:]]

    locality_prompts = []
    for item in record.get("neighbors", []):
        locality_prompts.append(
            {
                "prompt": str(item["prompt"]),
                "ground_truth": strip_leading_space(item["target"]),
            }
        )

    return {
        "id": record["id"],
        "concept": record.get("concept", record["id"]),
        "subject": subject,
        "subject_in_prompt": subject_in_prompt,
        "source": record.get("source", {}),
        "prompt": prompt,
        "target_new": target_new,
        "ground_truth": ground_truth,
        "paraphrase_prompts": paraphrase_prompts,
        "locality_prompts": locality_prompts,
        "metadata": {
            "suppression_record_id": record["id"],
            "new_target_source": new_target_source,
            "wrong_answer_used": record.get("metadata", {}).get("wrong_answer_used"),
            "subject_in_prompt": subject_in_prompt,
        },
    }


def main() -> None:
    args = parse_args()
    rows = list(load_jsonl(args.input_jsonl))
    if args.max_records is not None:
        rows = rows[: args.max_records]

    tasks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for record in rows:
        try:
            tasks.append(
                build_task(
                    record,
                    args.new_target_source,
                    args.require_subject_in_prompt,
                )
            )
        except Exception as exc:
            failures.append({"id": record.get("id"), "error": repr(exc)})

    write_jsonl(args.output_jsonl, tasks)
    manifest = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "tasks": len(tasks),
        "failures": len(failures),
        "require_subject_in_prompt": args.require_subject_in_prompt,
        "subject_in_prompt_tasks": sum(1 for task in tasks if task.get("subject_in_prompt")),
        "failure_examples": failures[:20],
    }
    manifest_path = Path(args.output_jsonl).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
