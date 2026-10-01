from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import (
    continuation_logprob,
    load_jsonl,
    load_model_and_tokenizer,
    write_jsonl,
)


FORBIDDEN_RECORD_EVAL_KEYS = {
    "eval",
    "neighbors",
    "capability",
    "target_eval",
    "neighbor_eval",
    "capability_eval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize fresh bidirectional PML build bundles into frozen records/eval_bank/"
            "assignments, repairing duplicate record ids and eval id conflicts."
        )
    )
    parser.add_argument(
        "--bundles-jsonl",
        default="/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000.build_bundles.jsonl",
    )
    parser.add_argument(
        "--output-prefix",
        default="/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000_v1",
    )
    parser.add_argument("--target-records", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument(
        "--sample-per-dataset",
        type=int,
        default=8,
        help="Rows per dataset in the human QA sample report.",
    )
    parser.add_argument(
        "--margin-sanity-model-path",
        default=None,
        help="Optional local HF model path. If set, score alpha=0 margins for a sample.",
    )
    parser.add_argument("--margin-sanity-sample", type=int, default=300)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_hash(parts: Iterable[Any], length: int = 12) -> str:
    payload = "\u0000".join(str(x) for x in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def snake_case(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] or "record"


def output_paths(prefix: str) -> Dict[str, Path]:
    p = Path(prefix)
    return {
        "records": Path(f"{p}.records.jsonl"),
        "eval_bank": Path(f"{p}.eval_bank.jsonl"),
        "assignments": Path(f"{p}.eval_assignments.jsonl"),
        "validation": Path(f"{p}.validation.json"),
        "sample_report": Path(f"{p}.sample_report.jsonl"),
        "rejected": Path(f"{p}.rejected_bundles.jsonl"),
        "margin_sanity": Path(f"{p}.margin_sanity.json"),
    }


def ensure_leading_space(value: Any) -> str:
    text = str(value or "")
    if text and text[0].isalnum():
        return " " + text
    return text


def normalize_eval_item(item: Dict[str, Any], eval_type: str) -> Dict[str, Any]:
    prompt = str(item.get("prompt", "")).strip()
    correct = ensure_leading_space(item.get("correct", item.get("target", "")))
    contrast = ensure_leading_space(item.get("contrast", item.get("incorrect", "")))
    if not prompt or not correct.strip() or not contrast.strip():
        raise ValueError(f"invalid {eval_type} eval item: {item}")
    if correct.strip().lower() == contrast.strip().lower():
        raise ValueError(f"{eval_type} correct equals contrast: {item}")
    return {
        "prompt": prompt,
        "correct": correct,
        "contrast": contrast,
        "source_kind": eval_type,
        "metadata": {
            k: v
            for k, v in item.items()
            if k not in {"eval_id", "eval_type", "prompt", "correct", "target", "contrast", "incorrect", "source_kind"}
        },
    }


def bundle_sort_key(bundle: Dict[str, Any], index: int) -> Tuple[int, str, int]:
    freshness = bundle.get("record", {}).get("metadata", {}).get("seed_freshness_group", "")
    freshness_rank = 0 if freshness == "recent_2024_2026" else 1
    seed_source_key = str(bundle.get("seed_source_key") or "")
    return (freshness_rank, seed_source_key, index)


def repaired_record_id(bundle: Dict[str, Any], used: set[str]) -> Tuple[str, bool]:
    record = bundle["record"]
    base = snake_case(str(record.get("record_id") or record.get("id") or record.get("concept") or "record"))
    if base not in used:
        used.add(base)
        return base, False
    source = record.get("source", {})
    suffix = stable_hash(
        [
            bundle.get("seed_source_key"),
            source.get("dataset"),
            source.get("dataset_config"),
            source.get("source_id"),
            source.get("question"),
        ],
        length=8,
    )
    candidate = f"{base}_{suffix}"
    n = 2
    while candidate in used:
        candidate = f"{base}_{suffix}_{n}"
        n += 1
    used.add(candidate)
    return candidate, True


def normalize_bundle(bundle: Dict[str, Any], index: int, used_record_ids: set[str]) -> Dict[str, Any]:
    if "record" not in bundle or not isinstance(bundle["record"], dict):
        raise ValueError("missing record")
    if "eval_items" not in bundle or not isinstance(bundle["eval_items"], list):
        raise ValueError("missing eval_items")
    if "assignment" not in bundle or not isinstance(bundle["assignment"], dict):
        raise ValueError("missing assignment")

    record = dict(bundle["record"])
    record_id, repaired = repaired_record_id(bundle, used_record_ids)
    record["record_id"] = record_id
    record.pop("id", None)
    for key in FORBIDDEN_RECORD_EVAL_KEYS:
        record.pop(key, None)
    if len(record.get("train_positive", [])) < 6 or len(record.get("train_negative", [])) < 6:
        raise ValueError(f"{record_id} has too few train_positive/train_negative rows")
    metadata = dict(record.get("metadata", {}))
    metadata["finalized_schema_version"] = "pml_bidirectional_v1_freeze"
    metadata["record_id_repaired"] = repaired
    metadata["finalize_input_index"] = index
    metadata["seed_source_key"] = bundle.get("seed_source_key")
    record["metadata"] = metadata

    grouped: Dict[str, List[Dict[str, Any]]] = {"target": [], "neighbor": [], "capability": []}
    for item in bundle["eval_items"]:
        eval_type = str(item.get("eval_type") or item.get("source_kind") or "").strip()
        if eval_type not in grouped:
            raise ValueError(f"{record_id} has unsupported eval_type={eval_type!r}")
        grouped[eval_type].append(normalize_eval_item(item, eval_type))

    expected = {"target": 3, "neighbor": 3, "capability": 4}
    for eval_type, n_expected in expected.items():
        if len(grouped[eval_type]) != n_expected:
            raise ValueError(f"{record_id} needs {n_expected} {eval_type} evals, got {len(grouped[eval_type])}")

    eval_items = []
    assignment = {
        "record_id": record_id,
        "target_eval_ids": [],
        "neighbor_eval_ids": [],
        "capability_eval_ids": [],
        "assignment_policy": {
            "target": "record_specific",
            "neighbor": "record_specific",
            "capability": "global_eval_bank_candidate_hash_prompt_correct_contrast",
        },
    }
    for eval_type in ["target", "neighbor", "capability"]:
        for idx, item in enumerate(grouped[eval_type]):
            if eval_type == "capability":
                eval_id = f"capability::{stable_hash([item['prompt'], item['correct'], item['contrast']], length=16)}"
                source_record_id = None
            else:
                eval_id = f"{eval_type}::{record_id}::{idx:02d}"
                source_record_id = record_id
            eval_row = {
                "eval_id": eval_id,
                "eval_type": eval_type,
                "prompt": item["prompt"],
                "correct": item["correct"],
                "contrast": item["contrast"],
                "source_record_id": source_record_id,
                "source_eval_index": idx,
                "metadata": item.get("metadata", {}) | {"source_kind": eval_type},
            }
            eval_items.append(eval_row)
            assignment[f"{eval_type}_eval_ids"].append(eval_id)

    return {
        "record_id": record_id,
        "record": record,
        "eval_items": eval_items,
        "assignment": assignment,
    }


def validate_outputs(
    records: List[Dict[str, Any]],
    eval_bank: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    record_ids = [row["record_id"] for row in records]
    eval_ids = [row["eval_id"] for row in eval_bank]
    record_id_set = set(record_ids)
    eval_id_set = set(eval_ids)

    missing_refs = []
    assignment_len_counts = {key: collections.Counter() for key in ["target_eval_ids", "neighbor_eval_ids", "capability_eval_ids"]}
    for row in assignments:
        if row["record_id"] not in record_id_set:
            missing_refs.append({"kind": "record_id", "value": row["record_id"]})
        for key in assignment_len_counts:
            ids = row.get(key, [])
            assignment_len_counts[key][len(ids)] += 1
            for eval_id in ids:
                if eval_id not in eval_id_set:
                    missing_refs.append({"kind": key, "value": eval_id, "record_id": row["record_id"]})

    bad_eval_fields = []
    leading_space_issues = []
    for row in eval_bank:
        if not row.get("prompt") or not row.get("correct") or not row.get("contrast"):
            bad_eval_fields.append({"eval_id": row.get("eval_id"), "reason": "empty"})
        elif row["correct"].strip().lower() == row["contrast"].strip().lower():
            bad_eval_fields.append({"eval_id": row.get("eval_id"), "reason": "same_correct_contrast"})
        for field in ["correct", "contrast"]:
            value = str(row.get(field, ""))
            if value and value[0].isalnum():
                leading_space_issues.append({"eval_id": row.get("eval_id"), "field": field, "value": value[:80]})

    forbidden_record_fields = [
        {"record_id": row["record_id"], "fields": sorted(FORBIDDEN_RECORD_EVAL_KEYS & set(row.keys()))}
        for row in records
        if FORBIDDEN_RECORD_EVAL_KEYS & set(row.keys())
    ]

    eval_key_counter = collections.Counter(
        (row["eval_type"], row["prompt"], row["correct"], row["contrast"]) for row in eval_bank
    )
    duplicate_eval_content = sum(1 for count in eval_key_counter.values() if count > 1)

    by_dataset = collections.Counter(row.get("source", {}).get("dataset") for row in records)
    by_domain = collections.Counter(row.get("source", {}).get("domain") for row in records)
    by_freshness = collections.Counter(row.get("metadata", {}).get("seed_freshness_group") for row in records)
    by_release_year = collections.Counter(str(row.get("metadata", {}).get("seed_release_year")) for row in records)
    by_eval_type = collections.Counter(row["eval_type"] for row in eval_bank)

    return {
        "schema": "pml_bidirectional_v1_freeze",
        "target_records": args.target_records,
        "n_records": len(records),
        "n_eval_bank": len(eval_bank),
        "n_assignments": len(assignments),
        "n_rejected_bundles": len(rejected),
        "n_duplicate_record_ids": len(record_ids) - len(record_id_set),
        "n_duplicate_eval_ids": len(eval_ids) - len(eval_id_set),
        "n_duplicate_eval_content": duplicate_eval_content,
        "n_missing_refs": len(missing_refs),
        "missing_refs_examples": missing_refs[:20],
        "n_bad_eval_fields": len(bad_eval_fields),
        "bad_eval_fields_examples": bad_eval_fields[:20],
        "n_leading_space_issues": len(leading_space_issues),
        "leading_space_issue_examples": leading_space_issues[:20],
        "n_forbidden_record_field_rows": len(forbidden_record_fields),
        "forbidden_record_field_examples": forbidden_record_fields[:20],
        "assignment_len_counts": {
            key: {str(k): v for k, v in counter.items()} for key, counter in assignment_len_counts.items()
        },
        "eval_type_counts": dict(by_eval_type),
        "dataset_counts": dict(by_dataset),
        "domain_counts": dict(by_domain),
        "freshness_group_counts": dict(by_freshness),
        "release_year_counts": dict(by_release_year),
        "is_freeze_ready": (
            len(records) >= args.target_records
            and len(record_ids) == len(record_id_set)
            and len(eval_ids) == len(eval_id_set)
            and not missing_refs
            and not bad_eval_fields
            and not leading_space_issues
            and not forbidden_record_fields
        ),
    }


def build_sample_report(
    records: List[Dict[str, Any]],
    eval_by_id: Dict[str, Dict[str, Any]],
    assignments_by_record: Dict[str, Dict[str, Any]],
    sample_per_dataset: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_dataset: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        by_dataset[str(row.get("source", {}).get("dataset"))].append(row)
    sample_rows = []
    for dataset, rows in sorted(by_dataset.items()):
        chosen = rows[:]
        rng.shuffle(chosen)
        for record in chosen[:sample_per_dataset]:
            assignment = assignments_by_record[record["record_id"]]
            sample_rows.append(
                {
                    "dataset": dataset,
                    "record_id": record["record_id"],
                    "concept": record.get("concept"),
                    "subject": record.get("subject"),
                    "seed_question": record.get("seed_question"),
                    "seed_answer": record.get("seed_answer"),
                    "train_positive_first3": record.get("train_positive", [])[:3],
                    "train_negative_first3": record.get("train_negative", [])[:3],
                    "target_eval": [eval_by_id[eid] for eid in assignment["target_eval_ids"]],
                    "neighbor_eval": [eval_by_id[eid] for eid in assignment["neighbor_eval_ids"]],
                    "capability_eval": [eval_by_id[eid] for eid in assignment["capability_eval_ids"]],
                }
            )
    return sample_rows


def run_margin_sanity(
    eval_bank: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    if not args.margin_sanity_model_path:
        return None
    rng = random.Random(args.seed)
    by_type: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in eval_bank:
        by_type[row["eval_type"]].append(row)

    sample_rows = []
    per_type = max(args.margin_sanity_sample // max(len(by_type), 1), 1)
    for rows in by_type.values():
        rows = rows[:]
        rng.shuffle(rows)
        sample_rows.extend(rows[:per_type])
    sample_rows = sample_rows[: args.margin_sanity_sample]

    model, tokenizer = load_model_and_tokenizer(
        args.margin_sanity_model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    rows_out = []
    counters = collections.defaultdict(lambda: {"n": 0, "positive_margin": 0, "sum_margin": 0.0})
    for row in sample_rows:
        correct_lp = continuation_logprob(model, tokenizer, row["prompt"], row["correct"])["total_logprob"]
        contrast_lp = continuation_logprob(model, tokenizer, row["prompt"], row["contrast"])["total_logprob"]
        margin = correct_lp - contrast_lp
        stat = counters[row["eval_type"]]
        stat["n"] += 1
        stat["positive_margin"] += int(margin > 0)
        stat["sum_margin"] += margin
        rows_out.append(
            {
                "eval_id": row["eval_id"],
                "eval_type": row["eval_type"],
                "prompt": row["prompt"],
                "correct": row["correct"],
                "contrast": row["contrast"],
                "correct_logprob": correct_lp,
                "contrast_logprob": contrast_lp,
                "margin": margin,
            }
        )

    summary = {}
    for eval_type, stat in counters.items():
        n = stat["n"]
        summary[eval_type] = {
            "n": n,
            "positive_margin_rate": stat["positive_margin"] / n if n else None,
            "mean_margin": stat["sum_margin"] / n if n else None,
        }
    return {
        "model_path": args.margin_sanity_model_path,
        "sample_size": len(rows_out),
        "summary": summary,
        "rows": rows_out,
    }


def main() -> None:
    args = parse_args()
    paths = output_paths(args.output_prefix)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            path.unlink(missing_ok=True)
        elif path.exists():
            raise FileExistsError(f"Output exists: {path}. Use --overwrite.")

    raw_bundles = list(load_jsonl(args.bundles_jsonl))
    indexed_bundles = sorted(
        [(idx, bundle) for idx, bundle in enumerate(raw_bundles)],
        key=lambda item: bundle_sort_key(item[1], item[0]),
    )

    used_record_ids: set[str] = set()
    accepted = []
    rejected = []
    for index, bundle in indexed_bundles:
        if len(accepted) >= args.target_records:
            break
        try:
            accepted.append(normalize_bundle(bundle, index, used_record_ids))
        except BaseException as exc:  # noqa: BLE001
            rejected.append(
                {
                    "input_index": index,
                    "record_id": bundle.get("record_id"),
                    "seed_source_key": bundle.get("seed_source_key"),
                    "error": repr(exc),
                }
            )

    records = [row["record"] for row in accepted]
    assignments = [row["assignment"] for row in accepted]
    eval_bank_by_id: Dict[str, Dict[str, Any]] = {}
    eval_conflicts = []
    for row in accepted:
        for item in row["eval_items"]:
            eval_id = item["eval_id"]
            if eval_id in eval_bank_by_id:
                old = eval_bank_by_id[eval_id]
                if (
                    old["prompt"] != item["prompt"]
                    or old["correct"] != item["correct"]
                    or old["contrast"] != item["contrast"]
                ):
                    eval_conflicts.append({"eval_id": eval_id, "old": old, "new": item})
                continue
            eval_bank_by_id[eval_id] = item
    eval_bank = [eval_bank_by_id[k] for k in sorted(eval_bank_by_id)]

    validation = validate_outputs(records, eval_bank, assignments, rejected, args)
    validation["input_bundles_jsonl"] = args.bundles_jsonl
    validation["n_input_bundles"] = len(raw_bundles)
    validation["n_eval_conflicts_after_repair"] = len(eval_conflicts)
    validation["eval_conflict_examples_after_repair"] = eval_conflicts[:20]
    validation["records_jsonl"] = str(paths["records"])
    validation["eval_bank_jsonl"] = str(paths["eval_bank"])
    validation["assignments_jsonl"] = str(paths["assignments"])
    validation["sample_report_jsonl"] = str(paths["sample_report"])
    validation["rejected_bundles_jsonl"] = str(paths["rejected"])

    eval_by_id = {row["eval_id"]: row for row in eval_bank}
    assignments_by_record = {row["record_id"]: row for row in assignments}
    sample_rows = build_sample_report(
        records,
        eval_by_id,
        assignments_by_record,
        args.sample_per_dataset,
        args.seed,
    )

    write_jsonl(paths["records"], records)
    write_jsonl(paths["eval_bank"], eval_bank)
    write_jsonl(paths["assignments"], assignments)
    write_jsonl(paths["sample_report"], sample_rows)
    write_jsonl(paths["rejected"], rejected)

    margin_sanity = run_margin_sanity(eval_bank, args)
    if margin_sanity is not None:
        paths["margin_sanity"].write_text(
            json.dumps(margin_sanity, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        validation["margin_sanity_json"] = str(paths["margin_sanity"])
        validation["margin_sanity_summary"] = margin_sanity["summary"]
    else:
        validation["margin_sanity_json"] = None
        validation["margin_sanity_summary"] = None

    paths["validation"].write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
