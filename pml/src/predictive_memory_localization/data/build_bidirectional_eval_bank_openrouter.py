from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import append_jsonl, load_jsonl, write_jsonl


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_PREFIX = "/data/neural_controllers/pml/data/main_datas/bidirectional_commonsense_3000"


SYSTEM_PROMPT = """You convert existing PML evaluation probes into contrastive evaluation probes.

Return exactly one JSON object. Do not include markdown, comments, or explanations.

The downstream experiment learns an intervention direction from train_positive versus train_negative.
It will evaluate both intervention signs:
- negative alpha: suppression / weakening of the target fact
- positive alpha: enhancement / strengthening of the target fact

Every evaluation item must be contrastive:
- prompt: a short prefix
- correct: the true continuation, with leading whitespace when it starts with a word
- contrast: a plausible but wrong competing continuation, with leading whitespace when it starts with a word

Important dataset design:
- Do NOT treat eval probes as fields inside the intervention record.
- The final dataset will use a global eval_bank and a separate eval_assignment table.
- Your job here is only to add or repair contrastive eval probes for one source record.

Requirements:
- Preserve the original target meaning in target_eval.
- For neighbor_eval, test nearby facts that should remain stable under target steering.
- For capability_eval, use unrelated ordinary commonsense or elementary-science probes that should remain stable.
- If the input already has eval/neighbors/capability probes, convert them and add contrast.
- If capability probes are present, do not replace them with target-related probes.
- If capability probes are missing, create exactly four unrelated safe commonsense probes.
- The contrast must be plausible enough to compete with correct, but clearly wrong for this prompt.
- Do not make contrast identical, synonymous, or trivially compatible with correct.
- Keep continuations short and natural.
- Avoid unsafe, sexual, medical, legal, private, political, hateful, or sensitive content.
- Do not include chain-of-thought or explanations.

Output schema:
{
  "target_eval": [
    {"prompt": "...", "correct": " ...", "contrast": " ..."}
  ],
  "neighbor_eval": [
    {"prompt": "...", "correct": " ...", "contrast": " ..."}
  ],
  "capability_eval": [
    {"prompt": "...", "correct": " ...", "contrast": " ..."}
  ]
}

Metric convention:
margin = logprob(correct) - logprob(contrast)
- suppression success means margin decreases enough under negative alpha
- enhancement success means margin increases enough under positive alpha
"""


USER_TEMPLATE = """Convert this existing PML record's eval probes into contrastive probes.

Input record JSON:
{record_json}

Return exactly one JSON object with target_eval, neighbor_eval, and capability_eval.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized bidirectional PML data: intervention records, global eval bank, "
            "and record-to-eval assignments. Uses OpenRouter only when contrasts are missing."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        default="/data/neural_controllers/pml/data/suppression_commonsense_3000_valid_with_capability.jsonl",
    )
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--eval-bank-jsonl", default=None)
    parser.add_argument("--assignments-jsonl", default=None)
    parser.add_argument("--bundles-jsonl", default=None)
    parser.add_argument("--failures-jsonl", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=OPENROUTER_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument(
        "--max-capability-per-record",
        type=int,
        default=4,
        help="Maximum capability eval IDs assigned to each record. Use 0 to keep all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    prefix = Path(args.output_prefix)
    return {
        "records": Path(args.records_jsonl or f"{prefix}.records.jsonl"),
        "eval_bank": Path(args.eval_bank_jsonl or f"{prefix}.eval_bank.jsonl"),
        "assignments": Path(args.assignments_jsonl or f"{prefix}.eval_assignments.jsonl"),
        "bundles": Path(args.bundles_jsonl or f"{prefix}.bundles.jsonl"),
        "failures": Path(args.failures_jsonl or f"{prefix}.failures.jsonl"),
        "report": Path(f"{prefix}.report.json"),
    }


def get_api_key(env_name: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(f"Missing API key. Set {env_name}=<OPENROUTER_API_KEY>.")
    return key


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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
        raise ValueError(f"Invalid {eval_type} eval item: {item}")
    if correct.strip().lower() == contrast.strip().lower():
        raise ValueError(f"{eval_type} contrast equals correct: {item}")
    return {
        "prompt": prompt,
        "correct": correct,
        "contrast": contrast,
        "source_kind": eval_type,
        **{
            k: v
            for k, v in item.items()
            if k
            not in {
                "prompt",
                "correct",
                "target",
                "contrast",
                "incorrect",
                "source_kind",
            }
        },
    }


def existing_contrastive_sections(record: Dict[str, Any]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    section_map = {
        "target_eval": ("target_eval", "eval", "target"),
        "neighbor_eval": ("neighbor_eval", "neighbors", "neighbor"),
        "capability_eval": ("capability_eval", "capability", "capability"),
    }
    out: Dict[str, List[Dict[str, Any]]] = {}
    for out_key, (new_key, legacy_key, eval_type) in section_map.items():
        items = record.get(new_key) or record.get(legacy_key) or []
        if not items:
            out[out_key] = []
            continue
        if not all(("contrast" in item and ("correct" in item or "target" in item)) for item in items):
            return None
        out[out_key] = [normalize_eval_item(item, eval_type) for item in items]
    if not out.get("target_eval"):
        return None
    return out


def validate_sections(sections: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    target_eval = [normalize_eval_item(item, "target") for item in sections.get("target_eval", [])]
    neighbor_eval = [normalize_eval_item(item, "neighbor") for item in sections.get("neighbor_eval", [])]
    capability_eval = [
        normalize_eval_item(item, "capability") for item in sections.get("capability_eval", [])
    ]
    if len(target_eval) < 1:
        raise ValueError(f"Record {original.get('id')} needs at least one target_eval")
    if len(neighbor_eval) < 1:
        raise ValueError(f"Record {original.get('id')} needs at least one neighbor_eval")
    if len(capability_eval) < 2:
        raise ValueError(f"Record {original.get('id')} needs at least two capability_eval items")
    return {
        "target_eval": target_eval,
        "neighbor_eval": neighbor_eval,
        "capability_eval": capability_eval,
    }


def call_openrouter(record: Dict[str, Any], args: argparse.Namespace, api_key: str) -> Dict[str, List[Dict[str, Any]]]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    record_json=json.dumps(record, ensure_ascii=False, indent=2)
                ),
            },
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Optional[BaseException] = None
    for attempt in range(args.retries):
        try:
            response = requests.post(args.base_url, headers=headers, json=payload, timeout=120)
            if response.status_code >= 400:
                raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
            parsed = extract_json_object(data["choices"][0]["message"]["content"])
            return validate_sections(parsed, record)
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(args.sleep * (2**attempt))
    raise RuntimeError(f"OpenRouter call failed after {args.retries} retries: {last_error}") from last_error


def strip_intervention_record(record: Dict[str, Any], model: str) -> Dict[str, Any]:
    keep_keys = [
        "id",
        "concept",
        "subject",
        "source",
        "seed_question",
        "seed_answer",
        "train_positive",
        "train_negative",
        "fact_or_concept",
    ]
    out = {key: record[key] for key in keep_keys if key in record}
    if not out.get("id"):
        raise ValueError("Missing record id")
    if not out.get("train_positive") or not out.get("train_negative"):
        raise ValueError(f"Record {out.get('id')} needs train_positive and train_negative")
    metadata = dict(record.get("metadata", {}))
    metadata["bidirectional_schema_version"] = "v2_eval_bank"
    metadata["contrast_generation_model"] = model
    out["metadata"] = metadata
    return out


def stable_hash(parts: List[str], length: int = 16) -> str:
    payload = "\u0000".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def make_eval_id(eval_type: str, record_id: str, index: int, item: Dict[str, Any]) -> str:
    if eval_type == "capability":
        return f"capability::{stable_hash([item['prompt'], item['correct']])}"
    return f"{eval_type}::{record_id}::{index:02d}"


def make_eval_bank_row(
    eval_id: str,
    eval_type: str,
    record_id: str,
    index: int,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = {
        k: v
        for k, v in item.items()
        if k not in {"prompt", "correct", "contrast", "source_kind"}
    }
    return {
        "eval_id": eval_id,
        "eval_type": eval_type,
        "prompt": item["prompt"],
        "correct": item["correct"],
        "contrast": item["contrast"],
        "source_record_id": record_id if eval_type in {"target", "neighbor"} else item.get("source_record_id"),
        "source_eval_index": index,
        "metadata": metadata,
    }


def build_bundle(record: Dict[str, Any], args: argparse.Namespace, api_key: Optional[str]) -> Dict[str, Any]:
    record_id = str(record.get("id", ""))
    sections = existing_contrastive_sections(record)
    if sections is None:
        if api_key is None:
            raise RuntimeError("Contrasts are missing and no API key is available.")
        sections = call_openrouter(record, args, api_key)
    else:
        sections = validate_sections(sections, record)

    intervention_record = strip_intervention_record(record, args.model)
    eval_items: List[Dict[str, Any]] = []
    assignment = {
        "record_id": record_id,
        "target_eval_ids": [],
        "neighbor_eval_ids": [],
        "capability_eval_ids": [],
        "eval_policy": {
            "target": "record_specific_eval_bank_items",
            "neighbor": "record_specific_eval_bank_items",
            "capability": "global_eval_bank_items_assigned_to_record",
        },
    }

    for eval_type, section_key, assignment_key in [
        ("target", "target_eval", "target_eval_ids"),
        ("neighbor", "neighbor_eval", "neighbor_eval_ids"),
        ("capability", "capability_eval", "capability_eval_ids"),
    ]:
        items = sections[section_key]
        if eval_type == "capability" and args.max_capability_per_record > 0:
            items = items[: args.max_capability_per_record]
        for index, item in enumerate(items):
            eval_id = make_eval_id(eval_type, record_id, index, item)
            eval_items.append(make_eval_bank_row(eval_id, eval_type, record_id, index, item))
            assignment[assignment_key].append(eval_id)

    return {
        "record_id": record_id,
        "record": intervention_record,
        "eval_items": eval_items,
        "assignment": assignment,
    }


def existing_bundle_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("record_id")) for row in load_jsonl(path) if row.get("record_id")}


def materialize_outputs(paths: Dict[str, Path]) -> Dict[str, int]:
    bundles = list(load_jsonl(paths["bundles"])) if paths["bundles"].exists() else []
    records: Dict[str, Dict[str, Any]] = {}
    eval_bank: Dict[str, Dict[str, Any]] = {}
    assignments: Dict[str, Dict[str, Any]] = {}
    capability_conflicts = 0

    for bundle in bundles:
        record_id = str(bundle["record_id"])
        records[record_id] = bundle["record"]
        assignments[record_id] = bundle["assignment"]
        for item in bundle["eval_items"]:
            eval_id = str(item["eval_id"])
            if eval_id in eval_bank:
                old = eval_bank[eval_id]
                if (
                    old.get("prompt") != item.get("prompt")
                    or old.get("correct") != item.get("correct")
                    or old.get("contrast") != item.get("contrast")
                ):
                    capability_conflicts += 1
                continue
            eval_bank[eval_id] = item

    write_jsonl(paths["records"], [records[k] for k in sorted(records)])
    write_jsonl(paths["eval_bank"], [eval_bank[k] for k in sorted(eval_bank)])
    write_jsonl(paths["assignments"], [assignments[k] for k in sorted(assignments)])
    return {
        "n_records": len(records),
        "n_eval_bank": len(eval_bank),
        "n_assignments": len(assignments),
        "capability_conflicts": capability_conflicts,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    paths = output_paths(args)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    elif paths["bundles"].exists() and not args.resume:
        raise FileExistsError(f"Bundle output exists: {paths['bundles']}. Use --resume or --overwrite.")

    input_path = Path(args.input_jsonl)
    records = list(load_jsonl(input_path))
    if args.offset:
        records = records[args.offset :]
    if args.max_records is not None:
        records = records[: args.max_records]

    if args.dry_run:
        print("SYSTEM_PROMPT:")
        print(SYSTEM_PROMPT)
        print("\nUSER_PROMPT_EXAMPLE:")
        print(USER_TEMPLATE.format(record_json=json.dumps(records[0], ensure_ascii=False, indent=2)))
        print("\nOUTPUTS:")
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    done_ids = existing_bundle_record_ids(paths["bundles"]) if args.resume else set()
    todo = [record for record in records if str(record.get("id", "")) not in done_ids]
    needs_api = any(existing_contrastive_sections(record) is None for record in todo)
    api_key = get_api_key(args.api_key_env) if needs_api else None

    print(f"Input records: {len(records)}")
    print(f"Already bundled: {len(done_ids)}")
    print(f"To process: {len(todo)}")
    print(f"Needs OpenRouter: {needs_api}")
    print(f"Records: {paths['records']}")
    print(f"Eval bank: {paths['eval_bank']}")
    print(f"Assignments: {paths['assignments']}")
    print(f"Bundles checkpoint: {paths['bundles']}")
    print(f"Failures: {paths['failures']}")

    def work(record: Dict[str, Any]) -> Dict[str, Any]:
        return build_bundle(record, args, api_key)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {executor.submit(work, record): record for record in todo}
        pbar = tqdm(total=len(pending), desc="bidirectional eval bank")
        while pending:
            done_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done_futures:
                record = pending.pop(future)
                try:
                    append_jsonl(paths["bundles"], future.result())
                except BaseException as exc:  # noqa: BLE001
                    append_jsonl(
                        paths["failures"],
                        {
                            "id": record.get("id"),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                pbar.update(1)
        pbar.close()

    counts = materialize_outputs(paths)
    failed_rows = sum(1 for _ in load_jsonl(paths["failures"])) if paths["failures"].exists() else 0
    report = {
        "input_jsonl": str(input_path),
        "records_jsonl": str(paths["records"]),
        "eval_bank_jsonl": str(paths["eval_bank"]),
        "assignments_jsonl": str(paths["assignments"]),
        "bundles_jsonl": str(paths["bundles"]),
        "failures_jsonl": str(paths["failures"]),
        "model": args.model,
        "requested_records": len(records),
        "failed_rows": failed_rows,
        "schema": "bidirectional_eval_bank_v2",
        **counts,
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report to {paths['report']}")


if __name__ == "__main__":
    main()
