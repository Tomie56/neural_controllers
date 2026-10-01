from __future__ import annotations

import argparse
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

from predictive_memory_localization.common import append_jsonl, load_jsonl


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


SYSTEM_PROMPT = """You rewrite memory-intervention JSON records into a bidirectional steering evaluation schema.

Return exactly one JSON object. Do not include markdown, comments, or explanations.

The downstream experiment learns a steering direction from train_positive vs train_negative. It then evaluates both:
- negative alpha: suppression / weakening of the target fact
- positive alpha: enhancement / strengthening of the target fact

Therefore every eval item must be contrastive. Each eval item must contain:
- prompt: a short prefix
- correct: the true continuation, with leading whitespace when it starts with a word
- contrast: a plausible false or competing continuation, also with leading whitespace when it starts with a word

Core requirements:
- Preserve the original record id, concept, subject, source, seed_question, seed_answer, train_positive, train_negative, metadata, and fact_or_concept when present.
- Do not invent unsafe, sexual, medical, legal, private, political, hateful, or sensitive content.
- Keep continuations short and natural.
- The contrast continuation should be plausible enough to compete with the correct continuation, but clearly wrong for this prompt.
- Do not make contrast identical, synonymous, or trivially compatible with correct.
- Preserve the meaning of the original correct target in target_eval.
- For neighbor_eval, test nearby facts that should remain stable under target steering.
- For capability_eval, use unrelated ordinary commonsense or elementary-science QA-style probes that should remain stable under steering.
- If the input already has capability/unrelated probes, convert them to capability_eval and add contrasts.
- If the input has no capability probes, create exactly four unrelated capability_eval probes by using ordinary safe commonsense facts unrelated to the target record.
- For bidirectional steering, target_eval should support margin evaluation:
  margin = logprob(correct) - logprob(contrast)
- Negative alpha success will be measured by margin decrease.
- Positive alpha success will be measured by margin increase.

Output schema:
{
  "id": "...",
  "concept": "...",
  "subject": "...",
  "source": {...},
  "seed_question": "...",
  "seed_answer": "...",
  "train_positive": ["..."],
  "train_negative": ["..."],
  "target_eval": [
    {"prompt": "...", "correct": " ...", "contrast": " ...", "source_kind": "target"}
  ],
  "neighbor_eval": [
    {"prompt": "...", "correct": " ...", "contrast": " ...", "source_kind": "neighbor"}
  ],
  "capability_eval": [
    {"prompt": "...", "correct": " ...", "contrast": " ...", "source_kind": "capability"}
  ],
  "metadata": {
    "...": "...",
    "bidirectional_schema_version": "v1",
    "contrast_generation_model": "..."
  },
  "fact_or_concept": "..."
}

Also include a legacy-compatible view:
- eval: same as target_eval but with target=correct
- neighbors: same as neighbor_eval but with target=correct
- capability: same as capability_eval but with target=correct

This preserves compatibility with old scripts while adding contrastive fields for new bidirectional metrics.
"""


USER_TEMPLATE = """Rewrite this existing PML suppression record into the bidirectional steering schema.

Important semantic convention:
- The learned direction points from negative examples toward positive examples.
- Negative alpha should suppress/weaken the target fact.
- Positive alpha should enhance/strengthen the target fact.
- Therefore target_eval must have correct-vs-contrast margins.

Input record JSON:
{record_json}

Produce exactly one JSON object following the schema from the system message.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance existing PML suppression records into bidirectional correct-vs-contrast schema via OpenRouter."
    )
    parser.add_argument(
        "--input-jsonl",
        default="/data/neural_controllers/pml/data/suppression_commonsense_3000_valid_with_capability.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/data/neural_controllers/pml/data/main_datas/bidirectional_commonsense_3000.jsonl",
    )
    parser.add_argument(
        "--failures-jsonl",
        default="/data/neural_controllers/pml/data/main_datas/bidirectional_commonsense_3000_failures.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=OPENROUTER_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
    text = str(value)
    if not text:
        return text
    if text[0].isalnum():
        return " " + text
    return text


def normalize_eval_item(item: Dict[str, Any], source_kind: str) -> Dict[str, Any]:
    prompt = str(item.get("prompt", "")).strip()
    correct = ensure_leading_space(item.get("correct", item.get("target", "")))
    contrast = ensure_leading_space(item.get("contrast", item.get("incorrect", "")))
    if not prompt or not correct.strip() or not contrast.strip():
        raise ValueError(f"Invalid {source_kind} eval item: {item}")
    if correct.strip().lower() == contrast.strip().lower():
        raise ValueError(f"{source_kind} contrast equals correct: {item}")
    out = dict(item)
    out["prompt"] = prompt
    out["correct"] = correct
    out["contrast"] = contrast
    out["source_kind"] = source_kind
    return out


def legacy_from(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "prompt": item["prompt"],
            "target": item["correct"],
            "contrast": item["contrast"],
            "source_kind": item.get("source_kind"),
        }
        for item in items
    ]


def validate_and_normalize(record: Dict[str, Any], original: Dict[str, Any], model: str) -> Dict[str, Any]:
    for key in ["id", "train_positive", "train_negative"]:
        if key not in record:
            record[key] = original.get(key)
    if not record.get("id"):
        raise ValueError("Missing id")
    if not record.get("train_positive") or not record.get("train_negative"):
        raise ValueError(f"Record {record.get('id')} needs train_positive and train_negative")

    for key in ["concept", "subject", "source", "seed_question", "seed_answer", "fact_or_concept"]:
        if key not in record and key in original:
            record[key] = original[key]

    target_eval = [normalize_eval_item(item, "target") for item in record.get("target_eval", [])]
    neighbor_eval = [normalize_eval_item(item, "neighbor") for item in record.get("neighbor_eval", [])]
    capability_eval = [normalize_eval_item(item, "capability") for item in record.get("capability_eval", [])]

    if len(target_eval) < 1:
        raise ValueError(f"Record {record.get('id')} needs at least one target_eval")
    if len(neighbor_eval) < 1:
        raise ValueError(f"Record {record.get('id')} needs at least one neighbor_eval")
    if len(capability_eval) < 2:
        raise ValueError(f"Record {record.get('id')} needs at least two capability_eval items")

    record["target_eval"] = target_eval
    record["neighbor_eval"] = neighbor_eval
    record["capability_eval"] = capability_eval
    record["eval"] = legacy_from(target_eval)
    record["neighbors"] = legacy_from(neighbor_eval)
    record["capability"] = legacy_from(capability_eval)

    metadata = dict(original.get("metadata", {}))
    metadata.update(record.get("metadata", {}))
    metadata["bidirectional_schema_version"] = "v1"
    metadata["contrast_generation_model"] = model
    record["metadata"] = metadata
    return record


def call_openrouter(record: Dict[str, Any], args: argparse.Namespace, api_key: str) -> Dict[str, Any]:
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
            content = data["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            return validate_and_normalize(parsed, record, args.model)
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(args.sleep * (2**attempt))
    raise RuntimeError(f"OpenRouter call failed after {args.retries} retries: {last_error}") from last_error


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in load_jsonl(path):
        if row.get("id"):
            ids.add(str(row["id"]))
    return ids


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    failures_path = Path(args.failures_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)
    elif output_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {output_path}. Use --resume or --overwrite.")

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
        return

    api_key = get_api_key(args.api_key_env)
    done = existing_ids(output_path) if args.resume else set()
    todo = [record for record in records if str(record.get("id", "")) not in done]
    print(f"Input records: {len(records)}")
    print(f"Already done: {len(done)}")
    print(f"To process: {len(todo)}")
    print(f"Output: {output_path}")
    print(f"Failures: {failures_path}")

    def work(record: Dict[str, Any]) -> Dict[str, Any]:
        return call_openrouter(record, args, api_key)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {executor.submit(work, record): record for record in todo}
        pbar = tqdm(total=len(pending), desc="bidirectional schema")
        while pending:
            done_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done_futures:
                record = pending.pop(future)
                try:
                    result = future.result()
                    append_jsonl(output_path, result)
                except BaseException as exc:  # noqa: BLE001
                    append_jsonl(
                        failures_path,
                        {
                            "id": record.get("id"),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                pbar.update(1)
        pbar.close()

    report = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "failures_jsonl": str(failures_path),
        "model": args.model,
        "requested_records": len(records),
        "completed_records": len(existing_ids(output_path)),
        "failed_rows": sum(1 for _ in load_jsonl(failures_path)) if failures_path.exists() else 0,
        "schema": "bidirectional_v1",
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
