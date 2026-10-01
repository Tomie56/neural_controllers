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
DEFAULT_PREFIX = "/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000"


SYSTEM_PROMPT = """You create normalized data for a Predictive Memory Localization experiment.

Return exactly one JSON object. Do not include markdown, comments, or explanations.

The experiment learns an intervention direction from train_positive versus train_negative statements for one factual or commonsense memory. It then evaluates bidirectional activation steering:
- negative alpha should suppress / weaken the target fact
- positive alpha should enhance / strengthen the target fact

You must produce:
1. one intervention record, used only for learning the direction
2. target_eval probes, used to measure whether this exact fact changed
3. neighbor_eval probes, nearby facts that should not be damaged
4. capability_eval probes, unrelated general probes that should not be damaged

Important data design:
- Do not put eval probes inside the record object.
- The final pipeline will store eval probes in a global eval_bank and connect them by assignment.
- All eval probes must be contrastive: prompt, correct, contrast.
- Metric: margin = logprob(correct) - logprob(contrast).

Requirements for the intervention record:
- Use ordinary, broadly accepted factual, commonsense, science, physical, social, or world knowledge.
- The target memory should be expressible as a short subject-property/relation fact.
- train_positive must contain 8 to 12 short paraphrases supporting the correct fact.
- train_negative must contain 8 to 12 matched false/counterfactual paraphrases using one coherent wrong alternative.
- Positive and negative statements should be parallel where possible.
- Do not use the exact target_eval prompt as a training statement.

Requirements for eval probes:
- target_eval: exactly 3 held-out probes for the target fact.
- neighbor_eval: exactly 3 nearby but distinct facts that should remain stable.
- capability_eval: exactly 4 unrelated general probes from ordinary commonsense, science, physical, social, or world knowledge.
- Every eval item must have prompt, correct, contrast.
- correct and contrast are continuations. If they start with an English word, include a leading whitespace.
- contrast must be plausible but clearly wrong for the prompt.
- contrast must not be identical, synonymous, or trivially compatible with correct.
- Keep prompts and continuations short.

Safety and quality:
- Avoid unsafe, sexual, private, medical advice, legal advice, political persuasion, hateful, or sensitive personal content.
- Avoid vague opinions.
- Avoid examples whose correct answer is debatable.
- Avoid named living persons unless the source question requires an uncontroversial stable fact.
- Do not include chain-of-thought or explanations.

Output schema:
{
  "record": {
    "record_id": "short_snake_case_id",
    "concept": "short concept name",
    "subject": "short subject phrase",
    "source": {
      "dataset": "...",
      "dataset_config": "...",
      "source_id": "...",
      "domain": "...",
      "question": "...",
      "correct_answer": "..."
    },
    "seed_question": "...",
    "seed_answer": "...",
    "fact_or_concept": "...",
    "train_positive": ["..."],
    "train_negative": ["..."],
    "metadata": {
      "wrong_answer_used": "...",
      "schema_version": "pml_bidirectional_v1"
    }
  },
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
"""


USER_TEMPLATE = """Create one PML bidirectional data item from this source QA example.

Source dataset: {dataset_name}
Dataset config: {dataset_config}
Domain: {domain}
Release year: {release_year}
Freshness group: {freshness_group}
Source id: {source_id}
Question/context: {question}
Correct answer: {correct_answer}
Wrong answer candidates: {wrong_answers}
Supporting fact/concept/context if available: {fact_or_concept}

Choose a clean target memory implied by the source item. If the original answer is awkward as a short continuation, use the underlying factual relation as the target memory and keep the original answer in source metadata.

Return exactly one JSON object following the schema from the system message.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fresh normalized bidirectional PML records/eval_bank/assignments via OpenRouter."
    )
    parser.add_argument(
        "--seed-jsonl",
        default="/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000.seeds.jsonl",
    )
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=OPENROUTER_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-records", type=int, default=3000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2800)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    prefix = Path(args.output_prefix)
    return {
        "records": Path(f"{prefix}.records.jsonl"),
        "eval_bank": Path(f"{prefix}.eval_bank.jsonl"),
        "assignments": Path(f"{prefix}.eval_assignments.jsonl"),
        "bundles": Path(f"{prefix}.build_bundles.jsonl"),
        "failures": Path(f"{prefix}.failures.jsonl"),
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
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def snake_case(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] or "record"


def stable_hash(parts: List[Any], length: int = 16) -> str:
    payload = "\u0000".join(str(x) for x in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


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
        raise ValueError(f"Invalid {eval_type} item: {item}")
    if correct.strip().lower() == contrast.strip().lower():
        raise ValueError(f"{eval_type} contrast equals correct: {item}")
    return {
        "prompt": prompt,
        "correct": correct,
        "contrast": contrast,
        "source_kind": eval_type,
    }


def validate_and_normalize_bundle(bundle: Dict[str, Any], seed_item: Dict[str, Any], model: str) -> Dict[str, Any]:
    record = dict(bundle.get("record") or {})
    if not record:
        raise ValueError("missing record")
    record_id = str(record.get("record_id") or record.get("id") or "").strip()
    if not record_id:
        base = record.get("concept") or record.get("subject") or seed_item.get("correct_answer") or seed_item["source_id"]
        record_id = snake_case(str(base))
    record_id = snake_case(record_id)
    record["record_id"] = record_id
    record.pop("id", None)
    if len(record.get("train_positive", [])) < 6 or len(record.get("train_negative", [])) < 6:
        raise ValueError(f"{record_id} needs at least 6 positive and 6 negative training statements")

    source = dict(record.get("source", {}))
    source.update(
        {
            "dataset": seed_item["dataset_name"],
            "dataset_config": seed_item.get("dataset_config", ""),
            "source_id": seed_item["source_id"],
            "domain": seed_item.get("domain", ""),
            "release_year": seed_item.get("release_year"),
            "freshness_group": seed_item.get("freshness_group", ""),
            "question": seed_item["question"],
            "correct_answer": seed_item["correct_answer"],
        }
    )
    record["source"] = source
    record.setdefault("seed_question", seed_item["question"])
    record.setdefault("seed_answer", seed_item["correct_answer"])
    record.setdefault("fact_or_concept", seed_item.get("fact_or_concept", ""))
    metadata = dict(record.get("metadata", {}))
    metadata["schema_version"] = "pml_bidirectional_v1"
    metadata["generator_model"] = model
    metadata["seed_dataset"] = seed_item["dataset_name"]
    metadata["seed_domain"] = seed_item.get("domain", "")
    metadata["seed_release_year"] = seed_item.get("release_year")
    metadata["seed_freshness_group"] = seed_item.get("freshness_group", "")
    record["metadata"] = metadata

    forbidden_eval_keys = ["eval", "neighbors", "capability", "target_eval", "neighbor_eval", "capability_eval"]
    for key in forbidden_eval_keys:
        record.pop(key, None)

    target_eval = [normalize_eval_item(item, "target") for item in bundle.get("target_eval", [])]
    neighbor_eval = [normalize_eval_item(item, "neighbor") for item in bundle.get("neighbor_eval", [])]
    capability_eval = [normalize_eval_item(item, "capability") for item in bundle.get("capability_eval", [])]
    if len(target_eval) != 3:
        raise ValueError(f"{record_id} needs exactly 3 target_eval items")
    if len(neighbor_eval) != 3:
        raise ValueError(f"{record_id} needs exactly 3 neighbor_eval items")
    if len(capability_eval) != 4:
        raise ValueError(f"{record_id} needs exactly 4 capability_eval items")

    return {
        "record_id": record_id,
        "seed_source_key": source_key(seed_item),
        "record": record,
        "target_eval": target_eval,
        "neighbor_eval": neighbor_eval,
        "capability_eval": capability_eval,
    }


def call_openrouter(seed_item: Dict[str, Any], args: argparse.Namespace, api_key: str) -> Dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(
        dataset_name=seed_item["dataset_name"],
        dataset_config=seed_item.get("dataset_config", ""),
        domain=seed_item.get("domain", ""),
        release_year=seed_item.get("release_year", ""),
        freshness_group=seed_item.get("freshness_group", ""),
        source_id=seed_item["source_id"],
        question=seed_item["question"],
        correct_answer=seed_item["correct_answer"],
        wrong_answers=json.dumps(seed_item.get("wrong_answers", []), ensure_ascii=False),
        fact_or_concept=seed_item.get("fact_or_concept", ""),
    )
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Optional[BaseException] = None
    for attempt in range(1, args.retries + 1):
        try:
            response = requests.post(args.base_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content") or ""
            parsed = extract_json_object(content)
            return validate_and_normalize_bundle(parsed, seed_item, args.model)
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.sleep * (2**attempt))
    raise RuntimeError(f"OpenRouter failed after {args.retries} retries: {last_error}") from last_error


def make_eval_id(eval_type: str, record_id: str, index: int, item: Dict[str, Any]) -> str:
    if eval_type == "capability":
        return f"capability::{stable_hash([item['prompt'], item['correct']])}"
    return f"{eval_type}::{record_id}::{index:02d}"


def eval_bank_row(eval_id: str, eval_type: str, record_id: str, index: int, item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "eval_id": eval_id,
        "eval_type": eval_type,
        "prompt": item["prompt"],
        "correct": item["correct"],
        "contrast": item["contrast"],
        "source_record_id": record_id if eval_type in {"target", "neighbor"} else None,
        "source_eval_index": index,
        "metadata": {
            "source_kind": item.get("source_kind", eval_type),
        },
    }


def source_key(seed_item: Dict[str, Any]) -> str:
    return f"{seed_item.get('dataset_name', '')}::{seed_item.get('source_id', '')}"


def bundle_to_checkpoint(bundle: Dict[str, Any]) -> Dict[str, Any]:
    record_id = bundle["record_id"]
    assignment = {
        "record_id": record_id,
        "target_eval_ids": [],
        "neighbor_eval_ids": [],
        "capability_eval_ids": [],
        "assignment_policy": {
            "target": "record_specific",
            "neighbor": "record_specific",
            "capability": "generated_global_bank_candidate",
        },
    }
    eval_items = []
    for eval_type, key, assignment_key in [
        ("target", "target_eval", "target_eval_ids"),
        ("neighbor", "neighbor_eval", "neighbor_eval_ids"),
        ("capability", "capability_eval", "capability_eval_ids"),
    ]:
        for index, item in enumerate(bundle[key]):
            eval_id = make_eval_id(eval_type, record_id, index, item)
            eval_items.append(eval_bank_row(eval_id, eval_type, record_id, index, item))
            assignment[assignment_key].append(eval_id)
    return {
        "record_id": record_id,
        "seed_source_key": bundle.get("seed_source_key"),
        "record": bundle["record"],
        "eval_items": eval_items,
        "assignment": assignment,
    }


def existing_seed_source_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for row in load_jsonl(path):
        if row.get("seed_source_key"):
            out.add(str(row["seed_source_key"]))
    return out


def materialize_outputs(paths: Dict[str, Path]) -> Dict[str, int]:
    bundles = list(load_jsonl(paths["bundles"])) if paths["bundles"].exists() else []
    records: Dict[str, Dict[str, Any]] = {}
    eval_bank: Dict[str, Dict[str, Any]] = {}
    assignments: Dict[str, Dict[str, Any]] = {}
    conflicts = 0
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
                    conflicts += 1
                continue
            eval_bank[eval_id] = item
    write_jsonl(paths["records"], [records[k] for k in sorted(records)])
    write_jsonl(paths["eval_bank"], [eval_bank[k] for k in sorted(eval_bank)])
    write_jsonl(paths["assignments"], [assignments[k] for k in sorted(assignments)])
    return {
        "n_records": len(records),
        "n_eval_bank": len(eval_bank),
        "n_assignments": len(assignments),
        "n_eval_conflicts": conflicts,
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
        raise FileExistsError(f"Output exists: {paths['bundles']}. Use --resume or --overwrite.")
    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")

    seeds = list(load_jsonl(args.seed_jsonl))
    if args.offset:
        seeds = seeds[args.offset :]
    if args.max_records is not None:
        seeds = seeds[: args.max_records]

    if args.dry_run:
        print("SYSTEM_PROMPT:")
        print(SYSTEM_PROMPT)
        print("\nUSER_PROMPT_EXAMPLE:")
        print(USER_TEMPLATE.format(
            dataset_name=seeds[0]["dataset_name"],
            dataset_config=seeds[0].get("dataset_config", ""),
            domain=seeds[0].get("domain", ""),
            release_year=seeds[0].get("release_year", ""),
            freshness_group=seeds[0].get("freshness_group", ""),
            source_id=seeds[0]["source_id"],
            question=seeds[0]["question"],
            correct_answer=seeds[0]["correct_answer"],
            wrong_answers=json.dumps(seeds[0].get("wrong_answers", []), ensure_ascii=False),
            fact_or_concept=seeds[0].get("fact_or_concept", ""),
        ))
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    api_key = get_api_key(args.api_key_env)
    done_source_keys = existing_seed_source_keys(paths["bundles"]) if args.resume else set()
    todo = []
    for seed_item in seeds:
        if args.resume and source_key(seed_item) in done_source_keys:
            continue
        todo.append(seed_item)

    print(f"Seed rows: {len(seeds)}")
    print(f"Already bundled seed source keys: {len(done_source_keys)}")
    print(f"To process: {len(todo)}")
    print(f"Records: {paths['records']}")
    print(f"Eval bank: {paths['eval_bank']}")
    print(f"Assignments: {paths['assignments']}")
    print(f"Bundles checkpoint: {paths['bundles']}")
    print(f"Failures: {paths['failures']}")

    def work(seed_item: Dict[str, Any]) -> Dict[str, Any]:
        return bundle_to_checkpoint(call_openrouter(seed_item, args, api_key))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {executor.submit(work, seed_item): seed_item for seed_item in todo}
        pbar = tqdm(total=len(pending), desc="generate PML bidirectional")
        while pending:
            done_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done_futures:
                seed_item = pending.pop(future)
                try:
                    append_jsonl(paths["bundles"], future.result())
                except BaseException as exc:  # noqa: BLE001
                    append_jsonl(
                        paths["failures"],
                        {
                            "source_id": seed_item.get("source_id"),
                            "dataset_name": seed_item.get("dataset_name"),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                pbar.update(1)
        pbar.close()

    counts = materialize_outputs(paths)
    failed_rows = sum(1 for _ in load_jsonl(paths["failures"])) if paths["failures"].exists() else 0
    report = {
        "seed_jsonl": args.seed_jsonl,
        "records_jsonl": str(paths["records"]),
        "eval_bank_jsonl": str(paths["eval_bank"]),
        "assignments_jsonl": str(paths["assignments"]),
        "bundles_jsonl": str(paths["bundles"]),
        "failures_jsonl": str(paths["failures"]),
        "model": args.model,
        "requested_records": len(seeds),
        "failed_rows": failed_rows,
        "schema": "pml_bidirectional_v1",
        **counts,
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report to {paths['report']}")


if __name__ == "__main__":
    main()
