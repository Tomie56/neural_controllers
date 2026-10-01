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
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import requests
from datasets import load_dataset, load_from_disk
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import append_jsonl


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


SYSTEM_PROMPT = """You create clean JSONL records for a language-model memory intervention experiment.

The experiment learns a direction from positive versus negative statements, then suppresses that direction at inference time.

Return exactly one JSON object, with no markdown and no commentary.

Requirements:
- Use ordinary, broadly accepted commonsense or elementary science knowledge.
- The positive statements must all support the correct answer.
- The negative statements must be plausible but false or counterfactual versions using one wrong answer.
- Choose a short subject phrase from the question or fact. It should be a concept/entity whose property, behavior, or category can be completed by a short target.
- The first eval prompt must contain the exact subject phrase and must end just before the target continuation.
- Do not use the answer option as subject unless it also appears verbatim in the first eval prompt.
- Avoid unsafe, sexual, private, medical, legal, or politically sensitive content.
- Keep prompts short and natural.
- eval targets must include leading whitespace when the continuation starts with a word, for example " Paris".
- neighbors should test nearby facts that should be preserved under suppression.
- Do not include chain-of-thought or explanations.
"""


USER_TEMPLATE = """Convert this multiple-choice commonsense QA item into one suppression-data JSON object.

Source dataset: {dataset_name}
Source id: {source_id}
Question: {question}
Correct answer: {correct_answer}
Wrong answer candidates: {wrong_answers}
Open-book fact or concept, if available: {fact_or_concept}

Output schema:
{{
  "id": "short_snake_case_id",
  "concept": "short human-readable concept",
  "subject": "short subject phrase appearing verbatim in the first eval prompt",
  "source": {{
    "dataset": "{dataset_name}",
    "source_id": "{source_id}"
  }},
  "seed_question": "...",
  "seed_answer": "...",
  "train_positive": [
    "... 8 to 12 short positive statements or QA-style prompts ..."
  ],
  "train_negative": [
    "... 8 to 12 matched false/counterfactual statements using one wrong answer ..."
  ],
  "eval": [
    {{"prompt": "... contains the subject phrase ...", "target": " ..."}},
    {{"prompt": "...", "target": " ..."}}
  ],
  "neighbors": [
    {{"prompt": "...", "target": " ..."}},
    {{"prompt": "...", "target": " ..."}}
  ],
  "metadata": {{
    "wrong_answer_used": "...",
    "original_correct_answer": "{correct_answer}",
    "fact_or_concept": "..."
  }}
}}

Make train_positive and train_negative parallel where possible.
Make the first eval prompt contain the exact subject phrase and test the same knowledge as a short continuation.
Prefer relation-style prompts that work for model editing, for example:
- subject: "a Punnett square", prompt: "A Punnett square is used to", target: " chart the likelihood of inherited traits"
- subject: "camouflage", prompt: "Camouflage helps animals", target: " avoid predators"
- subject: "fruit salad", prompt: "A fruit salad is a mixture whose ingredients can be", target: " easily separated"
If the original answer is awkward as an edit target, use the underlying commonsense fact as the target and store the original answer in metadata.
Make neighbors related but not identical facts.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Predictive Memory Localization suppression JSONL via OpenRouter."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="allenai/openbookqa",
        help=(
            "Seed dataset name. Supported schemas include allenai/openbookqa, "
            "tau/commonsense_qa, allenai/ai2_arc, and piqa."
        ),
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="main",
        help="OpenBookQA uses config 'main'. Use empty string for datasets without config.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Optional local dataset path. Supports datasets.save_to_disk directories "
            "and Hugging Face parquet snapshots such as /path/openbookqa/main/train-*.parquet."
        ),
    )
    parser.add_argument(
        "--seed-jsonl",
        type=str,
        default=None,
        help=(
            "Optional normalized seed JSONL with fields dataset_name, source_id, "
            "question, correct_answer, wrong_answers, and fact_or_concept."
        ),
    )
    parser.add_argument(
        "--local-config",
        type=str,
        default=None,
        help="For local parquet snapshots, subdirectory/config name. Defaults to --dataset-config.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="/data/neural_controllers/pml/data/suppression_openbookqa.jsonl",
    )
    parser.add_argument(
        "--failures-jsonl",
        type=str,
        default="/data/neural_controllers/pml/data/suppression_openbookqa_failures.jsonl",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=OPENROUTER_URL)
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable that contains the OpenRouter API key.",
    )
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent OpenRouter requests. JSONL writes remain single-threaded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing output file. Existing source ids are skipped and "
            "--max-records is treated as the target number of generated records."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first normalized seed item and exit without calling the API.",
    )
    return parser.parse_args()


def get_api_key(env_name: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(f"Missing API key. Set {env_name}=<OPENROUTER_API_KEY>.")
    return key


def load_seed_dataset(args: argparse.Namespace):
    if args.seed_jsonl:
        return [json.loads(line) for line in Path(args.seed_jsonl).open(encoding="utf-8") if line.strip()]
    if args.dataset_path:
        path = Path(args.dataset_path)
        config = args.local_config or args.dataset_config
        parquet_candidates: List[Path] = []
        if path.is_file() and path.suffix == ".parquet":
            parquet_candidates = [path]
        elif (path / "dataset_info.json").exists() or (path / "state.json").exists():
            ds = load_from_disk(str(path))
            if hasattr(ds, "keys") and args.split in ds:
                return ds[args.split]
            return ds
        elif config and (path / config).exists():
            parquet_candidates = sorted((path / config).glob(f"{args.split}-*.parquet"))
        else:
            parquet_candidates = sorted(path.glob(f"{args.split}-*.parquet"))

        if parquet_candidates:
            frames = [pd.read_parquet(p) for p in parquet_candidates]
            if len(frames) == 1:
                return frames[0].to_dict(orient="records")
            return pd.concat(frames, ignore_index=True).to_dict(orient="records")

        raise FileNotFoundError(
            f"Could not find a load_from_disk dataset or parquet files for split={args.split} under {path}"
        )

    if args.dataset_config:
        return load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    return load_dataset(args.dataset_name, split=args.split)


def choices_to_map(choices: Any) -> Dict[str, str]:
    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        return {str(label): str(text) for label, text in zip(labels, texts)}
    if isinstance(choices, list):
        out = {}
        for item in choices:
            if isinstance(item, dict):
                out[str(item.get("label", len(out)))] = str(item.get("text", ""))
            else:
                out[str(len(out))] = str(item)
        return out
    raise ValueError(f"Unsupported choices format: {type(choices)}")


def normalize_answer_key(value: Any) -> str:
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        return str(int(value))
    return str(value)


def pick_answer_from_choices(choices: Dict[str, str], answer_key: Any) -> Optional[str]:
    key = normalize_answer_key(answer_key)
    if key in choices:
        return choices[key]
    if key.isdigit():
        idx = int(key)
        values = list(choices.values())
        if 0 <= idx < len(values):
            return values[idx]
        labels = list(choices.keys())
        if 1 <= idx <= len(values) and str(idx - 1) in labels:
            return choices[str(idx - 1)]
    lowered = key.lower().strip()
    for value in choices.values():
        if value.lower().strip() == lowered:
            return value
    return None


def answer_key_candidates(row: Dict[str, Any]) -> Sequence[Any]:
    return [
        row.get("answerKey"),
        row.get("answer_key"),
        row.get("label"),
        row.get("answer"),
        row.get("correct"),
        row.get("correct_answer"),
    ]


def normalize_seed_item(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    if {
        "dataset_name",
        "source_id",
        "question",
        "correct_answer",
        "wrong_answers",
    }.issubset(row.keys()):
        return {
            "dataset_name": str(row["dataset_name"]),
            "source_id": str(row["source_id"]),
            "question": str(row["question"]),
            "correct_answer": str(row["correct_answer"]),
            "wrong_answers": [str(x) for x in row.get("wrong_answers", []) if str(x)],
            "fact_or_concept": str(row.get("fact_or_concept", "")),
        }

    source_id = str(row.get("id", row.get("qid", row.get("question_id", ""))))

    if "choices" in row:
        choices = choices_to_map(row["choices"])
        correct_answer = None
        answer_key = None
        for candidate in answer_key_candidates(row):
            if candidate is None:
                continue
            answer_key = normalize_answer_key(candidate)
            correct_answer = pick_answer_from_choices(choices, candidate)
            if correct_answer is not None:
                break
        if correct_answer is None:
            raise ValueError(f"Could not map answer key into choices={choices}; row keys={sorted(row.keys())}")
        wrong_answers = [v for v in choices.values() if v and v != correct_answer]
    elif {"sol1", "sol2", "label"}.issubset(row.keys()):
        choices = {"0": str(row["sol1"]), "1": str(row["sol2"])}
        answer_key = normalize_answer_key(row["label"])
        correct_answer = pick_answer_from_choices(choices, answer_key)
        if correct_answer is None:
            raise ValueError(f"Could not map PIQA label={answer_key} into choices={choices}")
        wrong_answers = [v for v in choices.values() if v != correct_answer]
    else:
        raise ValueError(f"Unsupported seed row schema. Keys={sorted(row.keys())}")

    if "question_stem" in row:
        question = str(row["question_stem"])
    elif "goal" in row:
        question = str(row["goal"])
    else:
        question = str(row.get("question", ""))

    fact_or_concept = str(
        row.get("fact1")
        or row.get("question_concept")
        or row.get("fact")
        or row.get("topic")
        or ""
    )
    return {
        "dataset_name": dataset_name,
        "source_id": source_id,
        "question": question,
        "correct_answer": correct_answer,
        "wrong_answers": wrong_answers,
        "fact_or_concept": fact_or_concept,
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy/parquet values into JSON-serializable objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(x) for x in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return repr(value)


class OpenRouterGenerationError(RuntimeError):
    def __init__(self, message: str, content: str = ""):
        super().__init__(message)
        self.content = content


def validate_generated_record(record: Dict[str, Any]) -> None:
    required = ["id", "concept", "subject", "train_positive", "train_negative", "eval"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Generated record missing fields: {missing}")
    if len(record["train_positive"]) < 4 or len(record["train_negative"]) < 4:
        raise ValueError("Need at least 4 positive and 4 negative training prompts")
    if not record["eval"]:
        raise ValueError("Need at least one eval prompt")
    for item in record["eval"]:
        if "prompt" not in item or "target" not in item:
            raise ValueError("Each eval item needs prompt and target")
    subject = str(record.get("subject", "")).strip()
    if not subject:
        raise ValueError("Generated record subject is empty")
    first_prompt = str(record["eval"][0]["prompt"])
    if subject.lower() not in first_prompt.lower():
        raise ValueError(f"Subject {subject!r} is not in the first eval prompt")
    for item in record.get("neighbors", []):
        if "prompt" not in item or "target" not in item:
            raise ValueError("Each neighbor item needs prompt and target")


def repair_generated_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Apply conservative schema repairs before validation."""
    record.setdefault("metadata", {})
    subject = str(record.get("subject", "")).strip()
    eval_items = record.get("eval", [])
    if subject and eval_items:
        first_prompt = str(eval_items[0].get("prompt", ""))
        if subject.lower() not in first_prompt.lower():
            eval_items[0]["prompt"] = f"{subject}: {first_prompt}"
            record["metadata"]["subject_prompt_repaired"] = True
            record["metadata"]["subject_prompt_original"] = first_prompt
    return record


def source_key(dataset: Any, source_id: Any) -> str:
    return f"{dataset}::{source_id}"


def existing_source_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = row.get("source", {})
            dataset = source.get("dataset", "")
            source_id = source.get("source_id")
            if source_id is not None:
                out.add(source_key(dataset, source_id))
    return out


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def generation_job(
    *,
    idx: int,
    row: Dict[str, Any],
    dataset_name: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool,
    retries: int,
) -> Dict[str, Any]:
    try:
        seed_item = normalize_seed_item(row, dataset_name)
        record = call_openrouter(
            base_url=base_url,
            api_key=api_key,
            model=model,
            seed_item=seed_item,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning=reasoning,
            retries=retries,
        )
        return {
            "ok": True,
            "idx": idx,
            "seed_item": seed_item,
            "record": record,
        }
    except BaseException as exc:
        raw_content = getattr(exc, "content", "")
        return {
            "ok": False,
            "idx": idx,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "raw_content_prefix": raw_content[:2000] if raw_content else "",
            "row": json_safe(row),
        }


def call_openrouter(
    *,
    base_url: str,
    api_key: str,
    model: str,
    seed_item: Dict[str, Any],
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool,
    retries: int,
) -> Dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(
        dataset_name=seed_item["dataset_name"],
        source_id=seed_item["source_id"],
        question=seed_item["question"],
        correct_answer=seed_item["correct_answer"],
        wrong_answers=json.dumps(seed_item["wrong_answers"], ensure_ascii=False),
        fact_or_concept=seed_item["fact_or_concept"],
    )
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning:
        payload["reasoning"] = {"enabled": True}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(base_url, headers=headers, data=json.dumps(payload), timeout=120)
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            try:
                record = extract_json_object(content)
                record = repair_generated_record(record)
                validate_generated_record(record)
            except Exception as parse_exc:
                raise OpenRouterGenerationError(
                    f"Could not parse/validate model JSON: {parse_exc}",
                    content=content,
                ) from parse_exc
            record.setdefault("source", {})
            record["source"].update(
                {
                    "dataset": seed_item["dataset_name"],
                    "source_id": seed_item["source_id"],
                    "question": seed_item["question"],
                    "correct_answer": seed_item["correct_answer"],
                }
            )
            record.setdefault("metadata", {})
            record["metadata"].setdefault("generator_model", model)
            return record
        except BaseException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_path = Path(args.output_jsonl)
    failures_path = Path(args.failures_jsonl)
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)
    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")

    dataset = load_seed_dataset(args)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.resume:
        indices = indices[args.offset :]
    else:
        indices = indices[args.offset : args.offset + args.max_records]

    first_seed = normalize_seed_item(dict(dataset[indices[0]]), args.dataset_name)
    if args.dry_run:
        print(json.dumps(first_seed, ensure_ascii=False, indent=2))
        return

    api_key = get_api_key(args.api_key_env)
    seen_source_ids = existing_source_ids(output_path) if args.resume else set()
    if args.resume:
        print(
            f"Resume mode: found {count_jsonl_rows(output_path)} existing rows "
            f"and {len(seen_source_ids)} existing source ids in {output_path}"
        )

    pending_items: List[tuple[int, Dict[str, Any]]] = []
    for idx in indices:
        if args.resume and len(seen_source_ids) + len(pending_items) >= args.max_records:
            break
        row = dict(dataset[idx])
        try:
            seed_item = normalize_seed_item(row, args.dataset_name)
        except BaseException as exc:
            append_jsonl(
                failures_path,
                {
                    "dataset_index": idx,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "row": json_safe(row),
                },
            )
            continue
        seed_source_key = source_key(seed_item["dataset_name"], seed_item["source_id"])
        if args.resume and seed_source_key in seen_source_ids:
            continue
        pending_items.append((idx, row))

    target_remaining = max(args.max_records - count_jsonl_rows(output_path), 0) if args.resume else len(pending_items)
    if args.resume:
        # Keep a small retry cushion so a few repeatedly malformed API responses
        # do not prevent resume mode from filling the requested number of rows.
        retry_cushion = max(args.workers * max(args.retries, 1), target_remaining)
        pending_items = pending_items[: min(len(pending_items), target_remaining + retry_cushion)]
    else:
        pending_items = pending_items[:target_remaining]
    if not pending_items:
        print(f"No pending records to generate for {output_path}")
    elif args.workers <= 1:
        for idx, row in tqdm(pending_items, desc="generate suppression data"):
            result = generation_job(
                idx=idx,
                row=row,
                dataset_name=args.dataset_name,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                reasoning=args.reasoning,
                retries=args.retries,
            )
            if result["ok"]:
                record = result["record"]
                append_jsonl(output_path, record)
                seed_item = result["seed_item"]
                seen_source_ids.add(source_key(seed_item["dataset_name"], seed_item["source_id"]))
                time.sleep(args.sleep)
            else:
                append_jsonl(
                    failures_path,
                    {
                        "dataset_index": result["idx"],
                        "error": result["error"],
                        "traceback": result["traceback"],
                        "raw_content_prefix": result["raw_content_prefix"],
                        "row": result["row"],
                    },
                )
    else:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        try:
            futures = {}
            next_item = 0
            pbar = tqdm(total=len(pending_items), desc=f"generate suppression data x{args.workers}")

            def submit_until_full() -> None:
                nonlocal next_item
                while next_item < len(pending_items) and len(futures) < args.workers:
                    idx, row = pending_items[next_item]
                    next_item += 1
                    fut = executor.submit(
                        generation_job,
                        idx=idx,
                        row=row,
                        dataset_name=args.dataset_name,
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        reasoning=args.reasoning,
                        retries=args.retries,
                    )
                    futures[fut] = idx
                    if args.sleep:
                        time.sleep(args.sleep)

            submit_until_full()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    futures.pop(fut, None)
                    result = fut.result()
                    if result["ok"]:
                        record = result["record"]
                        append_jsonl(output_path, record)
                        seed_item = result["seed_item"]
                        seen_source_ids.add(source_key(seed_item["dataset_name"], seed_item["source_id"]))
                    else:
                        append_jsonl(
                            failures_path,
                            {
                                "dataset_index": result["idx"],
                                "error": result["error"],
                                "traceback": result["traceback"],
                                "raw_content_prefix": result["raw_content_prefix"],
                                "row": result["row"],
                            },
                        )
                    pbar.update(1)
                if count_jsonl_rows(output_path) >= args.max_records:
                    break
                submit_until_full()
            pbar.close()
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    print(f"Wrote generated records to {output_path}")
    if failures_path.exists():
        print(f"Wrote failures to {failures_path}")


if __name__ == "__main__":
    main()
