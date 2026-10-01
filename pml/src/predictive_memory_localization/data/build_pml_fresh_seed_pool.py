from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from datasets import load_from_disk

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import append_jsonl
from predictive_memory_localization.data.download_pml_hf_sources import DEFAULT_SOURCES, safe_source_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize freshly downloaded HF datasets into a multi-domain PML seed pool."
    )
    parser.add_argument("--raw-root", default="/dev/shm/pml_datas")
    parser.add_argument(
        "--output-jsonl",
        default="/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000.seeds.jsonl",
    )
    parser.add_argument("--target-records", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--source-spec-json", default=None)
    parser.add_argument(
        "--backfill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If source quotas do not reach target_records, do a second pass over available sources.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_specs(path: Optional[str]) -> List[Dict[str, Any]]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return DEFAULT_SOURCES


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        return str(float(value))
    return str(value).strip()


def clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", as_text(value)).strip()
    return text


def choices_to_map(choices: Any) -> Dict[str, str]:
    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        return {as_text(label): clean_text(text) for label, text in zip(labels, texts)}
    if isinstance(choices, list):
        out = {}
        for idx, item in enumerate(choices):
            if isinstance(item, dict):
                out[as_text(item.get("label", idx))] = clean_text(item.get("text", ""))
            else:
                out[str(idx)] = clean_text(item)
        return out
    raise ValueError(f"Unsupported choices format: {type(choices)}")


def pick_answer_from_choices(choices: Dict[str, str], answer_key: Any) -> Optional[str]:
    key = as_text(answer_key)
    if key in choices:
        return choices[key]
    if key.isdigit():
        idx = int(key)
        values = list(choices.values())
        if 0 <= idx < len(values):
            return values[idx]
        if 1 <= idx <= len(values):
            return values[idx - 1]
    lowered = key.lower()
    for value in choices.values():
        if value.lower() == lowered:
            return value
    return None


def wrongs_from_choices(choices: Dict[str, str], correct: str) -> List[str]:
    out = []
    for value in choices.values():
        if value and value.strip().lower() != correct.strip().lower():
            out.append(value)
    return out


def normalize_openbookqa(row: Dict[str, Any]) -> Dict[str, Any]:
    choices = choices_to_map(row["choices"])
    correct = pick_answer_from_choices(choices, row.get("answerKey"))
    if not correct:
        raise ValueError("missing OpenBookQA correct answer")
    return {
        "source_id": clean_text(row.get("id")),
        "question": clean_text(row.get("question_stem")),
        "correct_answer": correct,
        "wrong_answers": wrongs_from_choices(choices, correct),
        "fact_or_concept": clean_text(row.get("fact1")),
    }


def normalize_arc(row: Dict[str, Any]) -> Dict[str, Any]:
    choices = choices_to_map(row["choices"])
    correct = pick_answer_from_choices(choices, row.get("answerKey"))
    if not correct:
        raise ValueError("missing ARC correct answer")
    return {
        "source_id": clean_text(row.get("id")),
        "question": clean_text(row.get("question")),
        "correct_answer": correct,
        "wrong_answers": wrongs_from_choices(choices, correct),
        "fact_or_concept": "",
    }


def normalize_sciq(row: Dict[str, Any]) -> Dict[str, Any]:
    correct = clean_text(row.get("correct_answer"))
    wrongs = [
        clean_text(row.get("distractor1")),
        clean_text(row.get("distractor2")),
        clean_text(row.get("distractor3")),
    ]
    return {
        "source_id": clean_text(row.get("id")),
        "question": clean_text(row.get("question")),
        "correct_answer": correct,
        "wrong_answers": [w for w in wrongs if w and w.lower() != correct.lower()],
        "fact_or_concept": clean_text(row.get("support")),
    }


def normalize_qasc(row: Dict[str, Any]) -> Dict[str, Any]:
    choices = choices_to_map(row["choices"])
    correct = pick_answer_from_choices(choices, row.get("answerKey"))
    if not correct:
        raise ValueError("missing QASC correct answer")
    facts = " ".join(clean_text(row.get(k)) for k in ["fact1", "fact2", "combinedfact"] if row.get(k))
    return {
        "source_id": clean_text(row.get("id")),
        "question": clean_text(row.get("question")),
        "correct_answer": correct,
        "wrong_answers": wrongs_from_choices(choices, correct),
        "fact_or_concept": facts,
    }


def normalize_commonsenseqa(row: Dict[str, Any]) -> Dict[str, Any]:
    choices = choices_to_map(row["choices"])
    correct = pick_answer_from_choices(choices, row.get("answerKey"))
    if not correct:
        raise ValueError("missing CommonsenseQA correct answer")
    return {
        "source_id": clean_text(row.get("id")),
        "question": clean_text(row.get("question")),
        "correct_answer": correct,
        "wrong_answers": wrongs_from_choices(choices, correct),
        "fact_or_concept": clean_text(row.get("question_concept")),
    }


def normalize_piqa(row: Dict[str, Any]) -> Dict[str, Any]:
    choices = {"0": clean_text(row.get("sol1")), "1": clean_text(row.get("sol2"))}
    correct = pick_answer_from_choices(choices, row.get("label"))
    if not correct:
        raise ValueError("missing PIQA correct answer")
    return {
        "source_id": clean_text(row.get("id")),
        "question": clean_text(row.get("goal")),
        "correct_answer": correct,
        "wrong_answers": wrongs_from_choices(choices, correct),
        "fact_or_concept": "physical interaction commonsense",
    }


def normalize_hellaswag(row: Dict[str, Any]) -> Dict[str, Any]:
    endings = [clean_text(x) for x in row.get("endings", [])]
    label = as_text(row.get("label"))
    if not label.isdigit() or int(label) >= len(endings):
        raise ValueError("missing HellaSwag correct answer")
    correct = endings[int(label)]
    question = clean_text(f"{row.get('ctx', '')}")
    return {
        "source_id": clean_text(row.get("ind") or row.get("source_id")),
        "question": question,
        "correct_answer": correct,
        "wrong_answers": [x for x in endings if x and x.lower() != correct.lower()],
        "fact_or_concept": clean_text(row.get("activity_label")),
    }


def normalize_boolq(row: Dict[str, Any]) -> Dict[str, Any]:
    answer = bool(row.get("answer"))
    passage = clean_text(row.get("passage"))
    question = clean_text(row.get("question"))
    correct = "yes" if answer else "no"
    wrong = "no" if answer else "yes"
    return {
        "source_id": clean_text(row.get("idx") or stable_key([question, passage[:80]])),
        "question": f"{question}?",
        "correct_answer": correct,
        "wrong_answers": [wrong],
        "fact_or_concept": passage[:800],
    }


def normalize_socialiqa(row: Dict[str, Any]) -> Dict[str, Any]:
    choices = {
        "1": clean_text(row.get("answerA")),
        "2": clean_text(row.get("answerB")),
        "3": clean_text(row.get("answerC")),
    }
    correct = pick_answer_from_choices(choices, row.get("label"))
    if not correct:
        raise ValueError("missing SocialIQA correct answer")
    context = clean_text(row.get("context"))
    question = clean_text(row.get("question"))
    return {
        "source_id": clean_text(row.get("id") or stable_key([context, question])),
        "question": f"{context} {question}",
        "correct_answer": correct,
        "wrong_answers": wrongs_from_choices(choices, correct),
        "fact_or_concept": "social commonsense",
    }


def normalize_truthfulqa_mc(row: Dict[str, Any]) -> Dict[str, Any]:
    choices_obj = row.get("mc1_targets") or row.get("mc2_targets")
    if not isinstance(choices_obj, dict):
        raise ValueError("missing TruthfulQA choices")
    choices = [clean_text(x) for x in choices_obj.get("choices", [])]
    labels = list(choices_obj.get("labels", []))
    corrects = [choice for choice, label in zip(choices, labels) if int(label) == 1]
    wrongs = [choice for choice, label in zip(choices, labels) if int(label) == 0]
    if not corrects or not wrongs:
        raise ValueError("missing TruthfulQA correct/wrong choices")
    return {
        "source_id": clean_text(row.get("id") or stable_key([row.get("question", "")])),
        "question": clean_text(row.get("question")),
        "correct_answer": corrects[0],
        "wrong_answers": wrongs,
        "fact_or_concept": clean_text(row.get("category")),
    }


def normalize_mmlu_pro(row: Dict[str, Any]) -> Dict[str, Any]:
    options = [clean_text(x) for x in row.get("options", []) if clean_text(x)]
    answer = row.get("answer")
    correct = ""
    if isinstance(answer, (int, np.integer)) and 0 <= int(answer) < len(options):
        correct = options[int(answer)]
    else:
        answer_text = clean_text(answer)
        if len(answer_text) == 1 and answer_text.upper().isalpha():
            idx = ord(answer_text.upper()) - ord("A")
            if 0 <= idx < len(options):
                correct = options[idx]
        if not correct and answer_text in options:
            correct = answer_text
    if not correct:
        raise ValueError("missing MMLU-Pro correct answer")
    return {
        "source_id": clean_text(row.get("question_id") or row.get("id") or stable_key([row.get("question", ""), correct])),
        "question": clean_text(row.get("question")),
        "correct_answer": correct,
        "wrong_answers": [x for x in options if x and x.lower() != correct.lower()],
        "fact_or_concept": clean_text(row.get("category") or row.get("src")),
    }


def normalize_mmlu_redux(row: Dict[str, Any]) -> Dict[str, Any]:
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except json.JSONDecodeError:
            choices = []
    options = [clean_text(x) for x in choices or [] if clean_text(x)]
    answer = row.get("answer") if "answer" in row else row.get("answer_index")
    correct = ""
    if isinstance(answer, (int, np.integer)) and 0 <= int(answer) < len(options):
        correct = options[int(answer)]
    else:
        answer_text = clean_text(answer)
        if len(answer_text) == 1 and answer_text.upper().isalpha():
            idx = ord(answer_text.upper()) - ord("A")
            if 0 <= idx < len(options):
                correct = options[idx]
        if not correct and answer_text in options:
            correct = answer_text
    if not correct:
        raise ValueError("missing MMLU-Redux correct answer")
    return {
        "source_id": clean_text(row.get("id") or stable_key([row.get("question", ""), correct])),
        "question": clean_text(row.get("question")),
        "correct_answer": correct,
        "wrong_answers": [x for x in options if x and x.lower() != correct.lower()],
        "fact_or_concept": clean_text(row.get("subject") or row.get("category")),
    }


def normalize_livebench_objective(row: Dict[str, Any]) -> Dict[str, Any]:
    question = clean_text(row.get("question") or row.get("turns") or row.get("prompt"))
    answer = row.get("answer")
    if isinstance(answer, list):
        correct = clean_text(answer[0] if answer else "")
    else:
        correct = clean_text(answer or row.get("ground_truth") or row.get("target"))
    if not correct:
        raise ValueError("missing LiveBench answer")
    wrongs = []
    for candidate in row.get("choices", []) or row.get("options", []) or []:
        text = clean_text(candidate)
        if text and text.lower() != correct.lower():
            wrongs.append(text)
    if not wrongs:
        if correct.lower() in {"yes", "true"}:
            wrongs = ["no"]
        elif correct.lower() in {"no", "false"}:
            wrongs = ["yes"]
        else:
            wrongs = ["not " + correct]
    return {
        "source_id": clean_text(row.get("id") or row.get("question_id") or stable_key([question, correct])),
        "question": question,
        "correct_answer": correct,
        "wrong_answers": wrongs,
        "fact_or_concept": clean_text(row.get("category") or row.get("task") or row.get("subtask")),
    }


NORMALIZERS = {
    "allenai/openbookqa": normalize_openbookqa,
    "allenai/ai2_arc": normalize_arc,
    "allenai/sciq": normalize_sciq,
    "allenai/qasc": normalize_qasc,
    "tau/commonsense_qa": normalize_commonsenseqa,
    "ybisk/piqa": normalize_piqa,
    "Rowan/hellaswag": normalize_hellaswag,
    "google/boolq": normalize_boolq,
    "allenai/social_i_qa": normalize_socialiqa,
    "EleutherAI/truthful_qa_mc": normalize_truthfulqa_mc,
    "TIGER-Lab/MMLU-Pro": normalize_mmlu_pro,
    "edinburgh-dawg/mmlu-redux-2.0": normalize_mmlu_redux,
    "livebench/reasoning": normalize_livebench_objective,
    "livebench/math": normalize_livebench_objective,
}


def stable_key(parts: List[Any]) -> str:
    import hashlib

    payload = "\u0000".join(clean_text(x) for x in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def namespace_source_id(spec: Dict[str, Any], source_id: str, fallback_idx: int) -> str:
    dataset = spec["dataset_name"].replace("/", "__")
    config = spec.get("dataset_config") or "default"
    sid = source_id or str(fallback_idx)
    sid = re.sub(r"[^A-Za-z0-9_.:-]+", "_", sid)
    return f"{dataset}::{config}::{sid}"


def load_source_dataset(raw_root: Path, spec: Dict[str, Any]):
    path = raw_root / "raw" / safe_source_name(spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw dataset at {path}. Run download_pml_hf_sources.py first.")
    return load_from_disk(str(path))


def iter_source_seeds(raw_root: Path, spec: Dict[str, Any], seed: int) -> Iterable[Dict[str, Any]]:
    dataset_name = spec["dataset_name"]
    if dataset_name not in NORMALIZERS:
        raise ValueError(f"No normalizer for {dataset_name}")
    normalizer = NORMALIZERS[dataset_name]
    dataset = load_source_dataset(raw_root, spec)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for fallback_idx, idx in enumerate(indices):
        row = dict(dataset[idx])
        item = normalizer(row)
        if not item["question"] or not item["correct_answer"] or not item["wrong_answers"]:
            continue
        item["dataset_name"] = dataset_name
        item["dataset_config"] = spec.get("dataset_config", "")
        item["split"] = spec.get("split", "train")
        item["domain"] = spec.get("domain", "")
        item["release_year"] = spec.get("release_year")
        item["freshness_group"] = spec.get("freshness_group", "")
        item["source_id"] = namespace_source_id(spec, item.get("source_id", ""), fallback_idx)
        item["seed_source"] = {
            "dataset_name": dataset_name,
            "dataset_config": spec.get("dataset_config", ""),
            "split": spec.get("split", "train"),
            "domain": spec.get("domain", ""),
            "release_year": spec.get("release_year"),
            "freshness_group": spec.get("freshness_group", ""),
            "raw_path": str(raw_root / "raw" / safe_source_name(spec)),
        }
        yield item


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        out_path.unlink(missing_ok=True)
    elif out_path.exists():
        raise FileExistsError(f"Output exists: {out_path}. Use --overwrite.")

    specs = source_specs(args.source_spec_json)
    written = 0
    seen = set()
    source_counts: Dict[str, int] = {}
    backfill_counts: Dict[str, int] = {}
    failures: List[Dict[str, str]] = []

    def write_seed_items(spec: Dict[str, Any], source_idx: int, limit: Optional[int], pass_name: str) -> int:
        nonlocal written
        source_key = f"{spec['dataset_name']}:{spec.get('dataset_config', '')}"
        count = 0
        try:
            iterator = iter_source_seeds(
                raw_root,
                spec,
                args.seed + source_idx * 997 + (0 if pass_name == "quota" else 1_000_003),
            )
            for seed_item in iterator:
                if written >= args.target_records:
                    break
                if limit is not None and count >= limit:
                    break
                key = (seed_item["dataset_name"], seed_item["source_id"])
                qa_key = (
                    seed_item["question"].lower(),
                    seed_item["correct_answer"].lower(),
                )
                if key in seen or qa_key in seen:
                    continue
                append_jsonl(out_path, seed_item)
                seen.add(key)
                seen.add(qa_key)
                count += 1
                written += 1
        except BaseException as exc:  # noqa: BLE001
            failures.append({"source": source_key, "pass": pass_name, "error": repr(exc)})
            print(f"WARNING: skipped source {source_key} in {pass_name} pass: {exc!r}", file=sys.stderr)
        return count

    for source_idx, spec in enumerate(specs):
        if written >= args.target_records:
            break
        source_key = f"{spec['dataset_name']}:{spec.get('dataset_config', '')}"
        source_counts[source_key] = write_seed_items(
            spec,
            source_idx,
            int(spec.get("target_records", args.target_records)),
            "quota",
        )

    if args.backfill and written < args.target_records:
        for source_idx, spec in enumerate(specs):
            if written >= args.target_records:
                break
            source_key = f"{spec['dataset_name']}:{spec.get('dataset_config', '')}"
            backfill_counts[source_key] = write_seed_items(spec, source_idx, None, "backfill")

    manifest = {
        "output_jsonl": str(out_path),
        "raw_root": str(raw_root),
        "target_records": args.target_records,
        "written": written,
        "source_counts": source_counts,
        "backfill_counts": backfill_counts,
        "failures": failures,
        "sources": specs,
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if written < args.target_records:
        raise SystemExit(f"Only wrote {written}/{args.target_records} seed rows")


if __name__ == "__main__":
    main()
