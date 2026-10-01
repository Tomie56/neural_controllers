from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization import generation_utils
from predictive_memory_localization.common import (
    append_jsonl,
    load_jsonl,
    load_model_and_tokenizer,
    set_seed,
    write_jsonl,
)
from predictive_memory_localization.methods.run_bidirectional_three_table import (
    DirectionController,
    as_train_data,
    compute_directions,
    eval_items_for_record,
    load_tables,
    projection_metrics,
    summarize_localization_features,
)


DEFAULT_DATA_DIR = "/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen"
DEFAULT_MODEL_PATH = "/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base"
DEFAULT_ALPHAS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_LAYERS = [-17]
ANSWER_MATCH_VERSION = "first_clause_token_boundary_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run endpoint generation sanity checks for bidirectional PML three-table data."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-local-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default="qwen3_1_7b")
    parser.add_argument(
        "--model-loader",
        default="auto",
        choices=["auto", "causal_lm", "image_text_to_text"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--control-method",
        default="mean_difference",
        choices=["linear", "logistic", "mean_difference", "random", "rfm_agop_top1"],
    )
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--record-offset", type=int, default=500)
    parser.add_argument(
        "--record-ids-file",
        default=None,
        help="Optional text file with one record_id per line. If provided, record-offset is ignored.",
    )
    parser.add_argument(
        "--record-ids",
        nargs="*",
        default=None,
        help="Optional explicit record_id list. If provided with --record-ids-file, the union is used.",
    )
    parser.add_argument("--layers", type=int, nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument("--alphas", type=float, nargs="*", default=DEFAULT_ALPHAS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--target-evals-per-record", type=int, default=1)
    parser.add_argument("--neighbor-evals-per-record", type=int, default=1)
    parser.add_argument("--capability-evals-per-record", type=int, default=1)
    parser.add_argument("--min-nonspace-chars", type=int, default=1)
    parser.add_argument("--repeat-ngram", type=int, default=3)
    parser.add_argument("--repeat-threshold", type=float, default=0.35)
    parser.add_argument("--answer-prefix-tokens", type=int, default=24)
    parser.add_argument("--rfm-iters", type=int, default=3)
    parser.add_argument("--rfm-reg", type=float, default=1e-3)
    parser.add_argument("--rfm-bandwidth", type=float, default=10.0)
    parser.add_argument("--rfm-kernel", type=str, default="laplace")
    parser.add_argument("--rfm-mem-gb", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    return text


def normalize_endpoint_reference(text: str) -> str:
    return str(text).replace("\x00b0", "°").replace("\x00", "")


TOKEN_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?|[a-z]+(?:'[a-z]+)?")
TOKEN_ALIASES = {
    "volts": "v",
    "volt": "v",
    "watts": "w",
    "watt": "w",
    "joules": "j",
    "joule": "j",
    "kilojoules": "kj",
    "kilojoule": "kj",
    "meters": "m",
    "meter": "m",
    "seconds": "s",
    "second": "s",
}


def first_answer_clause(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return re.split(r"\r?\n|[!?;]|(?<!\d)\.(?!\d)", stripped, maxsplit=1)[0].strip()


def answer_tokens(text: str, max_tokens: Optional[int] = None) -> List[str]:
    tokens = [TOKEN_ALIASES.get(token, token) for token in TOKEN_PATTERN.findall(text.lower())]
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def contains_token_sequence(tokens: Sequence[str], answer: Sequence[str]) -> bool:
    if not answer or len(answer) > len(tokens):
        return False
    width = len(answer)
    return any(list(tokens[index : index + width]) == list(answer) for index in range(len(tokens) - width + 1))


def contains_answer(generated: str, answer: str, max_prefix_tokens: int = 24) -> bool:
    generated_tokens = answer_tokens(
        first_answer_clause(normalize_endpoint_reference(generated)),
        max_tokens=max_prefix_tokens,
    )
    expected_tokens = answer_tokens(normalize_endpoint_reference(answer))
    return contains_token_sequence(generated_tokens, expected_tokens)


def repetition_ratio(text: str, n: int) -> float:
    tokens = normalize_text(text).split()
    if len(tokens) < n or n <= 0:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def collapse_metrics(text: str, args: argparse.Namespace) -> Dict[str, Any]:
    stripped = text.strip()
    rep = repetition_ratio(stripped, args.repeat_ngram)
    empty = len(stripped) == 0
    too_short = len(re.sub(r"\s+", "", stripped)) < args.min_nonspace_chars
    replacement_chars = stripped.count("\ufffd")
    collapse = bool(empty or too_short or rep >= args.repeat_threshold or replacement_chars > 0)
    return {
        "generated_chars": len(stripped),
        "generated_words": len(stripped.split()),
        "empty_generation": empty,
        "too_short_generation": too_short,
        "replacement_char_count": replacement_chars,
        "repeat_ngram": args.repeat_ngram,
        "repeat_ratio": rep,
        "collapse": collapse,
    }


@torch.no_grad()
def generate_with_control(
    controller: DirectionController,
    prompt: str,
    layers: Sequence[int],
    alpha: float,
    args: argparse.Namespace,
) -> str:
    hooks = {}
    try:
        if abs(alpha) > 0:
            hooks = generation_utils.hook_model(controller.model, controller.directions, layers, alpha)
        encoded = controller.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(controller.model.device)
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": controller.tokenizer.pad_token_id,
            "eos_token_id": controller.tokenizer.eos_token_id,
        }
        if args.temperature and args.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
        else:
            generation_kwargs.update({"do_sample": False})
        output_ids = controller.model.generate(**encoded, **generation_kwargs)
        new_ids = output_ids[0, encoded["input_ids"].shape[1] :]
        return controller.tokenizer.decode(new_ids, skip_special_tokens=True)
    finally:
        if hooks:
            generation_utils.clear_hooks(hooks)


def selected_eval_items(
    record_id: str,
    assignment: Dict[str, Any],
    eval_bank: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    limits = {
        "target": args.target_evals_per_record,
        "neighbor": args.neighbor_evals_per_record,
        "capability": args.capability_evals_per_record,
    }
    items = eval_items_for_record(record_id, assignment, eval_bank)
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        eval_type = item["eval_type"]
        if counts[eval_type] >= limits[eval_type]:
            continue
        selected.append(item)
        counts[eval_type] += 1
    return selected


def run_record(
    record: Dict[str, Any],
    assignment: Dict[str, Any],
    eval_bank: Dict[str, Dict[str, Any]],
    controller: DirectionController,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    record_id = record["record_id"]
    train_prompts, train_labels = as_train_data(record)
    hidden_layers = list(args.layers)
    directions, hidden_states = compute_directions(
        controller.model,
        controller.tokenizer,
        train_prompts,
        train_labels,
        hidden_layers,
        args.batch_size,
        args.control_method,
        args.seed,
        args,
    )
    controller.directions = directions
    controller.hidden_layers = hidden_layers
    projection = projection_metrics(directions, hidden_states, train_labels)
    feature_summary = summarize_localization_features(directions, projection)

    eval_rows = []
    for item in selected_eval_items(record_id, assignment, eval_bank, args):
        correct = normalize_endpoint_reference(item["correct"])
        contrast = normalize_endpoint_reference(item["contrast"])
        alpha_rows = []
        for alpha in args.alphas:
            generated = generate_with_control(controller, item["prompt"], hidden_layers, alpha, args)
            answer_clause = first_answer_clause(generated)
            correct_hit = contains_answer(generated, correct, args.answer_prefix_tokens)
            contrast_hit = contains_answer(generated, contrast, args.answer_prefix_tokens)
            metrics = collapse_metrics(generated, args)
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "generated": generated,
                    "answer_clause": answer_clause,
                    "correct_hit": correct_hit,
                    "contrast_hit": contrast_hit,
                    "endpoint_correct": bool(correct_hit and not contrast_hit),
                    "endpoint_wrong_contrast": bool(contrast_hit and not correct_hit),
                    **metrics,
                }
            )
        eval_rows.append(
            {
                "eval_id": item["eval_id"],
                "eval_type": item["eval_type"],
                "assignment_index": item["assignment_index"],
                "prompt": item["prompt"],
                "correct": correct,
                "contrast": contrast,
                "alphas": alpha_rows,
            }
        )

    return {
        "record_id": record_id,
        "concept": record.get("concept"),
        "subject": record.get("subject"),
        "source": record.get("source", {}),
        "metadata": record.get("metadata", {}),
        "control_method": args.control_method,
        "layers": hidden_layers,
        "projection_metrics": projection,
        "feature_summary": feature_summary,
        "eval": eval_rows,
    }


def flatten_per_eval(result: Dict[str, Any], answer_prefix_tokens: int = 24) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    features = result.get("feature_summary", {})
    source = result.get("source", {})
    metadata = result.get("metadata", {})
    for item in result.get("eval", []):
        correct = normalize_endpoint_reference(item["correct"])
        contrast = normalize_endpoint_reference(item["contrast"])
        rescored = []
        for alpha_row in item["alphas"]:
            correct_hit = contains_answer(alpha_row["generated"], correct, answer_prefix_tokens)
            contrast_hit = contains_answer(alpha_row["generated"], contrast, answer_prefix_tokens)
            rescored.append(
                {
                    **alpha_row,
                    "correct_hit": correct_hit,
                    "contrast_hit": contrast_hit,
                    "endpoint_correct": bool(correct_hit and not contrast_hit),
                    "endpoint_wrong_contrast": bool(contrast_hit and not correct_hit),
                }
            )
        base = next((row for row in rescored if abs(float(row["alpha"])) == 0), None)
        base_endpoint_correct = bool(base.get("endpoint_correct")) if base else False
        for alpha_row in rescored:
            rows.append(
                {
                    "record_id": result["record_id"],
                    "concept": result.get("concept"),
                    "subject": result.get("subject"),
                    "dataset": source.get("dataset"),
                    "domain": source.get("domain"),
                    "freshness_group": metadata.get("seed_freshness_group"),
                    "release_year": metadata.get("seed_release_year"),
                    "control_method": result["control_method"],
                    "layer": ",".join(str(x) for x in result.get("layers", [])),
                    "eval_id": item["eval_id"],
                    "eval_type": item["eval_type"],
                    "assignment_index": item["assignment_index"],
                    "prompt": item["prompt"],
                    "correct": correct,
                    "contrast": contrast,
                    "alpha": alpha_row["alpha"],
                    "generated": alpha_row["generated"],
                    "answer_clause": alpha_row.get("answer_clause", ""),
                    "base_endpoint_correct": base_endpoint_correct,
                    "endpoint_correct": alpha_row["endpoint_correct"],
                    "endpoint_wrong_contrast": alpha_row["endpoint_wrong_contrast"],
                    "correct_hit": alpha_row["correct_hit"],
                    "contrast_hit": alpha_row["contrast_hit"],
                    "collapse": alpha_row["collapse"],
                    "generated_chars": alpha_row["generated_chars"],
                    "generated_words": alpha_row["generated_words"],
                    "repeat_ratio": alpha_row["repeat_ratio"],
                    "endpoint_damage_from_base": bool(base_endpoint_correct and not alpha_row["endpoint_correct"]),
                    "endpoint_gain_from_base": bool((not base_endpoint_correct) and alpha_row["endpoint_correct"]),
                    **{f"feature_{k}": v for k, v in features.items() if not isinstance(v, list)},
                }
            )
    return rows


def mean(values: Sequence[Any]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_per_record(per_eval: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_eval:
        groups[(row["record_id"], float(row["alpha"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (record_id, alpha), rows in sorted(groups.items()):
        first = rows[0]
        by_type = {typ: [r for r in rows if r["eval_type"] == typ] for typ in ["target", "neighbor", "capability"]}
        target = by_type["target"]
        neighbor = by_type["neighbor"]
        capability = by_type["capability"]
        out.append(
            {
                "record_id": record_id,
                "control_method": first["control_method"],
                "layer": first["layer"],
                "dataset": first.get("dataset"),
                "domain": first.get("domain"),
                "freshness_group": first.get("freshness_group"),
                "release_year": first.get("release_year"),
                "alpha": alpha,
                "target_endpoint_correct_rate": mean([r["endpoint_correct"] for r in target]),
                "target_endpoint_gain_rate": mean([r["endpoint_gain_from_base"] for r in target]),
                "target_endpoint_damage_from_base_rate": mean([r["endpoint_damage_from_base"] for r in target]),
                "target_wrong_contrast_rate": mean([r["endpoint_wrong_contrast"] for r in target]),
                "neighbor_endpoint_correct_rate": mean([r["endpoint_correct"] for r in neighbor]),
                "neighbor_endpoint_damage_from_base_rate": mean([r["endpoint_damage_from_base"] for r in neighbor]),
                "capability_endpoint_correct_rate": mean([r["endpoint_correct"] for r in capability]),
                "capability_endpoint_damage_from_base_rate": mean([r["endpoint_damage_from_base"] for r in capability]),
                "collapse_rate": mean([r["collapse"] for r in rows]),
                "mean_generated_chars": mean([r["generated_chars"] for r in rows]),
                "mean_repeat_ratio": mean([r["repeat_ratio"] for r in rows]),
                "n_target_eval": len(target),
                "n_neighbor_eval": len(neighbor),
                "n_capability_eval": len(capability),
                **{k: v for k, v in first.items() if k.startswith("feature_")},
            }
        )
    return out


def summarize_alpha(per_record: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_record:
        groups[float(row["alpha"])].append(row)
    out: List[Dict[str, Any]] = []
    for alpha, rows in sorted(groups.items()):
        out.append(
            {
                "alpha": alpha,
                "n_records": len(rows),
                "target_endpoint_correct_rate": mean([r["target_endpoint_correct_rate"] for r in rows]),
                "target_endpoint_gain_rate": mean([r["target_endpoint_gain_rate"] for r in rows]),
                "target_endpoint_damage_from_base_rate": mean([r["target_endpoint_damage_from_base_rate"] for r in rows]),
                "target_wrong_contrast_rate": mean([r["target_wrong_contrast_rate"] for r in rows]),
                "neighbor_endpoint_correct_rate": mean([r["neighbor_endpoint_correct_rate"] for r in rows]),
                "neighbor_endpoint_damage_from_base_rate": mean([r["neighbor_endpoint_damage_from_base_rate"] for r in rows]),
                "capability_endpoint_correct_rate": mean([r["capability_endpoint_correct_rate"] for r in rows]),
                "capability_endpoint_damage_from_base_rate": mean([r["capability_endpoint_damage_from_base_rate"] for r in rows]),
                "collapse_rate": mean([r["collapse_rate"] for r in rows]),
                "mean_generated_chars": mean([r["mean_generated_chars"] for r in rows]),
                "mean_repeat_ratio": mean([r["mean_repeat_ratio"] for r in rows]),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: value.replace("\x00", "\\x00") if isinstance(value, str) else value
                for key, value in row.items()
            }
            for row in rows
        )


def write_outputs_from_raw(out_dir: Path, raw_path: Path, args: argparse.Namespace, elapsed: float) -> None:
    per_eval: List[Dict[str, Any]] = []
    for result in load_jsonl(raw_path):
        per_eval.extend(flatten_per_eval(result, args.answer_prefix_tokens))
    successful_ids = existing_record_ids(raw_path)
    failures_path = out_dir / "failures.jsonl"
    retain_unresolved_failures(failures_path, successful_ids)
    per_record = summarize_per_record(per_eval)
    alpha_summary = summarize_alpha(per_record)
    summary = {
        "n_records": len({row["record_id"] for row in per_record}),
        "n_per_eval_rows": len(per_eval),
        "n_per_record_rows": len(per_record),
        "alphas": sorted({float(row["alpha"]) for row in per_record}),
        "control_method": args.control_method,
        "layers": list(args.layers),
        "elapsed_seconds": elapsed,
        "n_failures": sum(1 for _ in load_jsonl(failures_path)) if failures_path.exists() else 0,
    }
    write_csv(out_dir / "per_eval_endpoint.csv", per_eval)
    write_csv(out_dir / "per_record_endpoint.csv", per_record)
    write_csv(out_dir / "alpha_endpoint_summary.csv", alpha_summary)
    (out_dir / "endpoint_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["record_id"]) for row in load_jsonl(path) if row.get("record_id")}


def retain_unresolved_failures(path: Path, successful_ids: set[str]) -> None:
    if not path.exists():
        return
    latest_by_record: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        record_id = str(row.get("record_id", ""))
        if record_id and record_id not in successful_ids:
            latest_by_record[record_id] = row
    write_jsonl(path, latest_by_record.values())


def run_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {**vars(args), "answer_match_version": ANSWER_MATCH_VERSION}


def validate_resume_config(out_dir: Path, args: argparse.Namespace) -> None:
    config_path = out_dir / "run_config.json"
    if not args.resume or not config_path.exists():
        return
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    current = run_config(args)
    fixed_keys = [
        "data_dir",
        "model_local_path",
        "model_name",
        "model_loader",
        "trust_remote_code",
        "control_method",
        "record_ids_file",
        "layers",
        "alphas",
        "seed",
        "torch_dtype",
        "device_map",
        "target_evals_per_record",
        "neighbor_evals_per_record",
        "capability_evals_per_record",
        "answer_prefix_tokens",
        "answer_match_version",
        "rfm_iters",
        "rfm_reg",
        "rfm_bandwidth",
        "rfm_kernel",
    ]
    mismatches = {
        key: {"previous": previous.get(key), "current": current.get(key)}
        for key in fixed_keys
        if previous.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(
            "Resume configuration does not match the existing endpoint run. "
            f"Use a new output directory or --overwrite. Mismatches: {mismatches}"
        )


def requested_record_ids(args: argparse.Namespace) -> Optional[set[str]]:
    ids: set[str] = set()
    if args.record_ids_file:
        path = Path(args.record_ids_file)
        with path.open("r", encoding="utf-8") as f:
            ids.update(line.strip() for line in f if line.strip())
    if args.record_ids:
        ids.update(str(record_id).strip() for record_id in args.record_ids if str(record_id).strip())
    return ids or None


def main() -> None:
    args = parse_args()
    if sum([args.overwrite, args.resume, args.postprocess_only]) > 1:
        raise ValueError("Use only one of --overwrite, --resume, or --postprocess-only.")
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_endpoint_generations.jsonl"
    failures_path = out_dir / "failures.jsonl"
    if args.postprocess_only:
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw endpoint results: {raw_path}")
        write_outputs_from_raw(out_dir, raw_path, args, elapsed=0.0)
        return
    validate_resume_config(out_dir, args)
    if args.overwrite:
        for name in [
            "raw_endpoint_generations.jsonl",
            "failures.jsonl",
            "per_eval_endpoint.csv",
            "per_record_endpoint.csv",
            "alpha_endpoint_summary.csv",
            "endpoint_summary.json",
            "run_config.json",
        ]:
            (out_dir / name).unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {raw_path}. Use --overwrite or --resume.")

    (out_dir / "run_config.json").write_text(
        json.dumps(run_config(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records, eval_bank, assignments = load_tables(Path(args.data_dir))
    selected_ids = requested_record_ids(args)
    if selected_ids is not None:
        records = [record for record in records if str(record["record_id"]) in selected_ids]
        missing_ids = sorted(selected_ids - {str(record["record_id"]) for record in records})
        if missing_ids:
            print(f"WARNING: {len(missing_ids)} requested record_ids were not found; first_missing={missing_ids[:5]}")
    else:
        records = records[args.record_offset :]
    if args.max_records is not None:
        records = records[: args.max_records]
    done = existing_record_ids(raw_path) if args.resume else set()
    if done:
        records = [record for record in records if str(record["record_id"]) not in done]
        print(f"Resume mode: skipping {len(done)} existing record_ids; remaining={len(records)}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_local_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        model_loader=args.model_loader,
        trust_remote_code=args.trust_remote_code,
    )
    controller = DirectionController(
        model=model,
        tokenizer=tokenizer,
        directions={},
        hidden_layers=list(args.layers),
        batch_size=args.batch_size,
    )

    t0 = time.time()
    for record in tqdm(records, desc=f"{args.control_method} endpoint records"):
        try:
            result = run_record(record, assignments[record["record_id"]], eval_bank, controller, args)
            append_jsonl(raw_path, result)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                failures_path,
                {
                    "record_id": record.get("record_id"),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_outputs_from_raw(out_dir, raw_path, args, elapsed=time.time() - t0)


if __name__ == "__main__":
    main()
