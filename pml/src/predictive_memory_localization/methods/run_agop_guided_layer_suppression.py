from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import (
    append_jsonl,
    continuation_logprob,
    load_jsonl,
    load_model_and_tokenizer,
    maybe_format_prompt,
    set_seed,
    validate_suppression_record,
)
from predictive_memory_localization import generation_utils
from predictive_memory_localization.methods.extract_agop_features import (
    agop_layer_features,
    fit_rfm_agop,
    summarize_agop_features,
)
from predictive_memory_localization.methods.run_activation_suppression import (
    as_train_data,
    fit_linear_direction,
    fit_logistic_direction,
    fit_mean_difference_direction,
    get_hidden_states,
    projection_metrics,
    summarize_localization_features,
)

AGOP_DEFAULT_LAYERS = [-1, -5, -9, -13, -17, -21]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGOP-guided layer selection with simple suppression directions.")
    parser.add_argument("--model-local-path", default="/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--direction-method",
        choices=["mean_difference", "logistic", "linear"],
        default="mean_difference",
        help="Direction fitted at the AGOP-selected layer.",
    )
    parser.add_argument(
        "--layer-selection",
        choices=["agop_top1_ratio", "agop_top_vec_cohen_d", "agop_top_vec_mean_gap"],
        default="agop_top1_ratio",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, nargs="*", default=AGOP_DEFAULT_LAYERS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--format-prompts", action="store_true")
    parser.add_argument("--rfm-iters", type=int, default=3)
    parser.add_argument("--rfm-reg", type=float, default=1e-3)
    parser.add_argument("--rfm-bandwidth", type=float, default=10.0)
    parser.add_argument("--rfm-kernel", type=str, default="laplace")
    parser.add_argument("--rfm-mem-gb", type=float, default=8.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--suppression-coefs",
        type=float,
        nargs="*",
        default=[-0.1, -0.25, -0.5, -1.0],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fit_direction(X: torch.Tensor, labels: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean_difference":
        return fit_mean_difference_direction(X, labels)
    if method == "logistic":
        return fit_logistic_direction(X, labels)
    if method == "linear":
        return fit_linear_direction(X, labels)
    raise ValueError(f"Unsupported direction method: {method}")


def orient_direction(vec: torch.Tensor, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scores = X.float() @ vec.reshape(-1).float()
    if scores[labels.bool()].mean() < scores[~labels.bool()].mean():
        vec = -vec
    return vec.reshape(1, -1)


def select_layer(layer_features: Dict[int, Dict[str, Any]], metric: str) -> int:
    key = metric.replace("agop_", "", 1)
    return max(layer_features, key=lambda layer: abs(float(layer_features[layer].get(key, 0.0))))


def score_with_optional_control(
    model,
    tokenizer,
    prompt: str,
    target: str,
    directions: Dict[int, torch.Tensor],
    selected_layer: int,
    coef: Optional[float],
) -> Dict[str, Any]:
    hooks = {}
    try:
        if coef is not None:
            hooks = generation_utils.hook_model(model, directions, [selected_layer], coef)
        return continuation_logprob(model, tokenizer, prompt, target)
    finally:
        if hooks:
            generation_utils.clear_hooks(hooks)


def run_record(record: Dict[str, Any], model, tokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    validate_suppression_record(record)
    train_prompts, train_labels = as_train_data(record, tokenizer, args.format_prompts)
    labels = torch.tensor(train_labels).long()
    hidden_layers = list(args.layers)
    hidden_states = get_hidden_states(train_prompts, model, tokenizer, hidden_layers, args.batch_size)

    agop_layer_metrics: Dict[int, Dict[str, Any]] = {}
    for layer in hidden_layers:
        agop = fit_rfm_agop(hidden_states[layer], train_labels, args)
        agop_layer_metrics[int(layer)] = agop_layer_features(agop, hidden_states[layer], train_labels, args.top_k)
    selected_layer = select_layer(agop_layer_metrics, args.layer_selection)

    selected_vec = fit_direction(hidden_states[selected_layer], labels, args.direction_method)
    selected_vec = orient_direction(selected_vec, hidden_states[selected_layer], labels)
    directions = {selected_layer: selected_vec}

    local_metrics = projection_metrics(directions, {selected_layer: hidden_states[selected_layer]}, train_labels)
    feature_summary = {
        **summarize_localization_features(directions, local_metrics),
        **summarize_agop_features(agop_layer_metrics),
        "agop_selected_layer": selected_layer,
        "agop_layer_selection_metric": args.layer_selection,
    }

    eval_rows: List[Dict[str, Any]] = []
    for kind, items in [("target", record.get("eval", [])), ("neighbor", record.get("neighbors", []))]:
        for idx, item in enumerate(items):
            prompt = maybe_format_prompt(tokenizer, str(item["prompt"]), args.format_prompts)
            target = str(item["target"])
            base = score_with_optional_control(model, tokenizer, prompt, target, directions, selected_layer, None)
            row: Dict[str, Any] = {
                "kind": kind,
                "index": idx,
                "prompt": item["prompt"],
                "target": target,
                "base": base,
                "controlled": [],
            }
            for coef in args.suppression_coefs:
                controlled = score_with_optional_control(model, tokenizer, prompt, target, directions, selected_layer, coef)
                controlled["coef"] = coef
                controlled["delta_total_logprob"] = controlled["total_logprob"] - base["total_logprob"]
                controlled["delta_mean_logprob"] = controlled["mean_logprob"] - base["mean_logprob"]
                row["controlled"].append(controlled)
            eval_rows.append(row)

    return {
        "id": record["id"],
        "concept": record.get("concept", record["id"]),
        "control_method": f"agop_layer_{args.direction_method}",
        "layers": [selected_layer],
        "candidate_layers": hidden_layers,
        "n_components": 1,
        "projection_metrics": local_metrics,
        "agop_layer_metrics": agop_layer_metrics,
        "feature_summary": feature_summary,
        "eval": eval_rows,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "suppression_results.jsonl"
    failures_path = out_dir / "failures.jsonl"
    config_path = out_dir / "config.json"

    if args.overwrite:
        for path in [results_path, failures_path]:
            if path.exists():
                path.unlink()
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    model, tokenizer = load_model_and_tokenizer(args.model_local_path, args.device_map, args.torch_dtype)
    records = list(load_jsonl(args.input_jsonl))
    if args.max_records is not None:
        records = records[: args.max_records]

    for record in tqdm(records, desc="agop-guided layer records"):
        try:
            append_jsonl(results_path, run_record(record, model, tokenizer, args))
        except Exception as exc:
            append_jsonl(
                failures_path,
                {
                    "id": record.get("id"),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Wrote results to {results_path}")
    if failures_path.exists():
        print(f"Wrote failures to {failures_path}")


if __name__ == "__main__":
    main()
