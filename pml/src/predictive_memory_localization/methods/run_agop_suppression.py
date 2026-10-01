from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from dataclasses import dataclass
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
    get_hidden_states,
    normalize_direction,
)

AGOP_DEFAULT_LAYERS = [-1, -5, -9, -13, -17, -21]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AGOP/RFM activation suppression.")
    parser.add_argument("--model-local-path", default="/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-method", choices=["agop_top1", "agop_topk_project"], default="agop_top1")
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
        help="Negative values suppress the AGOP memory direction/subspace.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass
class AgopControls:
    directions: Dict[int, torch.Tensor]
    bases: Dict[int, torch.Tensor]
    layer_features: Dict[int, Dict[str, Any]]


def oriented_top_vector(eigvecs: torch.Tensor, X: torch.Tensor, labels: Sequence[int]) -> torch.Tensor:
    vec = normalize_direction(eigvecs[:, 0])
    labels_t = torch.tensor(labels).bool()
    scores = X.float() @ vec.float()
    if scores[labels_t].mean() < scores[~labels_t].mean():
        vec = -vec
    return vec


def compute_agop_controls(
    model,
    tokenizer,
    train_prompts: Sequence[str],
    train_labels: Sequence[int],
    hidden_layers: Sequence[int],
    batch_size: int,
    args: argparse.Namespace,
) -> AgopControls:
    hidden_states = get_hidden_states(train_prompts, model, tokenizer, hidden_layers, batch_size)
    directions: Dict[int, torch.Tensor] = {}
    bases: Dict[int, torch.Tensor] = {}
    layer_features: Dict[int, Dict[str, Any]] = {}

    for layer in hidden_layers:
        X = hidden_states[layer]
        agop = fit_rfm_agop(X, train_labels, args)
        eigvals, eigvecs = torch.linalg.eigh(agop.float())
        order = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, order]
        top_vec = oriented_top_vector(eigvecs, X, train_labels)
        k = min(args.top_k, eigvecs.shape[1])
        basis = eigvecs[:, :k].float()
        basis[:, 0] = top_vec.float()
        directions[int(layer)] = top_vec.reshape(1, -1)
        bases[int(layer)] = basis
        layer_features[int(layer)] = agop_layer_features(agop, X, train_labels, args.top_k)

    return AgopControls(directions=directions, bases=bases, layer_features=layer_features)


def hook_agop_topk_project(model, bases: Dict[int, torch.Tensor], layers_to_control: Sequence[int], coef: float) -> Dict[int, Any]:
    hooks = {}
    strength = -float(coef)
    for layer_idx in layers_to_control:
        basis = bases[layer_idx].float()
        block = model.model.layers[layer_idx]

        def block_hook(module, inputs, output, basis=basis, strength=strength):
            hidden = output[0] if isinstance(output, tuple) else output
            new_output = hidden
            basis_d = basis.to(dtype=new_output.dtype, device=new_output.device)
            projection = torch.matmul(torch.matmul(new_output, basis_d), basis_d.T)
            new_output = new_output - strength * projection
            if isinstance(output, tuple):
                return (new_output,) + output[1:]
            return new_output

        hooks[layer_idx] = block.register_forward_hook(block_hook)
    return hooks


def score_with_control(
    model,
    tokenizer,
    prompt: str,
    target: str,
    controls: AgopControls,
    layers: Sequence[int],
    coef: Optional[float],
    control_method: str,
) -> Dict[str, Any]:
    hooks = {}
    try:
        if coef is not None:
            if control_method == "agop_top1":
                hooks = generation_utils.hook_model(model, controls.directions, layers, coef)
            elif control_method == "agop_topk_project":
                hooks = hook_agop_topk_project(model, controls.bases, layers, coef)
            else:
                raise ValueError(f"Unsupported control method: {control_method}")
        return continuation_logprob(model, tokenizer, prompt, target)
    finally:
        if hooks:
            generation_utils.clear_hooks(hooks)


def run_record(record: Dict[str, Any], model, tokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    validate_suppression_record(record)
    train_prompts, train_labels = as_train_data(record, tokenizer, args.format_prompts)
    hidden_layers = list(args.layers)
    controls = compute_agop_controls(
        model,
        tokenizer,
        train_prompts,
        train_labels,
        hidden_layers,
        args.batch_size,
        args,
    )
    feature_summary = summarize_agop_features(controls.layer_features)

    eval_rows: List[Dict[str, Any]] = []
    for kind, items in [("target", record.get("eval", [])), ("neighbor", record.get("neighbors", []))]:
        for idx, item in enumerate(items):
            prompt = maybe_format_prompt(tokenizer, str(item["prompt"]), args.format_prompts)
            target = str(item["target"])
            base = score_with_control(model, tokenizer, prompt, target, controls, hidden_layers, None, args.control_method)
            row: Dict[str, Any] = {
                "kind": kind,
                "index": idx,
                "prompt": item["prompt"],
                "target": target,
                "base": base,
                "controlled": [],
            }
            for coef in args.suppression_coefs:
                controlled = score_with_control(model, tokenizer, prompt, target, controls, hidden_layers, coef, args.control_method)
                controlled["coef"] = coef
                controlled["delta_total_logprob"] = controlled["total_logprob"] - base["total_logprob"]
                controlled["delta_mean_logprob"] = controlled["mean_logprob"] - base["mean_logprob"]
                row["controlled"].append(controlled)
            eval_rows.append(row)

    return {
        "id": record["id"],
        "concept": record.get("concept", record["id"]),
        "control_method": args.control_method,
        "layers": hidden_layers,
        "n_components": args.top_k if args.control_method == "agop_topk_project" else 1,
        "projection_metrics": controls.layer_features,
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

    for record in tqdm(records, desc="agop suppression records"):
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
