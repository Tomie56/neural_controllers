from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import append_jsonl, load_jsonl, load_model_and_tokenizer, set_seed
from predictive_memory_localization.methods.run_bidirectional_three_table import (
    DirectionController,
    as_train_data,
    compute_directions,
    eval_items_for_record,
    first_onset,
    load_tables,
    projection_metrics,
    summarize_localization_features,
)


DEFAULT_DATA_DIR = "/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen"
DEFAULT_MODEL_PATH = "/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base"
DEFAULT_ALPHAS = [
    -1.0,
    -0.75,
    -0.5,
    -0.35,
    -0.25,
    -0.2,
    -0.15,
    -0.1,
    -0.08,
    -0.05,
    -0.03,
    -0.02,
    -0.01,
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.1,
    0.15,
    0.2,
    0.25,
    0.35,
    0.5,
    0.75,
    1.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate bidirectional intervention paths from alpha=0 directional derivatives. "
            "This avoids scoring every alpha and predicts delta_margin(alpha) ~= alpha * dmargin/dalpha."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-local-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default="qwen3_1_7b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--control-method",
        default="mean_difference",
        choices=["linear", "logistic", "mean_difference", "random"],
    )
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--record-offset", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="*", default=[-21])
    parser.add_argument("--alphas", type=float, nargs="*", default=DEFAULT_ALPHAS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--target-threshold", type=float, default=0.1522890289624531)
    parser.add_argument("--neighbor-threshold", type=float, default=0.1522890289624531)
    parser.add_argument("--capability-threshold", type=float, default=0.1522890289624531)
    parser.add_argument("--target-probe-success-rate", type=float, default=0.5)
    parser.add_argument("--neighbor-damage-rate-threshold", type=float, default=0.5)
    parser.add_argument("--capability-damage-rate-threshold", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def freeze_model_parameters(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def hook_model_with_alpha(
    model: Any,
    directions: Dict[int, torch.Tensor],
    layers_to_control: Sequence[int],
    alpha: torch.Tensor,
    component_idx: int = 0,
) -> Dict[int, Any]:
    hooks: Dict[int, Any] = {}
    for layer_idx in layers_to_control:
        control_vec = directions[layer_idx][component_idx]
        if len(control_vec.shape) == 1:
            control_vec = control_vec.reshape(1, 1, -1)
        block = model.model.layers[layer_idx]

        def block_hook(module, input, output, control_vec=control_vec, alpha=alpha):
            is_tuple = isinstance(output, tuple)
            new_output = output[0] if is_tuple else output
            control = control_vec.to(dtype=new_output.dtype, device=new_output.device)
            new_output = new_output + alpha.to(dtype=new_output.dtype, device=new_output.device) * control
            if is_tuple:
                new_output = (new_output,) + output[1:]
            return new_output

        hooks[layer_idx] = block.register_forward_hook(block_hook)
    return hooks


def clear_hooks(hooks: Dict[int, Any]) -> None:
    for handle in hooks.values():
        handle.remove()


def continuation_logprob_tensor(model: Any, tokenizer: Any, prompt: str, continuation: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    full_text = prompt + continuation
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Continuation produced no additional tokens. Check whitespace/tokenization.")

    input_ids = full_ids[:-1].unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    target_ids = full_ids[1:].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(outputs.logits[0].float(), dim=-1)
    start = max(len(prompt_ids) - 1, 0)
    end = len(full_ids) - 1
    continuation_log_probs = log_probs[start:end]
    continuation_targets = target_ids[start:end]
    token_log_probs = continuation_log_probs.gather(dim=-1, index=continuation_targets.unsqueeze(-1)).squeeze(-1)
    total = token_log_probs.sum()
    mean = token_log_probs.mean()
    return total, mean, int(token_log_probs.numel())


def score_continuation_and_slope(
    controller: DirectionController,
    prompt: str,
    continuation: str,
    layers: Sequence[int],
) -> Dict[str, Any]:
    model = controller.model
    tokenizer = controller.tokenizer
    model.zero_grad(set_to_none=True)
    alpha = torch.zeros((), device=model.device, dtype=torch.float32, requires_grad=True)
    hooks = hook_model_with_alpha(model, controller.directions, layers, alpha)
    try:
        total, mean_logprob, num_tokens = continuation_logprob_tensor(model, tokenizer, prompt, continuation)
        total.backward()
        slope = float(alpha.grad.detach().cpu()) if alpha.grad is not None else 0.0
    finally:
        clear_hooks(hooks)
        model.zero_grad(set_to_none=True)
    return {
        "total_logprob": float(total.detach().cpu()),
        "mean_logprob": float(mean_logprob.detach().cpu()),
        "num_tokens": num_tokens,
        "directional_slope": slope,
    }


def score_margin_and_slope(
    controller: DirectionController,
    prompt: str,
    correct: str,
    contrast: str,
    layers: Sequence[int],
) -> Dict[str, Any]:
    correct_score = score_continuation_and_slope(controller, prompt, correct, layers)
    contrast_score = score_continuation_and_slope(controller, prompt, contrast, layers)
    margin = correct_score["total_logprob"] - contrast_score["total_logprob"]
    mean_margin = correct_score["mean_logprob"] - contrast_score["mean_logprob"]
    slope = correct_score["directional_slope"] - contrast_score["directional_slope"]
    return {
        "base_margin": float(margin),
        "base_mean_margin": float(mean_margin),
        "directional_slope": slope,
        "correct_total_logprob": correct_score["total_logprob"],
        "contrast_total_logprob": contrast_score["total_logprob"],
        "correct_mean_logprob": correct_score["mean_logprob"],
        "contrast_mean_logprob": contrast_score["mean_logprob"],
        "correct_num_tokens": correct_score["num_tokens"],
        "contrast_num_tokens": contrast_score["num_tokens"],
        "correct_directional_slope": correct_score["directional_slope"],
        "contrast_directional_slope": contrast_score["directional_slope"],
    }


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
    )
    controller.directions = directions
    controller.hidden_layers = hidden_layers
    projection = projection_metrics(directions, hidden_states, train_labels)
    feature_summary = summarize_localization_features(directions, projection)

    eval_rows = []
    for item in eval_items_for_record(record_id, assignment, eval_bank):
        scored = score_margin_and_slope(controller, item["prompt"], item["correct"], item["contrast"], hidden_layers)
        alpha_rows = []
        for alpha in args.alphas:
            predicted_delta = float(alpha) * float(scored["directional_slope"])
            alpha_rows.append(
                {
                    "alpha": float(alpha),
                    "predicted_margin": scored["base_margin"] + predicted_delta,
                    "predicted_delta_margin": predicted_delta,
                    "predicted_mean_margin": None,
                    "predicted_delta_mean_margin": None,
                }
            )
        eval_rows.append(
            {
                "eval_id": item["eval_id"],
                "eval_type": item["eval_type"],
                "assignment_index": item["assignment_index"],
                "prompt": item["prompt"],
                "correct": item["correct"],
                "contrast": item["contrast"],
                **scored,
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


def flatten_per_eval(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    features = result.get("feature_summary", {})
    source = result.get("source", {})
    metadata = result.get("metadata", {})
    for item in result.get("eval", []):
        for alpha_row in item["alphas"]:
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
                    "layer": ",".join(str(layer) for layer in result.get("layers", [])),
                    "eval_id": item["eval_id"],
                    "eval_type": item["eval_type"],
                    "assignment_index": item["assignment_index"],
                    "alpha": alpha_row["alpha"],
                    "base_margin": item["base_margin"],
                    "directional_slope": item["directional_slope"],
                    "predicted_margin": alpha_row["predicted_margin"],
                    "predicted_delta_margin": alpha_row["predicted_delta_margin"],
                    **{f"feature_{k}": v for k, v in features.items() if not isinstance(v, list)},
                }
            )
    return rows


def mean(values: Sequence[Any]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_per_record(per_eval: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for row in per_eval:
        groups.setdefault((row["record_id"], float(row["alpha"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (record_id, alpha), rows in sorted(groups.items()):
        first = rows[0]
        target = [r for r in rows if r["eval_type"] == "target"]
        neighbor = [r for r in rows if r["eval_type"] == "neighbor"]
        capability = [r for r in rows if r["eval_type"] == "capability"]
        if alpha < 0:
            target_success_rate = mean([float(r["predicted_delta_margin"]) <= -args.target_threshold for r in target])
        elif alpha > 0:
            target_success_rate = mean([float(r["predicted_delta_margin"]) >= args.target_threshold for r in target])
        else:
            target_success_rate = 0.0
        target_success = bool(target_success_rate >= args.target_probe_success_rate) if alpha != 0 else False
        neighbor_damage_rate = mean([float(r["predicted_delta_margin"]) <= -args.neighbor_threshold for r in neighbor])
        capability_damage_rate = mean([float(r["predicted_delta_margin"]) <= -args.capability_threshold for r in capability])
        out.append(
            {
                "record_id": record_id,
                "control_method": first["control_method"],
                "layer": first.get("layer"),
                "dataset": first.get("dataset"),
                "domain": first.get("domain"),
                "freshness_group": first.get("freshness_group"),
                "release_year": first.get("release_year"),
                "alpha": alpha,
                "target_predicted_delta_margin": mean([r["predicted_delta_margin"] for r in target]),
                "neighbor_predicted_delta_margin": mean([r["predicted_delta_margin"] for r in neighbor]),
                "capability_predicted_delta_margin": mean([r["predicted_delta_margin"] for r in capability]),
                "target_success_rate": target_success_rate,
                "target_success": target_success,
                "suppression_success": target_success if alpha < 0 else False,
                "enhancement_success": target_success if alpha > 0 else False,
                "neighbor_damage_rate": neighbor_damage_rate,
                "neighbor_damaged": bool(neighbor_damage_rate >= args.neighbor_damage_rate_threshold),
                "capability_damage_rate": capability_damage_rate,
                "capability_damaged": bool(capability_damage_rate >= args.capability_damage_rate_threshold),
                "target_slope_mean": mean([r["directional_slope"] for r in target]),
                "neighbor_slope_mean": mean([r["directional_slope"] for r in neighbor]),
                "capability_slope_mean": mean([r["directional_slope"] for r in capability]),
                "n_target_eval": len(target),
                "n_neighbor_eval": len(neighbor),
                "n_capability_eval": len(capability),
                **{k: v for k, v in first.items() if k.startswith("feature_")},
            }
        )
    return out


def path_summary(per_record: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_record:
        groups.setdefault(str(row["record_id"]), []).append(row)
    out: List[Dict[str, Any]] = []
    for record_id, rows in sorted(groups.items()):
        first = rows[0]
        supp_onset = first_onset(rows, "suppression_success", "negative")
        enh_onset = first_onset(rows, "enhancement_success", "positive")
        neg_damage = first_onset(
            [{**r, "any_damage": bool(r["neighbor_damaged"]) or bool(r["capability_damaged"])} for r in rows],
            "any_damage",
            "negative",
        )
        pos_damage = first_onset(
            [{**r, "any_damage": bool(r["neighbor_damaged"]) or bool(r["capability_damaged"])} for r in rows],
            "any_damage",
            "positive",
        )
        clean_supp = [
            r
            for r in rows
            if float(r["alpha"]) < 0 and r["suppression_success"] and not r["neighbor_damaged"] and not r["capability_damaged"]
        ]
        clean_enh = [
            r
            for r in rows
            if float(r["alpha"]) > 0 and r["enhancement_success"] and not r["neighbor_damaged"] and not r["capability_damaged"]
        ]
        out.append(
            {
                "record_id": record_id,
                "control_method": first["control_method"],
                "layer": first.get("layer"),
                "dataset": first.get("dataset"),
                "domain": first.get("domain"),
                "freshness_group": first.get("freshness_group"),
                "suppression_onset_alpha": supp_onset,
                "enhancement_onset_alpha": enh_onset,
                "negative_damage_onset_alpha": neg_damage,
                "positive_damage_onset_alpha": pos_damage,
                "clean_suppression_window_exists": bool(clean_supp),
                "clean_enhancement_window_exists": bool(clean_enh),
                "bidirectional_clean_control": bool(clean_supp and clean_enh),
                "suppression_damage_minus_target_onset": (abs(neg_damage) - abs(supp_onset))
                if neg_damage is not None and supp_onset is not None
                else None,
                "enhancement_damage_minus_target_onset": (abs(pos_damage) - abs(enh_onset))
                if pos_damage is not None and enh_onset is not None
                else None,
            }
        )
    return out


def summarize_method(per_record: List[Dict[str, Any]], paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_records = len({row["record_id"] for row in per_record})
    return {
        "n_records": n_records,
        "n_per_record_rows": len(per_record),
        "alphas": sorted({float(row["alpha"]) for row in per_record}),
        "suppression_any_rate": mean([p["suppression_onset_alpha"] is not None for p in paths]),
        "enhancement_any_rate": mean([p["enhancement_onset_alpha"] is not None for p in paths]),
        "clean_suppression_rate": mean([p["clean_suppression_window_exists"] for p in paths]),
        "clean_enhancement_rate": mean([p["clean_enhancement_window_exists"] for p in paths]),
        "bidirectional_clean_rate": mean([p["bidirectional_clean_control"] for p in paths]),
        "negative_damage_any_rate": mean([p["negative_damage_onset_alpha"] is not None for p in paths]),
        "positive_damage_any_rate": mean([p["positive_damage_onset_alpha"] is not None for p in paths]),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["record_id"]) for row in load_jsonl(path) if row.get("record_id")}


def write_report(path: Path, args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    lines = [
        "# Gradient Response Three-Table Run",
        "",
        "## Principle",
        "",
        "For a forced-choice margin `m(alpha)`, this run estimates the local response at alpha=0:",
        "",
        "```text",
        "m(alpha) = m(0) + alpha * dm/dalpha|0 + O(alpha^2)",
        "dm/dalpha|0 = <d m / d h_l, v_l>",
        "```",
        "",
        "It scores alpha=0 once and uses autograd through the same activation hook used by dense steering.",
        "",
        f"Model: `{args.model_local_path}`",
        f"Data: `{args.data_dir}`",
        f"Method: `{args.control_method}`",
        f"Layers: `{args.layers}`",
        f"Records: `{summary['n_records']}`",
        f"Predicted alphas: `{summary['alphas']}`",
        "",
        "## Predicted Path Summary",
        "",
        f"- suppression any rate: `{summary['suppression_any_rate']}`",
        f"- enhancement any rate: `{summary['enhancement_any_rate']}`",
        f"- clean suppression rate: `{summary['clean_suppression_rate']}`",
        f"- clean enhancement rate: `{summary['clean_enhancement_rate']}`",
        f"- bidirectional clean rate: `{summary['bidirectional_clean_rate']}`",
        f"- negative damage any rate: `{summary['negative_damage_any_rate']}`",
        f"- positive damage any rate: `{summary['positive_damage_any_rate']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_gradient_response.jsonl"
    failures_path = out_dir / "failures.jsonl"
    if args.overwrite:
        for name in [
            "raw_gradient_response.jsonl",
            "failures.jsonl",
            "per_eval_gradient.csv",
            "per_record_predicted.csv",
            "path_rows_predicted.csv",
            "method_summary_predicted.json",
            "GRADIENT_RESPONSE_REPORT.md",
            "run_config.json",
        ]:
            (out_dir / name).unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {raw_path}. Use --overwrite or --resume.")

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    records, eval_bank, assignments = load_tables(Path(args.data_dir))
    records = records[args.record_offset :]
    if args.max_records is not None:
        records = records[: args.max_records]
    done = existing_record_ids(raw_path) if args.resume else set()
    if done:
        records = [record for record in records if str(record["record_id"]) not in done]
        print(f"Resume mode: skipping {len(done)} existing record_ids; remaining={len(records)}")

    model, tokenizer = load_model_and_tokenizer(args.model_local_path, device_map=args.device_map, torch_dtype=args.torch_dtype)
    freeze_model_parameters(model)
    controller = DirectionController(model=model, tokenizer=tokenizer, directions={}, hidden_layers=list(args.layers), batch_size=args.batch_size)

    t0 = time.time()
    for record in tqdm(records, desc=f"{args.control_method} gradient records"):
        try:
            result = run_record(record, assignments[record["record_id"]], eval_bank, controller, args)
            append_jsonl(raw_path, result)
        except BaseException as exc:  # noqa: BLE001
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

    per_eval: List[Dict[str, Any]] = []
    if raw_path.exists():
        for result in load_jsonl(raw_path):
            per_eval.extend(flatten_per_eval(result))
    per_record = summarize_per_record(per_eval, args)
    paths = path_summary(per_record)
    summary = summarize_method(per_record, paths)
    summary.update(
        {
            "elapsed_seconds": time.time() - t0,
            "control_method": args.control_method,
            "model_name": args.model_name,
            "layers": list(args.layers),
            "path_id": f"{args.control_method}__layers_{'_'.join(str(layer) for layer in args.layers)}",
            "n_failures": sum(1 for _ in load_jsonl(failures_path)) if failures_path.exists() else 0,
        }
    )
    write_csv(out_dir / "per_eval_gradient.csv", per_eval)
    write_csv(out_dir / "per_record_predicted.csv", per_record)
    write_csv(out_dir / "path_rows_predicted.csv", paths)
    (out_dir / "method_summary_predicted.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "GRADIENT_RESPONSE_REPORT.md", args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
