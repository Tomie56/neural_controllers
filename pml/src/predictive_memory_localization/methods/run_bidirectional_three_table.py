from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization import generation_utils
from predictive_memory_localization.common import (
    append_jsonl,
    continuation_logprob,
    load_jsonl,
    load_model_and_tokenizer,
    resolve_model_layers,
    set_seed,
    write_jsonl,
)


DEFAULT_DATA_DIR = "/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen"
DEFAULT_MODEL_PATH = "/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base"
DEFAULT_ALPHAS = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
DEFAULT_LAYERS = [-1, -9, -17]


@dataclass
class DirectionController:
    model: Any
    tokenizer: Any
    directions: Dict[int, torch.Tensor]
    hidden_layers: List[int]
    batch_size: int
    control_scale: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bidirectional activation steering on normalized PML records/eval_bank/assignments."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-local-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default="qwen3_1_7b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--control-method",
        default="mean_difference",
        choices=[
            "linear",
            "logistic",
            "mean_difference",
            "pca_diff",
            "random",
            "matched_norm_random",
            "rfm_agop_top1",
        ],
    )
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--record-offset", type=int, default=0)
    parser.add_argument(
        "--record-ids-file",
        default=None,
        help="Optional text file with one record_id per line. Filtering happens before offset/max-records.",
    )
    parser.add_argument("--layers", type=int, nargs="*", default=DEFAULT_LAYERS)
    parser.add_argument("--alphas", type=float, nargs="*", default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--alpha-plan-jsonl",
        default=None,
        help=(
            "Optional JSONL with per-record alpha grids. Rows need record_id and alphas. "
            "If a record_id is present, its alphas override --alphas."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument(
        "--model-loader",
        default="auto",
        choices=["auto", "causal_lm", "image_text_to_text"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--rfm-iters", type=int, default=3)
    parser.add_argument("--rfm-reg", type=float, default=1e-3)
    parser.add_argument("--rfm-bandwidth", type=float, default=10.0)
    parser.add_argument("--rfm-kernel", type=str, default="laplace")
    parser.add_argument("--rfm-mem-gb", type=float, default=8.0)
    parser.add_argument(
        "--control-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply every reference alpha by this fixed layer-level scale before "
            "adding the unit-norm direction. Use 1.0 for the historical raw-alpha protocol."
        ),
    )
    parser.add_argument("--target-threshold", type=float, default=1.0)
    parser.add_argument("--neighbor-threshold", type=float, default=1.0)
    parser.add_argument("--capability-threshold", type=float, default=1.0)
    parser.add_argument("--target-probe-success-rate", type=float, default=0.5)
    parser.add_argument("--neighbor-damage-rate-threshold", type=float, default=0.5)
    parser.add_argument("--capability-damage-rate-threshold", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip record_ids already present in raw_scores.jsonl and rebuild summaries at the end.",
    )
    return parser.parse_args()


@torch.no_grad()
def get_hidden_states(
    prompts: Sequence[str],
    model,
    tokenizer,
    hidden_layers: Sequence[int],
    batch_size: int,
    rep_token: int = -1,
) -> Dict[int, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size)

    states_by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in hidden_layers}
    for input_ids, attention_mask in loader:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        for layer in hidden_layers:
            states_by_layer[layer].append(outputs.hidden_states[layer][:, rep_token, :].detach().cpu())

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in states_by_layer.items()}


def normalize_direction(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.reshape(-1).float()
    return vec / (vec.norm() + 1e-8)


def fit_linear_direction(X: torch.Tensor, y: torch.Tensor, reg: float = 1e-3) -> torch.Tensor:
    X = X.float()
    y = y.float().reshape(-1, 1)
    X_centered = X - X.mean(dim=0, keepdim=True)
    y_centered = y - y.mean()
    d = X_centered.shape[1]
    beta = torch.linalg.pinv(
        X_centered.T @ X_centered + reg * torch.eye(d, device=X_centered.device)
    ) @ X_centered.T @ y_centered
    return normalize_direction(beta.squeeze(1))


def fit_logistic_direction(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(fit_intercept=True, max_iter=1000)
    clf.fit(X.float().cpu().numpy(), y.long().cpu().numpy())
    return normalize_direction(torch.from_numpy(clf.coef_[0]))


def fit_mean_difference_direction(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_bool = y.bool()
    vec = X[y_bool].float().mean(dim=0) - X[~y_bool].float().mean(dim=0)
    return normalize_direction(vec)


def fit_pca_diff_direction(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_bool = y.bool()
    pos = X[y_bool].float()
    neg = X[~y_bool].float()
    n = min(pos.shape[0], neg.shape[0])
    if n < 2:
        return fit_mean_difference_direction(X, y)
    diffs = pos[:n] - neg[:n]
    diffs = diffs - diffs.mean(dim=0, keepdim=True)
    if torch.linalg.matrix_norm(diffs).item() <= 1e-8:
        return fit_mean_difference_direction(X, y)
    try:
        _, _, vh = torch.linalg.svd(diffs, full_matrices=False)
        return normalize_direction(vh[0])
    except RuntimeError:
        return fit_mean_difference_direction(X, y)


def fit_random_direction(X: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return normalize_direction(torch.randn(X.shape[1], generator=generator))


def fit_rfm_agop_top1_direction(X: torch.Tensor, y: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, Dict[str, Any]]:
    from predictive_memory_localization.methods.extract_agop_features import (
        agop_layer_features,
        fit_rfm_agop,
    )

    agop = fit_rfm_agop(X, y.tolist(), args)
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (agop.float() + agop.float().T))
    order = torch.argsort(eigvals, descending=True)
    vec = normalize_direction(eigvecs[:, order[0]])
    features = agop_layer_features(agop, X, y.tolist(), top_k=min(5, X.shape[1]))
    return vec, features


def compute_directions(
    model,
    tokenizer,
    train_prompts: Sequence[str],
    train_labels: Sequence[int],
    hidden_layers: Sequence[int],
    batch_size: int,
    control_method: str,
    seed: int,
    args: Optional[argparse.Namespace] = None,
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    hidden_states = get_hidden_states(train_prompts, model, tokenizer, hidden_layers, batch_size)
    labels = torch.tensor(train_labels).long()
    directions: Dict[int, torch.Tensor] = {}
    agop_features: Dict[str, Any] = {}

    for layer in hidden_layers:
        X = hidden_states[layer]
        if control_method == "linear":
            vec = fit_linear_direction(X, labels)
        elif control_method == "logistic":
            vec = fit_logistic_direction(X, labels)
        elif control_method == "mean_difference":
            vec = fit_mean_difference_direction(X, labels)
        elif control_method == "pca_diff":
            vec = fit_pca_diff_direction(X, labels)
        elif control_method in {"random", "matched_norm_random"}:
            vec = fit_random_direction(X, seed + abs(layer))
        elif control_method == "rfm_agop_top1":
            if args is None:
                raise ValueError("rfm_agop_top1 requires argparse args with RFM settings")
            vec, features = fit_rfm_agop_top1_direction(X, labels, args)
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    layer_name = str(layer).replace("-", "neg")
                    agop_features[f"agop_layer_{layer_name}_{key}"] = float(value)
        else:
            raise ValueError(f"Unsupported control method: {control_method}")

        projections = X.float() @ vec.float()
        pos_mean = projections[labels.bool()].mean()
        neg_mean = projections[~labels.bool()].mean()
        if pos_mean < neg_mean:
            vec = -vec
        directions[layer] = vec.reshape(1, -1)

    if agop_features:
        setattr(compute_directions, "_last_agop_features", agop_features)
    else:
        setattr(compute_directions, "_last_agop_features", {})
    return directions, hidden_states


def projection_metrics(
    directions: Dict[int, torch.Tensor],
    hidden_states: Dict[int, torch.Tensor],
    train_labels: Sequence[int],
) -> Dict[str, Any]:
    labels = torch.tensor(train_labels).bool()
    metrics: Dict[str, Any] = {}
    for layer, direction in directions.items():
        vec = direction[0].float()
        states = hidden_states[layer].float()
        scores = states @ vec
        pos = scores[labels]
        neg = scores[~labels]
        pos_mean = pos.mean()
        neg_mean = neg.mean()
        pooled = torch.sqrt(0.5 * (pos.var(unbiased=False) + neg.var(unbiased=False)) + 1e-8)
        threshold = 0.5 * (pos_mean + neg_mean)
        preds = scores > threshold
        metrics[str(layer)] = {
            "pos_mean": float(pos_mean.detach().cpu()),
            "neg_mean": float(neg_mean.detach().cpu()),
            "mean_gap": float((pos_mean - neg_mean).detach().cpu()),
            "cohen_d": float(((pos_mean - neg_mean) / pooled).detach().cpu()),
            "threshold_accuracy": float((preds == labels).float().mean().detach().cpu()),
            "direction_norm": float(vec.norm().detach().cpu()),
        }
    return metrics


def summarize_localization_features(
    directions: Dict[int, torch.Tensor],
    local_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    layer_items = sorted(((int(layer), metrics) for layer, metrics in local_metrics.items()), key=lambda x: x[0])
    if not layer_items:
        return {}
    saliency = [abs(float(metrics.get("cohen_d", 0.0))) for _, metrics in layer_items]
    saliency_sum = sum(saliency)
    if saliency_sum > 0:
        weights = [x / saliency_sum for x in saliency]
        entropy = -sum(w * math.log(w + 1e-12) for w in weights)
        normalized_entropy = entropy / math.log(len(weights)) if len(weights) > 1 else 0.0
        concentration = max(weights)
    else:
        normalized_entropy = 0.0
        concentration = 0.0
    top_idx = max(range(len(layer_items)), key=lambda idx: saliency[idx])
    direction_vectors = [
        (layer, direction.reshape(-1).float())
        for layer, direction in sorted(directions.items(), key=lambda item: item[0])
    ]
    pairwise_cosines: List[float] = []
    for i, (_, left) in enumerate(direction_vectors):
        for _, right in direction_vectors[i + 1 :]:
            denom = (left.norm() * right.norm()).clamp_min(1e-8)
            pairwise_cosines.append(float(torch.dot(left, right).div(denom).detach().cpu()))
    return {
        "layers": [layer for layer, _ in layer_items],
        "top_saliency_layer": layer_items[top_idx][0],
        "max_abs_cohen_d": saliency[top_idx],
        "mean_abs_cohen_d": float(sum(saliency) / len(saliency)),
        "saliency_concentration": float(concentration),
        "saliency_entropy": float(normalized_entropy),
        "mean_threshold_accuracy": float(sum(float(m.get("threshold_accuracy", 0.0)) for _, m in layer_items) / len(layer_items)),
        "direction_pairwise_cosine_mean": float(sum(pairwise_cosines) / len(pairwise_cosines)) if pairwise_cosines else 0.0,
    }


def score_continuation_with_control(
    controller: DirectionController,
    prompt: str,
    continuation: str,
    layers: Sequence[int],
    alpha: Optional[float],
) -> Dict[str, Any]:
    hooks = {}
    try:
        if alpha is not None and abs(alpha) > 0:
            effective_coef = float(alpha) * float(controller.control_scale)
            hooks = generation_utils.hook_model(
                controller.model,
                controller.directions,
                layers,
                effective_coef,
            )
        return continuation_logprob(controller.model, controller.tokenizer, prompt, continuation)
    finally:
        if hooks:
            generation_utils.clear_hooks(hooks)


def load_tables(data_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    records = list(load_jsonl(data_dir / "records.jsonl"))
    eval_bank = {row["eval_id"]: row for row in load_jsonl(data_dir / "eval_bank.jsonl")}
    assignments = {row["record_id"]: row for row in load_jsonl(data_dir / "eval_assignments.jsonl")}
    return records, eval_bank, assignments


def eval_items_for_record(record_id: str, assignment: Dict[str, Any], eval_bank: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for eval_type, key in [
        ("target", "target_eval_ids"),
        ("neighbor", "neighbor_eval_ids"),
        ("capability", "capability_eval_ids"),
    ]:
        for idx, eval_id in enumerate(assignment[key]):
            item = dict(eval_bank[eval_id])
            item["record_id"] = record_id
            item["assignment_index"] = idx
            item["eval_type"] = eval_type
            items.append(item)
    return items


def as_train_data(record: Dict[str, Any]) -> tuple[List[str], List[int]]:
    positives = [str(x) for x in record["train_positive"]]
    negatives = [str(x) for x in record["train_negative"]]
    return positives + negatives, [1] * len(positives) + [0] * len(negatives)


def run_record(
    record: Dict[str, Any],
    assignment: Dict[str, Any],
    eval_bank: Dict[str, Dict[str, Any]],
    controller: DirectionController,
    args: argparse.Namespace,
    alphas: Optional[Sequence[float]] = None,
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
    feature_summary.update(getattr(compute_directions, "_last_agop_features", {}))
    active_alphas = list(args.alphas if alphas is None else alphas)

    eval_rows = []
    for item in eval_items_for_record(record_id, assignment, eval_bank):
        base_correct = score_continuation_with_control(controller, item["prompt"], item["correct"], hidden_layers, None)
        base_contrast = score_continuation_with_control(controller, item["prompt"], item["contrast"], hidden_layers, None)
        base_margin = base_correct["total_logprob"] - base_contrast["total_logprob"]
        base_mean_margin = base_correct["mean_logprob"] - base_contrast["mean_logprob"]
        alpha_rows = []
        for alpha in active_alphas:
            if abs(alpha) == 0:
                correct = base_correct
                contrast = base_contrast
            else:
                correct = score_continuation_with_control(controller, item["prompt"], item["correct"], hidden_layers, alpha)
                contrast = score_continuation_with_control(controller, item["prompt"], item["contrast"], hidden_layers, alpha)
            margin = correct["total_logprob"] - contrast["total_logprob"]
            mean_margin = correct["mean_logprob"] - contrast["mean_logprob"]
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "control_scale": float(controller.control_scale),
                    "effective_control_coef": float(alpha) * float(controller.control_scale),
                    "correct_total_logprob": correct["total_logprob"],
                    "contrast_total_logprob": contrast["total_logprob"],
                    "correct_mean_logprob": correct["mean_logprob"],
                    "contrast_mean_logprob": contrast["mean_logprob"],
                    "margin": margin,
                    "mean_margin": mean_margin,
                    "delta_margin": margin - base_margin,
                    "delta_mean_margin": mean_margin - base_mean_margin,
                    "correct_num_tokens": correct["num_tokens"],
                    "contrast_num_tokens": contrast["num_tokens"],
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
                "base_margin": base_margin,
                "base_mean_margin": base_mean_margin,
                "base_correct_total_logprob": base_correct["total_logprob"],
                "base_contrast_total_logprob": base_contrast["total_logprob"],
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
        "control_scale": float(controller.control_scale),
        "layers": hidden_layers,
        "alphas": active_alphas,
        "projection_metrics": projection,
        "feature_summary": feature_summary,
        "eval": eval_rows,
    }


def flatten_per_eval(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
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
                    "eval_id": item["eval_id"],
                    "eval_type": item["eval_type"],
                    "assignment_index": item["assignment_index"],
                    "alpha": alpha_row["alpha"],
                    "control_scale": alpha_row.get("control_scale", result.get("control_scale", 1.0)),
                    "effective_control_coef": alpha_row.get(
                        "effective_control_coef", alpha_row["alpha"]
                    ),
                    "base_margin": item["base_margin"],
                    "margin": alpha_row["margin"],
                    "delta_margin": alpha_row["delta_margin"],
                    "mean_margin": alpha_row["mean_margin"],
                    "delta_mean_margin": alpha_row["delta_mean_margin"],
                    **{f"feature_{k}": v for k, v in features.items() if not isinstance(v, list)},
                }
            )
    return rows


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_per_record(per_eval: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for row in per_eval:
        groups.setdefault((row["record_id"], float(row["alpha"])), []).append(row)
    out = []
    for (record_id, alpha), rows in sorted(groups.items()):
        first = rows[0]
        target = [r for r in rows if r["eval_type"] == "target"]
        neighbor = [r for r in rows if r["eval_type"] == "neighbor"]
        capability = [r for r in rows if r["eval_type"] == "capability"]
        target_success_rate = mean([
            float(r["delta_margin"]) <= -args.target_threshold if alpha < 0 else float(r["delta_margin"]) >= args.target_threshold
            for r in target
        ]) if alpha != 0 else 0.0
        target_success = target_success_rate >= args.target_probe_success_rate if alpha != 0 else False
        neighbor_damage_rate = mean([float(r["delta_margin"]) <= -args.neighbor_threshold for r in neighbor])
        capability_damage_rate = mean([float(r["delta_margin"]) <= -args.capability_threshold for r in capability])
        out.append(
            {
                "record_id": record_id,
                "control_method": first["control_method"],
                "dataset": first.get("dataset"),
                "domain": first.get("domain"),
                "freshness_group": first.get("freshness_group"),
                "release_year": first.get("release_year"),
                "alpha": alpha,
                "target_mean_delta_margin": mean([float(r["delta_margin"]) for r in target]),
                "neighbor_mean_delta_margin": mean([float(r["delta_margin"]) for r in neighbor]),
                "capability_mean_delta_margin": mean([float(r["delta_margin"]) for r in capability]),
                "target_success_rate": target_success_rate,
                "target_success": target_success,
                "suppression_success": target_success if alpha < 0 else False,
                "enhancement_success": target_success if alpha > 0 else False,
                "neighbor_damage_rate": neighbor_damage_rate,
                "neighbor_damaged": neighbor_damage_rate >= args.neighbor_damage_rate_threshold,
                "capability_damage_rate": capability_damage_rate,
                "capability_damaged": capability_damage_rate >= args.capability_damage_rate_threshold,
                "n_target_eval": len(target),
                "n_neighbor_eval": len(neighbor),
                "n_capability_eval": len(capability),
                **{k: v for k, v in first.items() if k.startswith("feature_")},
            }
        )
    return out


def first_onset(rows: List[Dict[str, Any]], predicate_key: str, sign: str) -> Optional[float]:
    candidates = []
    for row in rows:
        alpha = float(row["alpha"])
        if sign == "negative" and alpha >= 0:
            continue
        if sign == "positive" and alpha <= 0:
            continue
        if row.get(predicate_key):
            candidates.append(alpha)
    if not candidates:
        return None
    return min(candidates, key=lambda a: abs(a))


def path_summary(per_record: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_record:
        groups.setdefault(row["record_id"], []).append(row)
    out = []
    for record_id, rows in sorted(groups.items()):
        first = rows[0]
        supp_onset = first_onset(rows, "suppression_success", "negative")
        enh_onset = first_onset(rows, "enhancement_success", "positive")
        neg_damage = first_onset(
            [
                {**r, "any_damage": bool(r["neighbor_damaged"]) or bool(r["capability_damaged"])}
                for r in rows
            ],
            "any_damage",
            "negative",
        )
        pos_damage = first_onset(
            [
                {**r, "any_damage": bool(r["neighbor_damaged"]) or bool(r["capability_damaged"])}
                for r in rows
            ],
            "any_damage",
            "positive",
        )
        clean_supp = [
            r for r in rows
            if float(r["alpha"]) < 0 and r["suppression_success"] and not r["neighbor_damaged"] and not r["capability_damaged"]
        ]
        clean_enh = [
            r for r in rows
            if float(r["alpha"]) > 0 and r["enhancement_success"] and not r["neighbor_damaged"] and not r["capability_damaged"]
        ]
        out.append(
            {
                "record_id": record_id,
                "control_method": first["control_method"],
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
                "suppression_damage_minus_target_onset": (abs(neg_damage) - abs(supp_onset)) if neg_damage is not None and supp_onset is not None else None,
                "enhancement_damage_minus_target_onset": (abs(pos_damage) - abs(enh_onset)) if pos_damage is not None and enh_onset is not None else None,
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


def write_report(path: Path, args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    lines = [
        "# Bidirectional Three-Table Run",
        "",
        f"Model: `{args.model_local_path}`",
        f"Data: `{args.data_dir}`",
        f"Method: `{args.control_method}`",
        f"Layers: `{args.layers}`",
        f"Records: `{summary['n_records']}`",
        f"Alphas: `{summary['alphas']}`",
        f"Control scale: `{args.control_scale}`",
        "",
        "## Summary",
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


def existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for row in load_jsonl(path):
        if row.get("record_id"):
            out.add(str(row["record_id"]))
    return out


def retain_unresolved_failures(path: Path, successful_ids: set[str]) -> None:
    if not path.exists():
        return
    latest_by_record: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        record_id = str(row.get("record_id", ""))
        if record_id and record_id not in successful_ids:
            latest_by_record[record_id] = row
    write_jsonl(path, latest_by_record.values())


def load_record_ids_file(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    values = {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError(f"Record id file is empty: {path}")
    return values


def validate_resume_config(out_dir: Path, args: argparse.Namespace) -> None:
    config_path = out_dir / "run_config.json"
    if not args.resume or not config_path.exists():
        return
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    current = vars(args)
    fixed_keys = [
        "data_dir",
        "model_local_path",
        "model_name",
        "control_method",
        "control_scale",
        "record_offset",
        "record_ids_file",
        "layers",
        "alphas",
        "alpha_plan_jsonl",
        "seed",
        "torch_dtype",
        "device_map",
        "model_loader",
        "trust_remote_code",
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
            "Resume configuration does not match the existing run. "
            f"Use a new output directory or --overwrite. Mismatches: {mismatches}"
        )


def load_alpha_plan(path: Optional[str]) -> Dict[str, List[float]]:
    if not path:
        return {}
    plan: Dict[str, List[float]] = {}
    for row in load_jsonl(path):
        record_id = str(row.get("record_id", "")).strip()
        alphas = row.get("alphas")
        if not record_id or not isinstance(alphas, list):
            continue
        parsed = sorted({float(alpha) for alpha in alphas})
        if 0.0 not in parsed:
            parsed.append(0.0)
            parsed = sorted(parsed)
        plan[record_id] = parsed
    return plan


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "raw_scores.jsonl"
    failures_path = out_dir / "failures.jsonl"
    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")
    validate_resume_config(out_dir, args)
    if args.overwrite:
        for name in [
            "raw_scores.jsonl",
            "failures.jsonl",
            "per_eval.csv",
            "per_record.csv",
            "path_rows.csv",
            "method_summary.json",
            "STAGE_RUN_REPORT.md",
        ]:
            (out_dir / name).unlink(missing_ok=True)
    elif results_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {results_path}. Use --overwrite or --resume.")

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "model_local_path": args.model_local_path,
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "data_manifest.json").write_text(json.dumps({"data_dir": args.data_dir}, ensure_ascii=False, indent=2), encoding="utf-8")

    records, eval_bank, assignments = load_tables(Path(args.data_dir))
    alpha_plan = load_alpha_plan(args.alpha_plan_jsonl)
    selected_record_ids = load_record_ids_file(args.record_ids_file)
    if selected_record_ids is not None:
        records = [record for record in records if str(record["record_id"]) in selected_record_ids]
        missing_ids = selected_record_ids - {str(record["record_id"]) for record in records}
        if missing_ids:
            raise ValueError(f"Unknown record_ids in {args.record_ids_file}: {sorted(missing_ids)[:20]}")
    records = records[args.record_offset :]
    if args.max_records is not None:
        records = records[: args.max_records]
    done_ids = existing_record_ids(results_path) if args.resume else set()
    if done_ids:
        records = [record for record in records if str(record["record_id"]) not in done_ids]
        print(f"Resume mode: skipping {len(done_ids)} existing record_ids; remaining={len(records)}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_local_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        model_loader=args.model_loader,
        trust_remote_code=args.trust_remote_code,
    )
    model_layers, layer_path = resolve_model_layers(model)
    invalid_layers = [
        layer for layer in args.layers if layer < -len(model_layers) or layer >= len(model_layers)
    ]
    if invalid_layers:
        raise ValueError(
            f"Layers {invalid_layers} are invalid for {len(model_layers)} text layers at {layer_path}"
        )
    print(
        f"Resolved model_loader={getattr(model, '_pml_model_loader', args.model_loader)} "
        f"text_layers={len(model_layers)} layer_path={layer_path}"
    )
    if not math.isfinite(float(args.control_scale)) or float(args.control_scale) <= 0:
        raise ValueError(f"control_scale must be finite and positive, got {args.control_scale}")
    controller = DirectionController(
        model=model,
        tokenizer=tokenizer,
        directions={},
        hidden_layers=list(args.layers),
        batch_size=args.batch_size,
        control_scale=float(args.control_scale),
    )

    t0 = time.time()
    for record in tqdm(records, desc=f"{args.control_method} records"):
        try:
            record_id = str(record["record_id"])
            result = run_record(record, assignments[record_id], eval_bank, controller, args, alpha_plan.get(record_id))
            append_jsonl(results_path, result)
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

    per_eval: List[Dict[str, Any]] = []
    if results_path.exists():
        for result in load_jsonl(results_path):
            per_eval.extend(flatten_per_eval(result))
    successful_ids = existing_record_ids(results_path)
    retain_unresolved_failures(failures_path, successful_ids)
    per_record = summarize_per_record(per_eval, args)
    paths = path_summary(per_record)
    method_summary = summarize_method(per_record, paths)
    method_summary["elapsed_seconds"] = time.time() - t0
    method_summary["control_method"] = args.control_method
    method_summary["model_name"] = args.model_name
    method_summary["control_scale"] = float(args.control_scale)
    method_summary["layers"] = list(args.layers)
    method_summary["path_id"] = f"{args.control_method}__layers_{'_'.join(str(layer) for layer in args.layers)}"
    method_summary["n_failures"] = sum(1 for _ in load_jsonl(failures_path)) if failures_path.exists() else 0

    write_csv(out_dir / "per_eval.csv", per_eval)
    write_csv(out_dir / "per_record.csv", per_record)
    write_csv(out_dir / "path_rows.csv", paths)
    (out_dir / "method_summary.json").write_text(json.dumps(method_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "STAGE_RUN_REPORT.md", args, method_summary)
    print(json.dumps(method_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
