from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset
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


DEFAULT_LAYERS = list(range(-1, -31, -4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pilot C for Predictive Memory Localization: learn memory directions "
            "and test inference-time activation suppression."
        )
    )
    parser.add_argument(
        "--model-local-path",
        type=str,
        default="/data/neural_controllers/models/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--model-name", type=str, default="llama_3_1_8b_it")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/neural_controllers/pml/results/suppression",
    )
    parser.add_argument(
        "--control-method",
        type=str,
        default="logistic",
        choices=["linear", "logistic", "mean_difference", "random"],
        help=(
            "Self-contained baseline directions. RFM/AGOP is not wired into this runner yet; "
            "use these methods as comparisons for the candidate AGOP/RFM localization method."
        ),
    )
    parser.add_argument("--n-components", type=int, default=1, help="Reserved for future RFM/AGOP runs; current pilot uses one direction.")
    parser.add_argument("--rfm-iters", type=int, default=8, help="Reserved for future RFM/AGOP runs.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=DEFAULT_LAYERS,
        help="Layer indices to learn and suppress. Default is a coarse sweep.",
    )
    parser.add_argument(
        "--suppression-coefs",
        type=float,
        nargs="*",
        default=[-0.1, -0.25, -0.5, -1.0],
        help="Negative values suppress the positive-label memory direction.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", type=str, default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument(
        "--format-prompts",
        action="store_true",
        help="Apply the tokenizer chat template to train/eval prompts.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also save greedy generations before and after suppression. Logprob scoring always runs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass
class DirectionController:
    model: Any
    tokenizer: Any
    directions: Dict[int, torch.Tensor]
    hidden_layers: List[int]
    batch_size: int

    def generate(self, prompt: str, layers_to_control=None, control_coef: Optional[float] = None, **kwargs) -> str:
        hooks = {}
        try:
            if layers_to_control and control_coef is not None:
                hooks = generation_utils.hook_model(
                    self.model,
                    self.directions,
                    layers_to_control,
                    control_coef,
                )
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            outputs = self.model.generate(**inputs, **kwargs)
            return self.tokenizer.decode(outputs[0])
        finally:
            if hooks:
                generation_utils.clear_hooks(hooks)


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
    for input_ids, attention_mask in tqdm(loader, desc="hidden states", leave=False):
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
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("control-method=logistic requires scikit-learn") from exc

    clf = LogisticRegression(fit_intercept=True, max_iter=1000)
    clf.fit(X.float().cpu().numpy(), y.long().cpu().numpy())
    return normalize_direction(torch.from_numpy(clf.coef_[0]))


def fit_mean_difference_direction(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_bool = y.bool()
    if y_bool.sum() == 0 or (~y_bool).sum() == 0:
        raise ValueError("mean_difference needs both positive and negative examples")
    vec = X[y_bool].float().mean(dim=0) - X[~y_bool].float().mean(dim=0)
    return normalize_direction(vec)


def fit_random_direction(X: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return normalize_direction(torch.randn(X.shape[1], generator=generator))


def compute_directions(
    model,
    tokenizer,
    train_prompts: Sequence[str],
    train_labels: Sequence[int],
    hidden_layers: Sequence[int],
    batch_size: int,
    control_method: str,
    seed: int,
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    hidden_states = get_hidden_states(train_prompts, model, tokenizer, hidden_layers, batch_size)
    labels = torch.tensor(train_labels).long()
    directions: Dict[int, torch.Tensor] = {}

    for layer in hidden_layers:
        X = hidden_states[layer]
        if control_method == "linear":
            vec = fit_linear_direction(X, labels)
        elif control_method == "logistic":
            vec = fit_logistic_direction(X, labels)
        elif control_method == "mean_difference":
            vec = fit_mean_difference_direction(X, labels)
        elif control_method == "random":
            vec = fit_random_direction(X, seed + abs(layer))
        else:
            raise ValueError(f"Unsupported control method: {control_method}")

        projections = X.float() @ vec.float()
        pos_mean = projections[labels.bool()].mean()
        neg_mean = projections[~labels.bool()].mean()
        if pos_mean < neg_mean:
            vec = -vec
        directions[layer] = vec.reshape(1, -1)

    return directions, hidden_states


def as_train_data(record: Dict[str, Any], tokenizer, format_prompts: bool) -> tuple[List[str], List[int]]:
    positives = [
        maybe_format_prompt(tokenizer, str(x), format_prompts)
        for x in record["train_positive"]
    ]
    negatives = [
        maybe_format_prompt(tokenizer, str(x), format_prompts)
        for x in record["train_negative"]
    ]
    return positives + negatives, [1] * len(positives) + [0] * len(negatives)


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
        accuracy = (preds == labels).float().mean()
        pos_accuracy = (pos > threshold).float().mean()
        neg_accuracy = (neg <= threshold).float().mean()
        margin = (scores[labels] - threshold).mean() - (scores[~labels] - threshold).mean()
        metrics[str(layer)] = {
            "pos_mean": float(pos_mean.detach().cpu()),
            "neg_mean": float(neg_mean.detach().cpu()),
            "mean_gap": float((pos_mean - neg_mean).detach().cpu()),
            "cohen_d": float(((pos_mean - neg_mean) / pooled).detach().cpu()),
            "threshold_accuracy": float(accuracy.detach().cpu()),
            "pos_accuracy": float(pos_accuracy.detach().cpu()),
            "neg_accuracy": float(neg_accuracy.detach().cpu()),
            "signed_margin_gap": float(margin.detach().cpu()),
            "score_std": float(scores.std(unbiased=False).detach().cpu()),
            "direction_norm": float(vec.norm().detach().cpu()),
        }
    return metrics


def summarize_localization_features(
    directions: Dict[int, torch.Tensor],
    local_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate per-layer localization metrics into record-level predictors."""
    if not local_metrics:
        return {}

    layer_items = sorted(
        ((int(layer), metrics) for layer, metrics in local_metrics.items()),
        key=lambda item: item[0],
    )
    saliency = [abs(float(metrics.get("cohen_d", 0.0))) for _, metrics in layer_items]
    saliency_sum = sum(saliency)
    if saliency_sum > 0:
        weights = [x / saliency_sum for x in saliency]
        entropy = -sum(w * math.log(w + 1e-12) for w in weights)
        normalized_entropy = entropy / math.log(len(weights)) if len(weights) > 1 else 0.0
        concentration = max(weights)
    else:
        weights = [0.0 for _ in saliency]
        normalized_entropy = 0.0
        concentration = 0.0

    top_idx = max(range(len(layer_items)), key=lambda idx: saliency[idx])
    accuracies = [float(metrics.get("threshold_accuracy", 0.0)) for _, metrics in layer_items]
    gaps = [float(metrics.get("mean_gap", 0.0)) for _, metrics in layer_items]
    margins = [float(metrics.get("signed_margin_gap", 0.0)) for _, metrics in layer_items]

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
        "mean_threshold_accuracy": float(sum(accuracies) / len(accuracies)),
        "max_threshold_accuracy": float(max(accuracies)),
        "mean_gap": float(sum(gaps) / len(gaps)),
        "max_gap": float(max(gaps)),
        "mean_signed_margin_gap": float(sum(margins) / len(margins)),
        "direction_pairwise_cosine_mean": float(sum(pairwise_cosines) / len(pairwise_cosines)) if pairwise_cosines else 0.0,
        "direction_pairwise_cosine_min": float(min(pairwise_cosines)) if pairwise_cosines else 0.0,
        "direction_pairwise_cosine_max": float(max(pairwise_cosines)) if pairwise_cosines else 0.0,
    }


def score_with_optional_control(
    controller: DirectionController,
    prompt: str,
    target: str,
    layers: Sequence[int],
    coef: float | None,
) -> Dict[str, Any]:
    hooks = {}
    try:
        if coef is not None:
            hooks = generation_utils.hook_model(controller.model, controller.directions, layers, coef)
        return continuation_logprob(controller.model, controller.tokenizer, prompt, target)
    finally:
        if hooks:
            generation_utils.clear_hooks(hooks)


def generate_with_optional_control(
    controller: DirectionController,
    prompt: str,
    layers: Sequence[int],
    coef: float | None,
    max_new_tokens: int,
) -> str:
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if coef is None:
        return controller.generate(prompt, **kwargs)
    return controller.generate(prompt, layers_to_control=list(layers), control_coef=coef, **kwargs)


def run_record(
    record: Dict[str, Any],
    controller: DirectionController,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    validate_suppression_record(record)
    train_prompts, train_labels = as_train_data(record, controller.tokenizer, args.format_prompts)
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
    local_metrics = projection_metrics(directions, hidden_states, train_labels)
    feature_summary = summarize_localization_features(directions, local_metrics)

    eval_rows: List[Dict[str, Any]] = []
    eval_items = list(record.get("eval", []))
    neighbor_items = list(record.get("neighbors", []))
    capability_items = list(record.get("capability", record.get("unrelated", [])))

    for kind, items in [
        ("target", eval_items),
        ("neighbor", neighbor_items),
        ("capability", capability_items),
    ]:
        for idx, item in enumerate(items):
            prompt = maybe_format_prompt(controller.tokenizer, str(item["prompt"]), args.format_prompts)
            target = str(item["target"])
            base = score_with_optional_control(controller, prompt, target, hidden_layers, None)

            row: Dict[str, Any] = {
                "kind": kind,
                "index": idx,
                "prompt": item["prompt"],
                "target": target,
                "base": base,
                "controlled": [],
            }
            if args.generate:
                row["base_generation"] = generate_with_optional_control(
                    controller,
                    prompt,
                    hidden_layers,
                    None,
                    args.max_new_tokens,
                )

            for coef in args.suppression_coefs:
                controlled = score_with_optional_control(controller, prompt, target, hidden_layers, coef)
                controlled["coef"] = coef
                controlled["delta_total_logprob"] = controlled["total_logprob"] - base["total_logprob"]
                controlled["delta_mean_logprob"] = controlled["mean_logprob"] - base["mean_logprob"]
                if args.generate:
                    controlled["generation"] = generate_with_optional_control(
                        controller,
                        prompt,
                        hidden_layers,
                        coef,
                        args.max_new_tokens,
                    )
                row["controlled"].append(controlled)

            eval_rows.append(row)

    return {
        "id": record["id"],
        "concept": record.get("concept", record["id"]),
        "control_method": args.control_method,
        "layers": hidden_layers,
        "n_components": args.n_components,
        "projection_metrics": local_metrics,
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

    if results_path.exists() and args.overwrite:
        results_path.unlink()
    if failures_path.exists() and args.overwrite:
        failures_path.unlink()

    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    model, tokenizer = load_model_and_tokenizer(
        args.model_local_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    controller = DirectionController(
        model,
        tokenizer,
        batch_size=args.batch_size,
        directions={},
        hidden_layers=list(args.layers),
    )

    records = list(load_jsonl(args.input_jsonl))
    if args.max_records is not None:
        records = records[: args.max_records]

    for record in tqdm(records, desc="suppression records"):
        try:
            result = run_record(record, controller, args)
            append_jsonl(results_path, result)
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
