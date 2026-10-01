from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import (
    append_jsonl,
    load_jsonl,
    load_model_and_tokenizer,
    maybe_format_prompt,
    set_seed,
    validate_suppression_record,
)
from predictive_memory_localization.methods.extract_agop_features import (
    agop_layer_features,
    fit_rfm_agop,
    flatten_layer_features,
    write_csv,
)
from predictive_memory_localization.methods.run_activation_suppression import (
    DEFAULT_LAYERS,
    as_train_data,
    get_hidden_states,
    normalize_direction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract target-neighbor AGOP overlap features.")
    parser.add_argument("--model-local-path", default="/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, nargs="*", default=DEFAULT_LAYERS)
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
    parser.add_argument("--neighbor-rank", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def item_texts(items: Sequence[Dict[str, Any]], tokenizer, format_prompts: bool) -> List[str]:
    texts = []
    for item in items:
        prompt = maybe_format_prompt(tokenizer, str(item["prompt"]), format_prompts)
        texts.append(prompt + str(item["target"]))
    return texts


def top_eigenspace(matrix: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    eigvals, eigvecs = torch.linalg.eigh(matrix.float())
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0)
    eigvecs = eigvecs[:, order]
    k = min(k, eigvecs.shape[1])
    return eigvals[:k], eigvecs[:, :k].float()


def pca_basis(states: torch.Tensor, rank: int) -> torch.Tensor:
    X = states.float()
    if X.shape[0] == 1:
        return normalize_direction(X[0]).reshape(-1, 1)
    X = X - X.mean(dim=0, keepdim=True)
    try:
        _, _, vh = torch.linalg.svd(X, full_matrices=False)
        k = min(rank, vh.shape[0])
        return vh[:k].T.contiguous().float()
    except RuntimeError:
        cov = X.T @ X
        _, basis = top_eigenspace(cov, rank)
        return basis.float()


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    q_left, _ = torch.linalg.qr(left.float(), mode="reduced")
    q_right, _ = torch.linalg.qr(right.float(), mode="reduced")
    singular_values = torch.linalg.svdvals(q_left.T @ q_right)
    return float((singular_values.square().mean()).detach().cpu())


def projection_norm(vec: torch.Tensor, basis: torch.Tensor) -> float:
    if basis.numel() == 0:
        return 0.0
    q, _ = torch.linalg.qr(basis.float(), mode="reduced")
    v = normalize_direction(vec.reshape(-1))
    return float((q.T @ v).square().sum().detach().cpu())


def cosine_to_centroid(vec: torch.Tensor, states: torch.Tensor) -> float:
    centroid = states.float().mean(dim=0)
    if centroid.norm() <= 0:
        return 0.0
    v = normalize_direction(vec.reshape(-1))
    c = normalize_direction(centroid)
    return float(torch.dot(v, c).abs().detach().cpu())


def layer_overlap_features(
    agop: torch.Tensor,
    train_states: torch.Tensor,
    train_labels: Sequence[int],
    eval_states: torch.Tensor,
    neighbor_states: torch.Tensor,
    top_k: int,
    neighbor_rank: int,
) -> Dict[str, Any]:
    layer_features = agop_layer_features(agop, train_states, train_labels, top_k)
    eigvals, eigvecs = top_eigenspace(agop, top_k)
    top_vec = eigvecs[:, 0]
    neighbor_basis = pca_basis(neighbor_states, neighbor_rank)
    eval_basis = pca_basis(eval_states, min(neighbor_rank, eval_states.shape[0]))
    combined_basis = pca_basis(torch.cat([eval_states, neighbor_states], dim=0), neighbor_rank)
    total = eigvals.sum().clamp_min(1e-12)
    weighted_neighbor_projection = 0.0
    weighted_eval_projection = 0.0
    for idx in range(eigvecs.shape[1]):
        weight = float((eigvals[idx] / total).detach().cpu())
        weighted_neighbor_projection += weight * projection_norm(eigvecs[:, idx], neighbor_basis)
        weighted_eval_projection += weight * projection_norm(eigvecs[:, idx], eval_basis)
    return {
        **layer_features,
        "neighbor_top1_projection": projection_norm(top_vec, neighbor_basis),
        "eval_top1_projection": projection_norm(top_vec, eval_basis),
        "combined_top1_projection": projection_norm(top_vec, combined_basis),
        "neighbor_topk_weighted_projection": weighted_neighbor_projection,
        "eval_topk_weighted_projection": weighted_eval_projection,
        "target_neighbor_subspace_overlap": subspace_overlap(eigvecs, neighbor_basis),
        "target_eval_subspace_overlap": subspace_overlap(eigvecs, eval_basis),
        "neighbor_eval_subspace_overlap": subspace_overlap(neighbor_basis, eval_basis),
        "top_vec_neighbor_centroid_cosine": cosine_to_centroid(top_vec, neighbor_states),
        "top_vec_eval_centroid_cosine": cosine_to_centroid(top_vec, eval_states),
    }


def summarize_overlap(layer_features: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overlap_layers": sorted(layer_features)}
    metric_names = sorted({key for values in layer_features.values() for key in values})
    for metric in metric_names:
        values = [float(layer_features[layer][metric]) for layer in sorted(layer_features) if metric in layer_features[layer]]
        if not values:
            continue
        out[f"overlap_mean_{metric}"] = float(sum(values) / len(values))
        out[f"overlap_max_{metric}"] = float(max(values))
        out[f"overlap_min_{metric}"] = float(min(values))
    top_layer = max(
        layer_features,
        key=lambda layer: float(layer_features[layer].get("target_neighbor_subspace_overlap", 0.0)),
    )
    out["overlap_top_neighbor_layer"] = int(top_layer)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_jsonl = out_dir / "overlap_features.jsonl"
    features_csv = out_dir / "overlap_features.csv"
    failures_jsonl = out_dir / "failures.jsonl"
    config_json = out_dir / "config.json"

    if args.overwrite:
        for path in [features_jsonl, features_csv, failures_jsonl]:
            if path.exists():
                path.unlink()
    config_json.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    model, tokenizer = load_model_and_tokenizer(args.model_local_path, args.device_map, args.torch_dtype)
    records = list(load_jsonl(args.input_jsonl))
    if args.max_records is not None:
        records = records[: args.max_records]

    all_rows: List[Dict[str, Any]] = []
    for record in tqdm(records, desc="agop overlap records"):
        try:
            validate_suppression_record(record)
            train_prompts, train_labels = as_train_data(record, tokenizer, args.format_prompts)
            eval_texts = item_texts(record.get("eval", []), tokenizer, args.format_prompts)
            neighbor_texts = item_texts(record.get("neighbors", []), tokenizer, args.format_prompts)
            if not eval_texts or not neighbor_texts:
                raise ValueError("Need nonempty eval and neighbor items for overlap")

            train_states = get_hidden_states(train_prompts, model, tokenizer, args.layers, args.batch_size)
            eval_states = get_hidden_states(eval_texts, model, tokenizer, args.layers, args.batch_size)
            neighbor_states = get_hidden_states(neighbor_texts, model, tokenizer, args.layers, args.batch_size)

            per_layer: Dict[int, Dict[str, Any]] = {}
            for layer in args.layers:
                agop = fit_rfm_agop(train_states[layer], train_labels, args)
                per_layer[int(layer)] = layer_overlap_features(
                    agop,
                    train_states[layer],
                    train_labels,
                    eval_states[layer],
                    neighbor_states[layer],
                    args.top_k,
                    args.neighbor_rank,
                )
            row = {
                "id": record["id"],
                "concept": record.get("concept", record["id"]),
                "source_dataset": record.get("source", {}).get("dataset", ""),
                "n_eval": len(eval_texts),
                "n_neighbor": len(neighbor_texts),
                **summarize_overlap(per_layer),
                **flatten_layer_features("overlap", per_layer),
            }
            append_jsonl(features_jsonl, row)
            all_rows.append(row)
        except Exception as exc:
            append_jsonl(
                failures_jsonl,
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

    write_csv(features_csv, all_rows)
    print(f"Wrote overlap features to {features_jsonl}")
    print(f"Wrote overlap feature CSV to {features_csv}")
    if failures_jsonl.exists():
        print(f"Wrote failures to {failures_jsonl}")


if __name__ == "__main__":
    main()
