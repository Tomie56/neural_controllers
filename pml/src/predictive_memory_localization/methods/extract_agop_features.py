from __future__ import annotations

import argparse
import csv
import gc
import json
import math
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
from predictive_memory_localization.methods.run_activation_suppression import (
    DEFAULT_LAYERS,
    as_train_data,
    fit_mean_difference_direction,
    get_hidden_states,
    normalize_direction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RFM/AGOP memory-sensitivity features.")
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_float(value: Any) -> float:
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def effective_rank(eigenvalues: torch.Tensor) -> float:
    vals = eigenvalues.clamp_min(0).float()
    total = vals.sum()
    if total <= 0:
        return 0.0
    probs = vals / total
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return float(torch.exp(entropy).detach().cpu())


def spectral_entropy(eigenvalues: torch.Tensor) -> float:
    vals = eigenvalues.clamp_min(0).float()
    total = vals.sum()
    if total <= 0 or len(vals) <= 1:
        return 0.0
    probs = vals / total
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return float((entropy / math.log(len(vals))).detach().cpu())


def fit_rfm_agop(X: torch.Tensor, labels: Sequence[int], args: argparse.Namespace) -> torch.Tensor:
    from xrfm import RFM

    y = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    X = X.float().cpu()
    model = RFM(
        kernel=args.rfm_kernel,
        iters=args.rfm_iters,
        bandwidth=args.rfm_bandwidth,
        device="cpu",
        verbose=False,
        tuning_metric="mse",
        mem_gb=args.rfm_mem_gb,
    )
    matrices = model.fit(
        (X, y),
        val_data=(X, y),
        iters=args.rfm_iters,
        method="lstsq",
        reg=args.rfm_reg,
        verbose=False,
        return_Ms=True,
    )
    if not matrices:
        raise RuntimeError("RFM returned no AGOP matrices")
    agop = matrices[-1].detach().float().cpu()
    agop = 0.5 * (agop + agop.T)
    return agop


def agop_layer_features(
    agop: torch.Tensor,
    X: torch.Tensor,
    labels: Sequence[int],
    top_k: int,
) -> Dict[str, Any]:
    labels_t = torch.tensor(labels).bool()
    eigvals, eigvecs = torch.linalg.eigh(agop.float())
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0)
    eigvecs = eigvecs[:, order]
    total = eigvals.sum().clamp_min(1e-12)
    k = min(top_k, eigvals.numel())
    top_vec = normalize_direction(eigvecs[:, 0])
    mean_diff = fit_mean_difference_direction(X, torch.tensor(labels).long())
    top_vec_cos_mean_diff = float(torch.dot(top_vec, mean_diff).abs().detach().cpu())

    projections = X.float() @ top_vec.float()
    pos = projections[labels_t]
    neg = projections[~labels_t]
    pos_mean = pos.mean()
    neg_mean = neg.mean()
    if pos_mean < neg_mean:
        top_vec = -top_vec
        projections = -projections
        pos = projections[labels_t]
        neg = projections[~labels_t]
        pos_mean = pos.mean()
        neg_mean = neg.mean()
    pooled = torch.sqrt(0.5 * (pos.var(unbiased=False) + neg.var(unbiased=False)) + 1e-8)
    threshold = 0.5 * (pos_mean + neg_mean)
    accuracy = ((projections > threshold) == labels_t).float().mean()

    top1 = eigvals[0] if eigvals.numel() else torch.tensor(0.0)
    top2 = eigvals[1] if eigvals.numel() > 1 else torch.tensor(0.0)
    smallest = eigvals[-1] if eigvals.numel() else torch.tensor(0.0)
    return {
        "top1_ratio": safe_float(top1 / total),
        "topk_ratio": safe_float(eigvals[:k].sum() / total),
        "spectral_entropy": spectral_entropy(eigvals),
        "effective_rank": effective_rank(eigvals),
        "eigengap_1_2": safe_float(top1 - top2),
        "eigengap_ratio_1_2": safe_float(top1 / (top2 + 1e-12)),
        "condition_number": safe_float(top1 / (smallest + 1e-12)),
        "trace": safe_float(total),
        "fro_norm": safe_float(torch.linalg.matrix_norm(agop.float(), ord="fro")),
        "top_vec_mean_gap": safe_float(pos_mean - neg_mean),
        "top_vec_cohen_d": safe_float((pos_mean - neg_mean) / pooled),
        "top_vec_threshold_accuracy": safe_float(accuracy),
        "top_vec_abs_cos_mean_diff": top_vec_cos_mean_diff,
    }


def summarize_agop_features(layer_features: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if not layer_features:
        return {}
    layers = sorted(layer_features)
    out: Dict[str, Any] = {"agop_layers": layers}
    metric_names = sorted({key for values in layer_features.values() for key in values})
    for metric in metric_names:
        values = [float(layer_features[layer][metric]) for layer in layers if metric in layer_features[layer]]
        if not values:
            continue
        out[f"agop_mean_{metric}"] = float(sum(values) / len(values))
        out[f"agop_max_{metric}"] = float(max(values))
        out[f"agop_min_{metric}"] = float(min(values))
    top_layer = max(layers, key=lambda layer: float(layer_features[layer].get("top1_ratio", 0.0)))
    out["agop_top1_ratio_layer"] = top_layer
    entropy_values = [float(layer_features[layer].get("spectral_entropy", 0.0)) for layer in layers]
    if entropy_values:
        total = sum(max(x, 0.0) for x in entropy_values)
        out["agop_layer_entropy_mean"] = float(sum(entropy_values) / len(entropy_values))
        out["agop_layer_entropy_concentration"] = float(max(entropy_values) / total) if total > 0 else 0.0
    return out


def flatten_layer_features(prefix: str, layer_features: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for layer in sorted(layer_features):
        layer_name = str(layer).replace("-", "neg")
        for metric, value in layer_features[layer].items():
            if isinstance(value, (int, float)):
                out[f"{prefix}_layer_{layer_name}_{metric}"] = float(value)
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
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_jsonl = out_dir / "agop_features.jsonl"
    features_csv = out_dir / "agop_features.csv"
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
    for record in tqdm(records, desc="agop records"):
        try:
            validate_suppression_record(record)
            train_prompts, train_labels = as_train_data(record, tokenizer, args.format_prompts)
            hidden_states = get_hidden_states(train_prompts, model, tokenizer, args.layers, args.batch_size)
            per_layer: Dict[int, Dict[str, Any]] = {}
            for layer in args.layers:
                agop = fit_rfm_agop(hidden_states[layer], train_labels, args)
                per_layer[int(layer)] = agop_layer_features(agop, hidden_states[layer], train_labels, args.top_k)
            summary = summarize_agop_features(per_layer)
            row = {
                "id": record["id"],
                "concept": record.get("concept", record["id"]),
                "source_dataset": record.get("source", {}).get("dataset", ""),
                "n_train_positive": len(record.get("train_positive", [])),
                "n_train_negative": len(record.get("train_negative", [])),
                **summary,
                **flatten_layer_features("agop", per_layer),
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
    print(f"Wrote AGOP features to {features_jsonl}")
    print(f"Wrote AGOP feature CSV to {features_csv}")
    if failures_jsonl.exists():
        print(f"Wrote failures to {failures_jsonl}")


if __name__ == "__main__":
    main()
