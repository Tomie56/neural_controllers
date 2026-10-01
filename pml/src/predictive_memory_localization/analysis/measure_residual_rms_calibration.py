from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from predictive_memory_localization.common import (
    append_jsonl,
    load_jsonl,
    load_model_and_tokenizer,
    resolve_model_layers,
    set_seed,
)
from predictive_memory_localization.methods.run_bidirectional_three_table import (
    as_train_data,
    get_hidden_states,
)


DEFAULT_DATA_DIR = "/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure last-token residual RMS on frozen direction-construction prompts."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--record-ids-file", required=True)
    parser.add_argument("--model-local-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument(
        "--model-loader",
        default="auto",
        choices=["auto", "causal_lm", "image_text_to_text"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["record_id"]) for row in load_jsonl(path) if row.get("record_id")}


def quantile_summary(values: List[float]) -> Dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(array):
        raise ValueError("No finite residual RMS values")
    return {
        "n_values": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def build_summary(raw_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(load_jsonl(raw_path))
    values_by_layer: Dict[str, List[float]] = {str(layer): [] for layer in config["layers"]}
    prompt_counts: Dict[str, int] = {str(layer): 0 for layer in config["layers"]}
    for row in rows:
        for layer, values in row["residual_rms"].items():
            parsed = [float(value) for value in values]
            values_by_layer[layer].extend(parsed)
            prompt_counts[layer] += len(parsed)
    return {
        **config,
        "n_records": len({str(row["record_id"]) for row in rows}),
        "layers_summary": {
            layer: {
                **quantile_summary(values),
                "n_prompts": prompt_counts[layer],
            }
            for layer, values in values_by_layer.items()
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    ids_path = Path(args.record_ids_file)
    output_dir = Path(args.output_dir)
    raw_path = output_dir / "residual_rms_records.jsonl"
    failures_path = output_dir / "failures.jsonl"
    summary_path = output_dir / "residual_rms_summary.json"
    config_path = output_dir / "calibration_config.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "data_dir": str(data_dir),
        "record_ids_file": str(ids_path),
        "record_ids_sha256": file_sha256(ids_path),
        "model_local_path": args.model_local_path,
        "model_name": args.model_name,
        "layers": list(args.layers),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "model_loader": args.model_loader,
        "measurement": "last_token_per_coordinate_residual_rms_on_direction_construction_prompts",
    }
    if config_path.exists() and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"previous": previous.get(key), "current": value}
            for key, value in config.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Residual RMS calibration configuration mismatch. Use a new output directory "
                f"or --overwrite. Mismatches: {mismatches}"
            )
    if args.overwrite:
        for path in [raw_path, failures_path, summary_path, config_path]:
            path.unlink(missing_ok=True)
    elif raw_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {raw_path}. Use --resume or --overwrite.")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    requested_ids = {
        line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    records = [
        record
        for record in load_jsonl(data_dir / "records.jsonl")
        if str(record["record_id"]) in requested_ids
    ]
    found_ids = {str(record["record_id"]) for record in records}
    missing = sorted(requested_ids - found_ids)
    if missing:
        raise ValueError(f"Unknown calibration record IDs: {missing[:10]}")
    done = existing_record_ids(raw_path) if args.resume else set()
    records = [record for record in records if str(record["record_id"]) not in done]

    model, tokenizer = load_model_and_tokenizer(
        args.model_local_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        model_loader=args.model_loader,
        trust_remote_code=args.trust_remote_code,
    )
    model_layers, layer_path = resolve_model_layers(model)
    invalid = [layer for layer in args.layers if layer < -len(model_layers) or layer >= len(model_layers)]
    if invalid:
        raise ValueError(f"Invalid layers {invalid} for {len(model_layers)} layers at {layer_path}")
    print(
        f"Residual RMS calibration model={args.model_name} records_remaining={len(records)} "
        f"layers={args.layers} layer_path={layer_path}"
    )

    for record in tqdm(records, desc=f"{args.model_name} RMS records"):
        try:
            prompts, _ = as_train_data(record)
            states = get_hidden_states(prompts, model, tokenizer, args.layers, args.batch_size)
            residual_rms = {
                str(layer): states[layer].float().pow(2).mean(dim=1).sqrt().tolist()
                for layer in args.layers
            }
            append_jsonl(
                raw_path,
                {
                    "record_id": str(record["record_id"]),
                    "n_prompts": len(prompts),
                    "residual_rms": residual_rms,
                },
            )
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = build_summary(raw_path, config)
    unresolved_failures = list(load_jsonl(failures_path)) if failures_path.exists() else []
    successful_ids = existing_record_ids(raw_path)
    unresolved_failures = [
        row for row in unresolved_failures if str(row.get("record_id")) not in successful_ids
    ]
    if unresolved_failures:
        raise RuntimeError(
            f"Residual RMS calibration has {len(unresolved_failures)} unresolved failures; rerun with --resume."
        )
    if summary["n_records"] != len(requested_ids):
        raise RuntimeError(
            f"Residual RMS calibration incomplete: {summary['n_records']}/{len(requested_ids)} records"
        )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
