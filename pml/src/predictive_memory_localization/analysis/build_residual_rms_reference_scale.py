from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed layer-level residual-RMS scales relative to a reference model."
    )
    parser.add_argument("--reference-model-key", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--reference-mapping", required=True)
    parser.add_argument("--target-model-keys", nargs="+", required=True)
    parser.add_argument("--target-summaries", nargs="+", required=True)
    parser.add_argument("--target-mappings", nargs="+", required=True)
    parser.add_argument("--reference-alphas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_median(summary: Dict[str, Any], layer: int) -> float:
    try:
        value = float(summary["layers_summary"][str(layer)]["median"])
    except KeyError as exc:
        raise ValueError(f"Residual-RMS summary is missing layer {layer}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid residual RMS median for layer {layer}: {value}")
    return value


def model_hidden_size(summary: Dict[str, Any]) -> int:
    config_path = Path(str(summary.get("model_local_path", ""))) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config for hidden-size lookup: {config_path}")
    config = load_json(config_path)
    text_config = config.get("text_config") or config
    hidden_size = int(text_config.get("hidden_size") or 0)
    if hidden_size <= 0:
        raise ValueError(f"Invalid hidden_size in {config_path}: {hidden_size}")
    return hidden_size


def mapping_by_reference(mapping: Dict[str, Any]) -> Dict[int, int]:
    rows = mapping.get("mappings") or []
    result = {int(row["reference_layer"]): int(row["target_layer"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("Layer mapping contains duplicate reference layers")
    return result


def model_payload(
    model_key: str,
    summary_path: str | Path,
    mapping_path: str | Path,
    reference_summary: Dict[str, Any],
    reference_mapping: Dict[int, int],
    reference_hidden_size: int,
    reference_alphas: List[float],
) -> Dict[str, Any]:
    summary = load_json(summary_path)
    target_hidden_size = model_hidden_size(summary)
    target_mapping = mapping_by_reference(load_json(mapping_path))
    if set(target_mapping) != set(reference_mapping):
        raise ValueError(
            f"Reference-layer mismatch for {model_key}: "
            f"reference={sorted(reference_mapping)} target={sorted(target_mapping)}"
        )
    layers: Dict[str, Any] = {}
    for reference_layer in reference_mapping:
        reference_target_layer = reference_mapping[reference_layer]
        target_layer = target_mapping[reference_layer]
        reference_rms = layer_median(reference_summary, reference_target_layer)
        target_rms = layer_median(summary, target_layer)
        residual_rms_ratio = target_rms / reference_rms
        hidden_size_ratio = target_hidden_size / reference_hidden_size
        dimension_correction = math.sqrt(hidden_size_ratio)
        control_scale = residual_rms_ratio * dimension_correction
        layers[str(target_layer)] = {
            "reference_layer": reference_layer,
            "reference_target_layer": reference_target_layer,
            "target_layer": target_layer,
            "reference_residual_rms_median": reference_rms,
            "target_residual_rms_median": target_rms,
            "reference_hidden_size": reference_hidden_size,
            "target_hidden_size": target_hidden_size,
            "residual_rms_ratio": residual_rms_ratio,
            "hidden_size_ratio": hidden_size_ratio,
            "dimension_correction_sqrt_ratio": dimension_correction,
            "control_scale": control_scale,
            "effective_control_coefficients": {
                str(alpha): alpha * control_scale for alpha in reference_alphas
            },
        }
    return {
        "model_key": model_key,
        "model_name": summary.get("model_name"),
        "model_local_path": summary.get("model_local_path"),
        "hidden_size": target_hidden_size,
        "summary_path": str(Path(summary_path)),
        "summary_sha256": file_sha256(summary_path),
        "mapping_path": str(Path(mapping_path)),
        "mapping_sha256": file_sha256(mapping_path),
        "layers": layers,
    }


def main() -> None:
    args = parse_args()
    if not (
        len(args.target_model_keys)
        == len(args.target_summaries)
        == len(args.target_mappings)
    ):
        raise ValueError("Target model keys, summaries, and mappings must have equal lengths")

    output_path = Path(args.output_json)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")

    reference_summary = load_json(args.reference_summary)
    reference_hidden_size = model_hidden_size(reference_summary)
    reference_mapping_payload = load_json(args.reference_mapping)
    reference_mapping = mapping_by_reference(reference_mapping_payload)
    calibration_hash = reference_summary.get("record_ids_sha256")
    if not calibration_hash:
        raise ValueError("Reference summary is missing record_ids_sha256")

    all_inputs = [
        (args.reference_model_key, args.reference_summary, args.reference_mapping),
        *zip(args.target_model_keys, args.target_summaries, args.target_mappings),
    ]
    for model_key, summary_path, _ in all_inputs:
        summary = load_json(summary_path)
        if summary.get("record_ids_sha256") != calibration_hash:
            raise ValueError(
                f"Calibration record mismatch for {model_key}: "
                f"{summary.get('record_ids_sha256')} != {calibration_hash}"
            )

    models: Dict[str, Any] = {}
    for model_key, summary_path, mapping_path in all_inputs:
        models[model_key] = model_payload(
            model_key=model_key,
            summary_path=summary_path,
            mapping_path=mapping_path,
            reference_summary=reference_summary,
            reference_mapping=reference_mapping,
            reference_hidden_size=reference_hidden_size,
            reference_alphas=args.reference_alphas,
        )

    payload = {
        "protocol": "residual_rms_reference_matching_dimension_corrected_v2",
        "measurement": "median_last_token_per_coordinate_residual_rms_on_direction_construction_prompts",
        "direction_normalization": "unit_l2_norm",
        "reference_model_key": args.reference_model_key,
        "reference_hidden_size": reference_hidden_size,
        "reference_alphas": args.reference_alphas,
        "record_ids_sha256": calibration_hash,
        "n_calibration_records": int(reference_summary.get("n_records", 0)),
        "matched_quantity": "l2_steering_perturbation_norm / residual_state_l2_norm",
        "scale_rule": "(target_residual_rms / reference_residual_rms) * sqrt(target_hidden_size / reference_hidden_size)",
        "derivation": "For unit-L2 direction v, ||alpha*v||_2 / ||h||_2 = |alpha| / (sqrt(hidden_size) * residual_rms).",
        "application_rule": "effective_control_coefficient = reference_alpha * control_scale",
        "models": models,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
