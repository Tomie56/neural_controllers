from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset import (
    atomic_write_csv,
    atomic_write_json,
    discover_runs,
    numeric,
)


DEFAULT_MAIN_ROOT = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach leakage-free later-only path labels to an existing Activation-Space "
            "PML prediction dataset without recomputing its features."
        )
    )
    parser.add_argument("--source-dataset-dir", default=f"{DEFAULT_MAIN_ROOT}/prediction_dataset")
    parser.add_argument("--stage-root", default=f"{DEFAULT_MAIN_ROOT}/outcomes")
    parser.add_argument("--output-dir", default=f"{DEFAULT_MAIN_ROOT}/prediction_dataset_strict_later")
    parser.add_argument("--early-alpha", type=float, default=0.1)
    parser.add_argument("--later-alphas", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_later_alphas(values: Sequence[float], early_alpha: float) -> List[float]:
    normalized = sorted({abs(float(value)) for value in values})
    if any(value <= 0 for value in normalized):
        raise ValueError(f"Later alphas must be non-zero: {values}")
    if any(np.isclose(value, abs(early_alpha)) for value in normalized):
        raise ValueError(
            f"Later alphas {normalized} overlap early alpha {abs(early_alpha)}. "
            "Use disjoint strengths to avoid feature-label leakage."
        )
    return normalized


def strict_labels_for_run(
    method: str,
    layer: int,
    layer_dir: Path,
    thresholds: Dict[str, float],
    later_alphas: Sequence[float],
) -> pd.DataFrame:
    columns = [
        "record_id",
        "alpha",
        "target_mean_delta_margin",
        "neighbor_mean_delta_margin",
        "capability_mean_delta_margin",
    ]
    frame = pd.read_csv(layer_dir / "per_record.csv", usecols=columns)
    frame["record_id"] = frame["record_id"].astype(str)
    frame["alpha"] = numeric(frame["alpha"])
    absolute_alpha = frame["alpha"].abs().to_numpy()
    later_mask = np.zeros(len(frame), dtype=bool)
    for alpha in later_alphas:
        later_mask |= np.isclose(absolute_alpha, alpha)
    later = frame.loc[later_mask].copy()
    observed = sorted(float(value) for value in later["alpha"].abs().dropna().unique())
    missing_alphas = [
        alpha for alpha in later_alphas if not any(np.isclose(alpha, value) for value in observed)
    ]
    if missing_alphas:
        raise ValueError(
            f"Run {method}/layer={layer} is missing later alphas {missing_alphas}; observed={observed}"
        )

    target_delta = numeric(later["target_mean_delta_margin"])
    neighbor_delta = numeric(later["neighbor_mean_delta_margin"])
    capability_delta = numeric(later["capability_mean_delta_margin"])
    negative = later["alpha"] < 0
    positive = later["alpha"] > 0
    target_success = (negative & (target_delta <= -thresholds["target"])) | (
        positive & (target_delta >= thresholds["target"])
    )
    neighbor_damage = neighbor_delta <= -thresholds["neighbor"]
    capability_damage = capability_delta <= -thresholds["capability"]
    any_damage = neighbor_damage | capability_damage

    later["later_suppression_path"] = negative & target_success
    later["later_enhancement_path"] = positive & target_success
    later["later_neighbor_damage_any_path"] = neighbor_damage
    later["later_capability_damage_any_path"] = capability_damage
    later["later_clean_suppression_path"] = negative & target_success & ~any_damage
    later["later_clean_enhancement_path"] = positive & target_success & ~any_damage

    label_columns = [
        "later_suppression_path",
        "later_enhancement_path",
        "later_neighbor_damage_any_path",
        "later_capability_damage_any_path",
        "later_clean_suppression_path",
        "later_clean_enhancement_path",
    ]
    grouped = later.groupby("record_id", sort=False)[label_columns].max().astype(int).reset_index()
    grouped["later_target_any_path"] = (
        grouped["later_suppression_path"] | grouped["later_enhancement_path"]
    ).astype(int)
    grouped["later_clean_any_path"] = (
        grouped["later_clean_suppression_path"] | grouped["later_clean_enhancement_path"]
    ).astype(int)
    grouped["later_no_effect_path"] = (1 - grouped["later_target_any_path"]).astype(int)
    grouped.insert(1, "method", method)
    grouped.insert(2, "layer", layer)
    return grouped


def expected_config(
    source_dataset_dir: Path,
    stage_root: Path,
    early_alpha: float,
    later_alphas: Sequence[float],
) -> Dict[str, Any]:
    return {
        "source_dataset_dir": str(source_dataset_dir),
        "stage_root": str(stage_root),
        "early_alpha": early_alpha,
        "later_alphas": list(later_alphas),
    }


def validate_existing_manifest(path: Path, expected: Dict[str, Any]) -> Dict[str, Any]:
    previous = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"previous": previous.get(key), "current": value}
        for key, value in expected.items()
        if previous.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Strict-later dataset resume configuration mismatch. Use a new output directory "
            f"or --overwrite. Mismatches: {mismatches}"
        )
    return previous


def main() -> None:
    args = parse_args()
    source_dataset_dir = Path(args.source_dataset_dir)
    stage_root = Path(args.stage_root)
    output_dir = Path(args.output_dir)
    later_alphas = normalize_later_alphas(args.later_alphas, args.early_alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "pml_path_prediction_dataset.csv"
    manifest_path = output_dir / "prediction_dataset_manifest.json"
    config = expected_config(source_dataset_dir, stage_root, args.early_alpha, later_alphas)

    if dataset_path.exists() and manifest_path.exists() and not args.overwrite:
        previous = validate_existing_manifest(manifest_path, config)
        print(json.dumps(previous, ensure_ascii=False, indent=2))
        return

    source_dataset_path = source_dataset_dir / "pml_path_prediction_dataset.csv"
    source_manifest_path = source_dataset_dir / "prediction_dataset_manifest.json"
    if not source_dataset_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError(f"Missing source prediction dataset under {source_dataset_dir}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    thresholds = source_manifest.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"Missing thresholds in {source_manifest_path}")

    source = pd.read_csv(source_dataset_path)
    source["record_id"] = source["record_id"].astype(str)
    runs = discover_runs(stage_root)
    label_frames: List[pd.DataFrame] = []
    for method, layer, layer_dir in runs:
        print(f"[strict later] method={method} layer={layer}")
        label_frames.append(
            strict_labels_for_run(method, layer, layer_dir, thresholds, later_alphas)
        )
    labels = pd.concat(label_frames, ignore_index=True)
    key_columns = ["record_id", "method", "layer"]
    if labels.duplicated(key_columns).any():
        raise ValueError("Strict later labels contain duplicate record/method/layer keys")
    merged = source.merge(labels, on=key_columns, how="left", validate="one_to_one")
    later_columns = [column for column in labels.columns if column.startswith("later_")]
    missing = merged[later_columns].isna().any(axis=1)
    if missing.any():
        examples = merged.loc[missing, key_columns].head(10).to_dict(orient="records")
        raise ValueError(f"Missing strict later labels for {int(missing.sum())} rows: {examples}")
    merged[later_columns] = merged[later_columns].astype(int)
    atomic_write_csv(dataset_path, merged)

    manifest = {
        **source_manifest,
        **config,
        "output_dir": str(output_dir),
        "dataset_path": str(dataset_path),
        "source_dataset_path": str(source_dataset_path),
        "later_label_protocol": (
            "Features in group R use only +/-early_alpha. Labels with the later_ prefix "
            "use only strengths in later_alphas, which are validated to be disjoint."
        ),
        "n_rows": int(len(merged)),
        "n_records": int(merged["record_id"].nunique()),
        "strict_later_targets": later_columns,
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
