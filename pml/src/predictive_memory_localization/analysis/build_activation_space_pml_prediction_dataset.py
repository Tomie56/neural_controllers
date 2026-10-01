from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_STAGE_ROOT = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b/outcomes"
)
DEFAULT_OUTPUT_DIR = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b/prediction_dataset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the path-level dataset for the Activation-Space PML main prediction experiment."
    )
    parser.add_argument("--stage-root", default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-method", default="random")
    parser.add_argument("--null-quantile", type=float, default=0.95)
    parser.add_argument("--early-alpha", type=float, default=0.1)
    parser.add_argument("--medium-alpha", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def layer_from_dir(path: Path) -> int:
    return int(path.name.replace("layer_", "").replace("neg", "-"))


def discover_runs(stage_root: Path) -> List[tuple[str, int, Path]]:
    runs: List[tuple[str, int, Path]] = []
    for method_dir in sorted(path for path in stage_root.iterdir() if path.is_dir()):
        for layer_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            if (layer_dir / "per_record.csv").is_file() and (layer_dir / "per_eval.csv").is_file():
                runs.append((method_dir.name, layer_from_dir(layer_dir), layer_dir))
    if not runs:
        raise FileNotFoundError(f"No completed method/layer runs found under {stage_root}")
    return runs


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def finite(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def compute_null_thresholds(
    runs: Sequence[tuple[str, int, Path]], random_method: str, quantile: float
) -> Dict[str, float]:
    values: Dict[str, List[float]] = {"target": [], "neighbor": [], "capability": []}
    random_runs = 0
    for method, _, layer_dir in runs:
        if method != random_method:
            continue
        random_runs += 1
        frame = pd.read_csv(
            layer_dir / "per_record.csv",
            usecols=[
                "alpha",
                "target_mean_delta_margin",
                "neighbor_mean_delta_margin",
                "capability_mean_delta_margin",
            ],
        )
        frame = frame[~np.isclose(numeric(frame["alpha"]), 0.0)]
        for metric in values:
            column = numeric(frame[f"{metric}_mean_delta_margin"]).abs().dropna()
            values[metric].extend(column.tolist())
    if random_runs == 0:
        raise ValueError(f"No runs found for random method {random_method!r}")
    return {
        metric: float(np.quantile(finite(metric_values), quantile))
        for metric, metric_values in values.items()
    }


def summarize_base_margins(per_eval: pd.DataFrame) -> pd.DataFrame:
    base = per_eval[np.isclose(numeric(per_eval["alpha"]), 0.0)].copy()
    base["base_margin"] = numeric(base["base_margin"])
    mean_margin_column = "base_mean_margin" if "base_mean_margin" in base.columns else "mean_margin"
    base[mean_margin_column] = numeric(base[mean_margin_column])
    rows: List[Dict[str, Any]] = []
    for record_id, record_frame in base.groupby("record_id", sort=False):
        row: Dict[str, Any] = {"record_id": record_id}
        for eval_type in ["target", "neighbor", "capability"]:
            subset = record_frame[record_frame["eval_type"] == eval_type]
            margins = numeric(subset["base_margin"]).dropna()
            mean_margins = numeric(subset[mean_margin_column]).dropna()
            prefix = f"base_{eval_type}"
            row[f"{prefix}_margin_mean"] = float(margins.mean()) if len(margins) else np.nan
            row[f"{prefix}_margin_median"] = float(margins.median()) if len(margins) else np.nan
            row[f"{prefix}_margin_min"] = float(margins.min()) if len(margins) else np.nan
            row[f"{prefix}_margin_max"] = float(margins.max()) if len(margins) else np.nan
            row[f"{prefix}_positive_rate"] = float((margins > 0).mean()) if len(margins) else np.nan
            row[f"{prefix}_mean_margin_mean"] = float(mean_margins.mean()) if len(mean_margins) else np.nan
        row["base_target_neighbor_gap"] = (
            row["base_target_margin_mean"] - row["base_neighbor_margin_mean"]
        )
        row["base_target_capability_gap"] = (
            row["base_target_margin_mean"] - row["base_capability_margin_mean"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def row_at_alpha(frame: pd.DataFrame, alpha: float) -> pd.Series | None:
    selected = frame[np.isclose(numeric(frame["alpha"]), alpha)]
    return None if selected.empty else selected.iloc[0]


def value(row: pd.Series | None, column: str) -> float:
    if row is None:
        return float("nan")
    parsed = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float("nan")


def first_onset(frame: pd.DataFrame, sign: int, predicate) -> float:
    signed = frame[numeric(frame["alpha"]) * sign > 0].copy()
    signed["abs_alpha"] = numeric(signed["alpha"]).abs()
    for _, row in signed.sort_values("abs_alpha").iterrows():
        if predicate(row):
            return float(row["alpha"])
    return float("nan")


def build_path_row(
    method: str,
    layer: int,
    frame: pd.DataFrame,
    base_row: Dict[str, Any],
    thresholds: Dict[str, float],
    early_alpha: float,
    medium_alpha: float,
) -> Dict[str, Any]:
    first = frame.iloc[0]
    target_tau = thresholds["target"]
    neighbor_tau = thresholds["neighbor"]
    capability_tau = thresholds["capability"]

    def target_success(row: pd.Series) -> bool:
        alpha = float(row["alpha"])
        delta = float(row["target_mean_delta_margin"])
        return (alpha < 0 and delta <= -target_tau) or (alpha > 0 and delta >= target_tau)

    def neighbor_damage(row: pd.Series) -> bool:
        return float(row["neighbor_mean_delta_margin"]) <= -neighbor_tau

    def capability_damage(row: pd.Series) -> bool:
        return float(row["capability_mean_delta_margin"]) <= -capability_tau

    def any_damage(row: pd.Series) -> bool:
        return neighbor_damage(row) or capability_damage(row)

    negative = frame[numeric(frame["alpha"]) < 0]
    positive = frame[numeric(frame["alpha"]) > 0]
    suppression_onset = first_onset(frame, -1, target_success)
    enhancement_onset = first_onset(frame, 1, target_success)
    negative_damage_onset = first_onset(frame, -1, any_damage)
    positive_damage_onset = first_onset(frame, 1, any_damage)
    negative_neighbor_onset = first_onset(frame, -1, neighbor_damage)
    positive_neighbor_onset = first_onset(frame, 1, neighbor_damage)
    negative_capability_onset = first_onset(frame, -1, capability_damage)
    positive_capability_onset = first_onset(frame, 1, capability_damage)
    clean_suppression = any(target_success(row) and not any_damage(row) for _, row in negative.iterrows())
    clean_enhancement = any(target_success(row) and not any_damage(row) for _, row in positive.iterrows())

    medium_neg = row_at_alpha(frame, -abs(medium_alpha))
    medium_pos = row_at_alpha(frame, abs(medium_alpha))
    early_neg = row_at_alpha(frame, -abs(early_alpha))
    early_pos = row_at_alpha(frame, abs(early_alpha))

    row: Dict[str, Any] = {
        **base_row,
        "record_id": str(first["record_id"]),
        "method": method,
        "layer": layer,
        "dataset": first.get("dataset"),
        "domain": first.get("domain"),
        "freshness_group": first.get("freshness_group"),
        "release_year": first.get("release_year"),
        "n_target_eval": value(first, "n_target_eval"),
        "n_neighbor_eval": value(first, "n_neighbor_eval"),
        "n_capability_eval": value(first, "n_capability_eval"),
        "suppression_onset_alpha": suppression_onset,
        "enhancement_onset_alpha": enhancement_onset,
        "negative_damage_onset_alpha": negative_damage_onset,
        "positive_damage_onset_alpha": positive_damage_onset,
        "negative_neighbor_onset_alpha": negative_neighbor_onset,
        "positive_neighbor_onset_alpha": positive_neighbor_onset,
        "negative_capability_onset_alpha": negative_capability_onset,
        "positive_capability_onset_alpha": positive_capability_onset,
        "suppression_any_path": int(math.isfinite(suppression_onset)),
        "enhancement_any_path": int(math.isfinite(enhancement_onset)),
        "target_any_path": int(math.isfinite(suppression_onset) or math.isfinite(enhancement_onset)),
        "neighbor_damage_any_path": int(
            math.isfinite(negative_neighbor_onset) or math.isfinite(positive_neighbor_onset)
        ),
        "capability_damage_any_path": int(
            math.isfinite(negative_capability_onset) or math.isfinite(positive_capability_onset)
        ),
        "damage_any_path": int(math.isfinite(negative_damage_onset) or math.isfinite(positive_damage_onset)),
        "clean_suppression_path": int(clean_suppression),
        "clean_enhancement_path": int(clean_enhancement),
        "clean_any_path": int(clean_suppression or clean_enhancement),
        "bidirectional_clean_path": int(clean_suppression and clean_enhancement),
        "no_effect_path": int(not math.isfinite(suppression_onset) and not math.isfinite(enhancement_onset)),
        "suppression_gain_medium": -value(medium_neg, "target_mean_delta_margin"),
        "enhancement_gain_medium": value(medium_pos, "target_mean_delta_margin"),
        "neighbor_delta_negative_medium": value(medium_neg, "neighbor_mean_delta_margin"),
        "neighbor_delta_positive_medium": value(medium_pos, "neighbor_mean_delta_margin"),
        "capability_delta_negative_medium": value(medium_neg, "capability_mean_delta_margin"),
        "capability_delta_positive_medium": value(medium_pos, "capability_mean_delta_margin"),
        "suppression_success_medium": int(
            value(medium_neg, "target_mean_delta_margin") <= -target_tau
        ),
        "enhancement_success_medium": int(
            value(medium_pos, "target_mean_delta_margin") >= target_tau
        ),
        "negative_neighbor_damage_medium": int(
            value(medium_neg, "neighbor_mean_delta_margin") <= -neighbor_tau
        ),
        "positive_neighbor_damage_medium": int(
            value(medium_pos, "neighbor_mean_delta_margin") <= -neighbor_tau
        ),
        "negative_capability_damage_medium": int(
            value(medium_neg, "capability_mean_delta_margin") <= -capability_tau
        ),
        "positive_capability_damage_medium": int(
            value(medium_pos, "capability_mean_delta_margin") <= -capability_tau
        ),
        "early_target_delta_negative": value(early_neg, "target_mean_delta_margin"),
        "early_target_delta_positive": value(early_pos, "target_mean_delta_margin"),
        "early_neighbor_delta_negative": value(early_neg, "neighbor_mean_delta_margin"),
        "early_neighbor_delta_positive": value(early_pos, "neighbor_mean_delta_margin"),
        "early_capability_delta_negative": value(early_neg, "capability_mean_delta_margin"),
        "early_capability_delta_positive": value(early_pos, "capability_mean_delta_margin"),
    }
    row["clean_suppression_medium"] = int(
        row["suppression_success_medium"]
        and not row["negative_neighbor_damage_medium"]
        and not row["negative_capability_damage_medium"]
    )
    row["clean_enhancement_medium"] = int(
        row["enhancement_success_medium"]
        and not row["positive_neighbor_damage_medium"]
        and not row["positive_capability_damage_medium"]
    )
    row["early_target_slope"] = (
        row["early_target_delta_positive"] - row["early_target_delta_negative"]
    ) / (2.0 * abs(early_alpha))
    row["early_neighbor_slope"] = (
        row["early_neighbor_delta_positive"] - row["early_neighbor_delta_negative"]
    ) / (2.0 * abs(early_alpha))
    row["early_capability_slope"] = (
        row["early_capability_delta_positive"] - row["early_capability_delta_negative"]
    ) / (2.0 * abs(early_alpha))
    for column in frame.columns:
        if column.startswith("feature_"):
            row[column] = first.get(column)
    return row


def build_run_rows(
    method: str,
    layer: int,
    layer_dir: Path,
    thresholds: Dict[str, float],
    early_alpha: float,
    medium_alpha: float,
) -> List[Dict[str, Any]]:
    per_record = pd.read_csv(layer_dir / "per_record.csv")
    per_eval_path = layer_dir / "per_eval.csv"
    header = pd.read_csv(per_eval_path, nrows=0).columns
    mean_margin_column = "base_mean_margin" if "base_mean_margin" in header else "mean_margin"
    per_eval = pd.read_csv(
        per_eval_path,
        usecols=["alpha", "record_id", "eval_type", "base_margin", mean_margin_column],
    )
    base_frame = summarize_base_margins(per_eval)
    base_by_record = base_frame.set_index("record_id").to_dict(orient="index")
    rows: List[Dict[str, Any]] = []
    for record_id, frame in per_record.groupby("record_id", sort=False):
        rows.append(
            build_path_row(
                method,
                layer,
                frame,
                base_by_record.get(record_id, {}),
                thresholds,
                early_alpha,
                medium_alpha,
            )
        )
    return rows


def feature_groups(columns: Sequence[str]) -> Dict[str, List[str]]:
    base = sorted(column for column in columns if column.startswith("base_"))
    metadata = [
        column
        for column in ["method", "layer", "dataset", "domain", "freshness_group", "release_year"]
        if column in columns
    ]
    geometry = sorted(
        column
        for column in columns
        if column.startswith("feature_") and "agop_" not in column
    )
    agop = sorted(column for column in columns if column.startswith("feature_agop_"))
    response = sorted(column for column in columns if column.startswith("early_"))
    return {"B": base, "M": metadata, "L": geometry, "G": agop, "R": response}


def main() -> None:
    args = parse_args()
    stage_root = Path(args.stage_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "pml_path_prediction_dataset.csv"
    manifest_path = output_dir / "prediction_dataset_manifest.json"
    if dataset_path.exists() and manifest_path.exists() and not args.overwrite:
        print(manifest_path.read_text(encoding="utf-8"))
        return
    runs = discover_runs(stage_root)
    thresholds = compute_null_thresholds(runs, args.random_method, args.null_quantile)
    rows: List[Dict[str, Any]] = []
    for method, layer, layer_dir in runs:
        print(f"[dataset] method={method} layer={layer} dir={layer_dir}")
        rows.extend(
            build_run_rows(
                method,
                layer,
                layer_dir,
                thresholds,
                args.early_alpha,
                args.medium_alpha,
            )
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["record_id", "method", "layer"]).reset_index(drop=True)
    atomic_write_csv(dataset_path, frame)
    groups = feature_groups(frame.columns)
    manifest = {
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "dataset_path": str(dataset_path),
        "random_method": args.random_method,
        "null_quantile": args.null_quantile,
        "thresholds": thresholds,
        "early_alpha": args.early_alpha,
        "medium_alpha": args.medium_alpha,
        "n_runs": len(runs),
        "n_rows": int(len(frame)),
        "n_records": int(frame["record_id"].nunique()),
        "methods": sorted(frame["method"].dropna().unique().tolist()),
        "layers": sorted(int(value) for value in frame["layer"].dropna().unique()),
        "datasets": sorted(str(value) for value in frame["dataset"].dropna().unique()),
        "domains": sorted(str(value) for value in frame["domain"].dropna().unique()),
        "feature_groups": groups,
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
