from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_STAGE_ROOT = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2_bidirectional_pilot_500/qwen3_1_7b"
)
DEFAULT_OUTPUT_DIR = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2_bidirectional_pilot_500/qwen3_1_7b/selector_dataset"
)
DEFAULT_EARLY_ALPHAS = [0.01, 0.02, 0.05, 0.1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build path-level and alpha-level datasets for lightweight PML strength selectors."
    )
    parser.add_argument("--stage-root", default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--tau-source", default="global_random_q95")
    parser.add_argument("--early-alphas", type=float, nargs="*", default=DEFAULT_EARLY_ALPHAS)
    parser.add_argument("--random-method-name", default="random")
    parser.add_argument("--lambda-neighbor", type=float, default=1.0)
    parser.add_argument("--lambda-capability", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def layer_from_dir(path: Path) -> int:
    value = path.name.replace("layer_", "").replace("neg", "-")
    return int(value)


def method_layer_dirs(stage_root: Path) -> List[tuple[str, int, Path]]:
    out: List[tuple[str, int, Path]] = []
    for method_dir in sorted(path for path in stage_root.iterdir() if path.is_dir()):
        if method_dir.name in {"threshold_reanalysis", "selector_dataset"}:
            continue
        for layer_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            if (layer_dir / "per_record.csv").exists() and (layer_dir / "per_eval.csv").exists():
                out.append((method_dir.name, layer_from_dir(layer_dir), layer_dir))
    return out


def to_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def quantile(values: Sequence[float], q: float) -> float | None:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else None


def median_abs_plus_2mad(values: Sequence[float]) -> float | None:
    vals = [abs(v) for v in values if not math.isnan(v)]
    if not vals:
        return None
    med = median(vals)
    deviations = [abs(v - med) for v in vals]
    return med + 2.0 * median(deviations)


def compute_global_random_tau(stage_root: Path, random_method_name: str) -> Dict[str, float]:
    values: List[float] = []
    for method, _, layer_dir in method_layer_dirs(stage_root):
        if method != random_method_name:
            continue
        for row in read_csv(layer_dir / "per_record.csv"):
            if to_float(row["alpha"]) == 0:
                continue
            for metric in ["target", "neighbor", "capability"]:
                values.append(abs(to_float(row[f"{metric}_mean_delta_margin"])))
    return {
        "global_random_q95": float(quantile(values, 0.95) or 0.0),
        "global_random_median_plus_2mad": float(median_abs_plus_2mad(values) or 0.0),
    }


def resolve_tau(stage_root: Path, tau: float | None, tau_source: str, random_method_name: str) -> tuple[float, Dict[str, float]]:
    random_tau = compute_global_random_tau(stage_root, random_method_name)
    if tau is not None:
        return tau, random_tau
    if tau_source not in random_tau:
        raise ValueError(f"Unsupported tau source {tau_source}. Available: {sorted(random_tau)}")
    return random_tau[tau_source], random_tau


def summarize(values: Sequence[float]) -> Dict[str, float | None]:
    return {
        "mean": mean(values),
        "median": quantile(values, 0.5),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def base_margin_features(per_eval_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_record_type: Dict[tuple[str, str], List[float]] = defaultdict(list)
    metadata_by_record: Dict[str, Dict[str, Any]] = {}
    for row in per_eval_rows:
        if to_float(row["alpha"]) != 0:
            continue
        record_id = row["record_id"]
        eval_type = row["eval_type"]
        by_record_type[(record_id, eval_type)].append(to_float(row["base_margin"]))
        metadata_by_record.setdefault(
            record_id,
            {
                "dataset": row.get("dataset"),
                "domain": row.get("domain"),
                "freshness_group": row.get("freshness_group"),
                "release_year": row.get("release_year"),
            },
        )

    out: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for record_id, metadata in metadata_by_record.items():
        out[record_id].update(metadata)
    for (record_id, eval_type), values in by_record_type.items():
        stats = summarize(values)
        prefix = f"{eval_type}_base_margin"
        out[record_id][f"{prefix}_mean"] = stats["mean"]
        out[record_id][f"{prefix}_median"] = stats["median"]
        out[record_id][f"{prefix}_min"] = stats["min"]
        out[record_id][f"{prefix}_max"] = stats["max"]
        out[record_id][f"{eval_type}_base_positive_rate"] = mean([v > 0 for v in values])
    return dict(out)


def metric_delta(row: Dict[str, Any], metric: str) -> float:
    return to_float(row[f"{metric}_mean_delta_margin"])


def signed_target_success(alpha: float, target_delta: float, tau: float) -> bool:
    if alpha < 0:
        return target_delta <= -tau
    if alpha > 0:
        return target_delta >= tau
    return False


def damage(delta: float, tau: float) -> bool:
    return delta <= -tau


def utility(alpha: float, target_delta: float, neighbor_delta: float, capability_delta: float, lambda_neighbor: float, lambda_capability: float) -> float:
    if alpha < 0:
        target_gain = -target_delta
    elif alpha > 0:
        target_gain = target_delta
    else:
        target_gain = 0.0
    neighbor_penalty = max(0.0, -neighbor_delta)
    capability_penalty = max(0.0, -capability_delta)
    return target_gain - lambda_neighbor * neighbor_penalty - lambda_capability * capability_penalty


def early_response_features(rows: Sequence[Dict[str, Any]], early_alphas: Sequence[float]) -> Dict[str, Any]:
    by_alpha = {round(to_float(row["alpha"]), 8): row for row in rows}
    out: Dict[str, Any] = {}
    for alpha_abs in early_alphas:
        for sign_name, signed_alpha in [("neg", -abs(alpha_abs)), ("pos", abs(alpha_abs))]:
            row = by_alpha.get(round(signed_alpha, 8))
            for metric in ["target", "neighbor", "capability"]:
                out[f"early_{metric}_delta_{sign_name}_{alpha_abs:g}"] = (
                    metric_delta(row, metric) if row is not None else None
                )
    for metric in ["target", "neighbor", "capability"]:
        slopes = []
        for alpha_abs in early_alphas:
            neg = out.get(f"early_{metric}_delta_neg_{alpha_abs:g}")
            pos = out.get(f"early_{metric}_delta_pos_{alpha_abs:g}")
            if neg is not None and not math.isnan(float(neg)):
                slopes.append(float(neg) / (-abs(alpha_abs)))
            if pos is not None and not math.isnan(float(pos)):
                slopes.append(float(pos) / abs(alpha_abs))
        out[f"early_{metric}_slope_mean"] = mean(slopes)
    target_signs = []
    for alpha_abs in early_alphas:
        neg = out.get(f"early_target_delta_neg_{alpha_abs:g}")
        pos = out.get(f"early_target_delta_pos_{alpha_abs:g}")
        if neg is not None and not math.isnan(float(neg)):
            target_signs.append(float(neg) <= 0)
        if pos is not None and not math.isnan(float(pos)):
            target_signs.append(float(pos) >= 0)
    out["early_target_sign_consistency"] = mean(target_signs)
    out["early_damage_flag"] = any(
        (value is not None and not math.isnan(float(value)) and float(value) < 0)
        for key, value in out.items()
        if key.startswith("early_neighbor_delta_") or key.startswith("early_capability_delta_")
    )
    return out


def path_labels(rows: Sequence[Dict[str, Any]], tau: float, lambda_neighbor: float, lambda_capability: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    best_supp = None
    best_enh = None
    supp_onset = None
    enh_onset = None
    neg_damage_onset = None
    pos_damage_onset = None
    clean_supp = False
    clean_enh = False
    for row in sorted(rows, key=lambda r: abs(to_float(r["alpha"]))):
        alpha = to_float(row["alpha"])
        target = metric_delta(row, "target")
        neighbor = metric_delta(row, "neighbor")
        capability = metric_delta(row, "capability")
        is_target = signed_target_success(alpha, target, tau)
        is_damage = damage(neighbor, tau) or damage(capability, tau)
        is_clean = is_target and not is_damage
        row_utility = utility(alpha, target, neighbor, capability, lambda_neighbor, lambda_capability)
        if alpha < 0:
            if is_target and supp_onset is None:
                supp_onset = alpha
            if is_damage and neg_damage_onset is None:
                neg_damage_onset = alpha
            clean_supp = clean_supp or is_clean
            if best_supp is None or row_utility > best_supp["utility"]:
                best_supp = {"alpha": alpha, "utility": row_utility, "is_clean": is_clean}
        elif alpha > 0:
            if is_target and enh_onset is None:
                enh_onset = alpha
            if is_damage and pos_damage_onset is None:
                pos_damage_onset = alpha
            clean_enh = clean_enh or is_clean
            if best_enh is None or row_utility > best_enh["utility"]:
                best_enh = {"alpha": alpha, "utility": row_utility, "is_clean": is_clean}
    out.update(
        {
            "suppression_onset_alpha": supp_onset,
            "enhancement_onset_alpha": enh_onset,
            "negative_damage_onset_alpha": neg_damage_onset,
            "positive_damage_onset_alpha": pos_damage_onset,
            "clean_suppression_window_exists": clean_supp,
            "clean_enhancement_window_exists": clean_enh,
            "bidirectional_clean_control": clean_supp and clean_enh,
            "best_suppression_alpha": best_supp["alpha"] if best_supp else None,
            "best_suppression_utility": best_supp["utility"] if best_supp else None,
            "best_suppression_is_clean": best_supp["is_clean"] if best_supp else None,
            "best_enhancement_alpha": best_enh["alpha"] if best_enh else None,
            "best_enhancement_utility": best_enh["utility"] if best_enh else None,
            "best_enhancement_is_clean": best_enh["is_clean"] if best_enh else None,
        }
    )
    if clean_supp and clean_enh:
        out["path_type"] = "bidirectional_clean"
    elif clean_supp or clean_enh:
        out["path_type"] = "single_side_clean"
    elif supp_onset is not None or enh_onset is not None:
        out["path_type"] = "target_with_damage_or_unclean"
    elif neg_damage_onset is not None or pos_damage_onset is not None:
        out["path_type"] = "damage_only"
    else:
        out["path_type"] = "no_effect"
    return out


def build_for_path(
    method: str,
    layer: int,
    layer_dir: Path,
    tau: float,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_record = read_csv(layer_dir / "per_record.csv")
    per_eval = read_csv(layer_dir / "per_eval.csv")
    base_features = base_margin_features(per_eval)
    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_record:
        by_record[row["record_id"]].append(row)

    path_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []
    for record_id, rows in sorted(by_record.items()):
        first = rows[0]
        feature_summary = {
            key: value
            for key, value in first.items()
            if key.startswith("feature_")
        }
        common: Dict[str, Any] = {
            "record_id": record_id,
            "method": method,
            "layer": layer,
            "tau": tau,
            **{k: first.get(k) for k in ["dataset", "domain", "freshness_group", "release_year"]},
            **base_features.get(record_id, {}),
            **feature_summary,
            **early_response_features(rows, args.early_alphas),
        }
        common["target_neighbor_base_margin_gap"] = (
            to_float(common.get("target_base_margin_mean")) - to_float(common.get("neighbor_base_margin_mean"))
        )
        common["target_capability_base_margin_gap"] = (
            to_float(common.get("target_base_margin_mean")) - to_float(common.get("capability_base_margin_mean"))
        )
        labels = path_labels(rows, tau, args.lambda_neighbor, args.lambda_capability)
        path_rows.append({**common, **labels})
        for row in rows:
            alpha = to_float(row["alpha"])
            target = metric_delta(row, "target")
            neighbor = metric_delta(row, "neighbor")
            capability = metric_delta(row, "capability")
            is_target = signed_target_success(alpha, target, tau)
            neighbor_damage = damage(neighbor, tau)
            capability_damage = damage(capability, tau)
            row_utility = utility(alpha, target, neighbor, capability, args.lambda_neighbor, args.lambda_capability)
            alpha_rows.append(
                {
                    **common,
                    "alpha": alpha,
                    "abs_alpha": abs(alpha),
                    "alpha_sign": -1 if alpha < 0 else (1 if alpha > 0 else 0),
                    "alpha_squared": alpha * alpha,
                    "target_delta": target,
                    "neighbor_delta": neighbor,
                    "capability_delta": capability,
                    "target_success": is_target,
                    "neighbor_damage": neighbor_damage,
                    "capability_damage": capability_damage,
                    "any_damage": neighbor_damage or capability_damage,
                    "is_clean": is_target and not neighbor_damage and not capability_damage,
                    "utility": row_utility,
                    "direction": "suppression" if alpha < 0 else ("enhancement" if alpha > 0 else "zero"),
                }
            )
    return path_rows, alpha_rows


def main() -> None:
    args = parse_args()
    stage_root = Path(args.stage_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        "path_level_selector_dataset.csv",
        "alpha_level_selector_dataset.csv",
        "selector_dataset_manifest.json",
    ]
    if any((out_dir / name).exists() for name in outputs) and not args.overwrite:
        raise FileExistsError(f"Selector dataset already exists in {out_dir}. Use --overwrite.")
    if args.overwrite:
        for name in outputs:
            (out_dir / name).unlink(missing_ok=True)

    tau, tau_info = resolve_tau(stage_root, args.tau, args.tau_source, args.random_method_name)
    path_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []
    method_layer_paths = method_layer_dirs(stage_root)
    for method, layer, layer_dir in method_layer_paths:
        path_part, alpha_part = build_for_path(method, layer, layer_dir, tau, args)
        path_rows.extend(path_part)
        alpha_rows.extend(alpha_part)

    write_csv(out_dir / "path_level_selector_dataset.csv", path_rows)
    write_csv(out_dir / "alpha_level_selector_dataset.csv", alpha_rows)
    manifest = {
        "stage_root": str(stage_root),
        "output_dir": str(out_dir),
        "tau": tau,
        "tau_source": args.tau_source,
        "tau_info": tau_info,
        "early_alphas": args.early_alphas,
        "lambda_neighbor": args.lambda_neighbor,
        "lambda_capability": args.lambda_capability,
        "n_method_layer_paths": len(method_layer_paths),
        "n_path_rows": len(path_rows),
        "n_alpha_rows": len(alpha_rows),
    }
    (out_dir / "selector_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
