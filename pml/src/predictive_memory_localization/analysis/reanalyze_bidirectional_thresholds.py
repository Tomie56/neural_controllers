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
    "stage2_bidirectional_pilot_500/qwen3_1_7b/threshold_reanalysis"
)
DEFAULT_TAUS = [0.05, 0.1, 0.2, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline reanalysis for bidirectional PML curves: alpha curves, fixed tau summaries, "
            "and random/null calibrated tau without rerunning model scoring."
        )
    )
    parser.add_argument("--stage-root", default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixed-taus", type=float, nargs="*", default=DEFAULT_TAUS)
    parser.add_argument("--random-method-name", default="random")
    parser.add_argument("--target-probe-success-rate", type=float, default=0.5)
    parser.add_argument("--neighbor-damage-rate-threshold", type=float, default=0.5)
    parser.add_argument("--capability-damage-rate-threshold", type=float, default=0.2)
    parser.add_argument("--tiny-alpha-max", type=float, default=0.02)
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
    name = path.name
    if not name.startswith("layer_"):
        raise ValueError(f"Unexpected layer directory name: {name}")
    value = name[len("layer_") :].replace("neg", "-")
    return int(value)


def method_layer_dirs(stage_root: Path) -> List[tuple[str, int, Path]]:
    out: List[tuple[str, int, Path]] = []
    for method_dir in sorted(path for path in stage_root.iterdir() if path.is_dir()):
        if method_dir.name == "threshold_reanalysis":
            continue
        for layer_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            if (layer_dir / "per_record.csv").exists() and (layer_dir / "per_eval.csv").exists():
                out.append((method_dir.name, layer_from_dir(layer_dir), layer_dir))
    return out


def to_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


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


def mad_tau(values: Sequence[float]) -> float | None:
    vals = [abs(v) for v in values if not math.isnan(v)]
    if not vals:
        return None
    med = median(vals)
    deviations = [abs(v - med) for v in vals]
    return med + 2.0 * median(deviations)


def rate(values: Iterable[bool]) -> float:
    vals = list(values)
    return sum(bool(v) for v in vals) / len(vals) if vals else 0.0


def metric_value(row: Dict[str, Any], metric: str) -> float:
    return to_float(row[f"{metric}_mean_delta_margin"])


def target_success(row: Dict[str, Any], tau: float) -> bool:
    alpha = to_float(row["alpha"])
    target_delta = metric_value(row, "target")
    if alpha < 0:
        return target_delta <= -tau
    if alpha > 0:
        return target_delta >= tau
    return False


def neighbor_damaged(row: Dict[str, Any], tau: float) -> bool:
    return metric_value(row, "neighbor") <= -tau


def capability_damaged(row: Dict[str, Any], tau: float) -> bool:
    return metric_value(row, "capability") <= -tau


def first_onset(rows: Sequence[Dict[str, Any]], predicate, sign: str) -> float | None:
    candidates = []
    for row in rows:
        alpha = to_float(row["alpha"])
        if sign == "negative" and alpha >= 0:
            continue
        if sign == "positive" and alpha <= 0:
            continue
        if predicate(row):
            candidates.append(alpha)
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x))


def alpha_curve_rows(method: str, layer: int, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_alpha: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_alpha[to_float(row["alpha"])].append(row)
    out = []
    for alpha, alpha_rows in sorted(by_alpha.items()):
        record_ids = {row["record_id"] for row in alpha_rows}
        target = [metric_value(row, "target") for row in alpha_rows]
        neighbor = [metric_value(row, "neighbor") for row in alpha_rows]
        capability = [metric_value(row, "capability") for row in alpha_rows]
        out.append(
            {
                "method": method,
                "layer": layer,
                "alpha": alpha,
                "n_records": len(record_ids),
                "target_mean_delta": mean(target),
                "target_median_delta": quantile(target, 0.5),
                "target_p05_delta": quantile(target, 0.05),
                "target_p95_delta": quantile(target, 0.95),
                "neighbor_mean_delta": mean(neighbor),
                "neighbor_median_delta": quantile(neighbor, 0.5),
                "capability_mean_delta": mean(capability),
                "capability_median_delta": quantile(capability, 0.5),
                "target_signed_direction_rate": rate(
                    (v <= 0 if alpha < 0 else v >= 0) for v in target
                )
                if alpha != 0
                else None,
            }
        )
    return out


def tau_summary_rows(
    method: str,
    layer: int,
    rows: Sequence[Dict[str, Any]],
    taus: Sequence[tuple[str, float]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_record[row["record_id"]].append(row)

    summary_out: List[Dict[str, Any]] = []
    onset_out: List[Dict[str, Any]] = []
    for tau_label, tau in taus:
        path_rows = []
        for record_id, record_rows in sorted(by_record.items()):
            supp_onset = first_onset(record_rows, lambda r: target_success(r, tau), "negative")
            enh_onset = first_onset(record_rows, lambda r: target_success(r, tau), "positive")
            neg_neighbor_onset = first_onset(record_rows, lambda r: neighbor_damaged(r, tau), "negative")
            pos_neighbor_onset = first_onset(record_rows, lambda r: neighbor_damaged(r, tau), "positive")
            neg_capability_onset = first_onset(record_rows, lambda r: capability_damaged(r, tau), "negative")
            pos_capability_onset = first_onset(record_rows, lambda r: capability_damaged(r, tau), "positive")
            neg_damage_onset = first_onset(
                record_rows, lambda r: neighbor_damaged(r, tau) or capability_damaged(r, tau), "negative"
            )
            pos_damage_onset = first_onset(
                record_rows, lambda r: neighbor_damaged(r, tau) or capability_damaged(r, tau), "positive"
            )
            clean_supp = any(
                to_float(row["alpha"]) < 0
                and target_success(row, tau)
                and not neighbor_damaged(row, tau)
                and not capability_damaged(row, tau)
                for row in record_rows
            )
            clean_enh = any(
                to_float(row["alpha"]) > 0
                and target_success(row, tau)
                and not neighbor_damaged(row, tau)
                and not capability_damaged(row, tau)
                for row in record_rows
            )
            row = {
                "method": method,
                "layer": layer,
                "tau_label": tau_label,
                "tau": tau,
                "record_id": record_id,
                "suppression_onset_alpha": supp_onset,
                "enhancement_onset_alpha": enh_onset,
                "negative_neighbor_onset_alpha": neg_neighbor_onset,
                "positive_neighbor_onset_alpha": pos_neighbor_onset,
                "negative_capability_onset_alpha": neg_capability_onset,
                "positive_capability_onset_alpha": pos_capability_onset,
                "negative_damage_onset_alpha": neg_damage_onset,
                "positive_damage_onset_alpha": pos_damage_onset,
                "clean_suppression_window_exists": clean_supp,
                "clean_enhancement_window_exists": clean_enh,
                "bidirectional_clean_control": clean_supp and clean_enh,
            }
            path_rows.append(row)
            onset_out.append(row)

        summary_out.append(
            {
                "method": method,
                "layer": layer,
                "tau_label": tau_label,
                "tau": tau,
                "n_records": len(path_rows),
                "suppression_any_rate": rate(r["suppression_onset_alpha"] is not None for r in path_rows),
                "enhancement_any_rate": rate(r["enhancement_onset_alpha"] is not None for r in path_rows),
                "clean_suppression_rate": rate(r["clean_suppression_window_exists"] for r in path_rows),
                "clean_enhancement_rate": rate(r["clean_enhancement_window_exists"] for r in path_rows),
                "bidirectional_clean_rate": rate(r["bidirectional_clean_control"] for r in path_rows),
                "negative_neighbor_any_rate": rate(r["negative_neighbor_onset_alpha"] is not None for r in path_rows),
                "positive_neighbor_any_rate": rate(r["positive_neighbor_onset_alpha"] is not None for r in path_rows),
                "negative_capability_any_rate": rate(r["negative_capability_onset_alpha"] is not None for r in path_rows),
                "positive_capability_any_rate": rate(r["positive_capability_onset_alpha"] is not None for r in path_rows),
                "negative_damage_any_rate": rate(r["negative_damage_onset_alpha"] is not None for r in path_rows),
                "positive_damage_any_rate": rate(r["positive_damage_onset_alpha"] is not None for r in path_rows),
            }
        )
    return summary_out, onset_out


def collect_random_null(
    all_rows: Dict[tuple[str, int], List[Dict[str, Any]]],
    random_method_name: str,
    tiny_alpha_max: float,
) -> tuple[List[Dict[str, Any]], Dict[int, Dict[str, float]], Dict[str, float]]:
    values_by_layer_metric: Dict[tuple[int, str], List[float]] = defaultdict(list)
    tiny_values_by_layer_metric: Dict[tuple[int, str], List[float]] = defaultdict(list)
    global_values: List[float] = []
    tiny_global_values: List[float] = []

    for (method, layer), rows in all_rows.items():
        if method != random_method_name:
            continue
        for row in rows:
            alpha = abs(to_float(row["alpha"]))
            if alpha == 0:
                continue
            for metric in ["target", "neighbor", "capability"]:
                value = abs(metric_value(row, metric))
                values_by_layer_metric[(layer, metric)].append(value)
                global_values.append(value)
                if alpha <= tiny_alpha_max:
                    tiny_values_by_layer_metric[(layer, metric)].append(value)
                    tiny_global_values.append(value)

    rows_out: List[Dict[str, Any]] = []
    path_tau: Dict[int, Dict[str, float]] = defaultdict(dict)
    for (layer, metric), values in sorted(values_by_layer_metric.items()):
        tiny = tiny_values_by_layer_metric.get((layer, metric), [])
        q95 = quantile(values, 0.95)
        mad = mad_tau(values)
        tiny_q95 = quantile(tiny, 0.95)
        tiny_mad = mad_tau(tiny)
        rows_out.append(
            {
                "scope": "layer_metric",
                "layer": layer,
                "metric": metric,
                "n_random_values": len(values),
                "random_abs_delta_median": quantile(values, 0.5),
                "random_abs_delta_q95": q95,
                "random_abs_delta_median_plus_2mad": mad,
                "n_tiny_alpha_values": len(tiny),
                "tiny_alpha_abs_delta_q95": tiny_q95,
                "tiny_alpha_abs_delta_median_plus_2mad": tiny_mad,
            }
        )
        if q95 is not None:
            path_tau[layer][metric] = q95

    global_q95 = quantile(global_values, 0.95)
    global_mad = mad_tau(global_values)
    tiny_global_q95 = quantile(tiny_global_values, 0.95)
    tiny_global_mad = mad_tau(tiny_global_values)
    rows_out.append(
        {
            "scope": "global",
            "layer": "all",
            "metric": "all",
            "n_random_values": len(global_values),
            "random_abs_delta_median": quantile(global_values, 0.5),
            "random_abs_delta_q95": global_q95,
            "random_abs_delta_median_plus_2mad": global_mad,
            "n_tiny_alpha_values": len(tiny_global_values),
            "tiny_alpha_abs_delta_q95": tiny_global_q95,
            "tiny_alpha_abs_delta_median_plus_2mad": tiny_global_mad,
        }
    )
    global_tau = {
        "random_abs_delta_q95": global_q95 or 0.0,
        "random_abs_delta_median_plus_2mad": global_mad or 0.0,
        "tiny_alpha_abs_delta_q95": tiny_global_q95 or 0.0,
        "tiny_alpha_abs_delta_median_plus_2mad": tiny_global_mad or 0.0,
    }
    return rows_out, path_tau, global_tau


def write_report(path: Path, args: argparse.Namespace, summary_rows: Sequence[Dict[str, Any]], null_rows: Sequence[Dict[str, Any]]) -> None:
    fixed_main = [
        row for row in summary_rows
        if row["tau_label"] in {"fixed_0.1", "fixed_0.2", "fixed_1.0"}
    ]
    lines = [
        "# Bidirectional Threshold Reanalysis",
        "",
        f"Stage root: `{args.stage_root}`",
        f"Output dir: `{args.output_dir}`",
        "",
        "## Why This Exists",
        "",
        "`method_summary.json` used a fixed strict `tau=1.0`, which is too strict for this pilot.",
        "This reanalysis keeps the observed alpha scan fixed and only changes the offline event threshold.",
        "",
        "Definitions:",
        "",
        "- `alpha`: intervention strength used during model scoring.",
        "- `tau`: minimum contrastive margin delta required to count success or damage.",
        "- `delta_margin = margin(alpha) - margin(alpha=0)`.",
        "- suppression: `alpha < 0` and target delta <= `-tau`.",
        "- enhancement: `alpha > 0` and target delta >= `+tau`.",
        "- neighbor/capability damage: corresponding delta <= `-tau`.",
        "",
        "## Random/Null Calibrated Tau",
        "",
        "| Scope | Layer | Metric | q95 abs random delta | median+2MAD | tiny-alpha q95 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in null_rows:
        if row["scope"] == "global" or row["metric"] == "target":
            lines.append(
                "| {scope} | {layer} | {metric} | {q95:.6g} | {mad:.6g} | {tiny:.6g} |".format(
                    scope=row["scope"],
                    layer=row["layer"],
                    metric=row["metric"],
                    q95=float(row.get("random_abs_delta_q95") or 0.0),
                    mad=float(row.get("random_abs_delta_median_plus_2mad") or 0.0),
                    tiny=float(row.get("tiny_alpha_abs_delta_q95") or 0.0),
                )
            )
    lines.extend(
        [
            "",
            "## Fixed Tau Summary",
            "",
            "| Tau | Method | Layer | Supp any | Enh any | Clean supp | Clean enh | Neg damage | Pos damage |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fixed_main:
        lines.append(
            "| {tau} | `{method}` | {layer} | {supp:.3f} | {enh:.3f} | {cs:.3f} | {ce:.3f} | {nd:.3f} | {pd:.3f} |".format(
                tau=row["tau"],
                method=row["method"],
                layer=row["layer"],
                supp=float(row["suppression_any_rate"]),
                enh=float(row["enhancement_any_rate"]),
                cs=float(row["clean_suppression_rate"]),
                ce=float(row["clean_enhancement_rate"]),
                nd=float(row["negative_damage_any_rate"]),
                pd=float(row["positive_damage_any_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `method_layer_alpha_curve.csv`",
            "- `calibrated_tau_table.csv`",
            "- `method_layer_tau_summary.csv`",
            "- `record_onsets_by_tau.csv`",
            "- `threshold_reanalysis_config.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage_root = Path(args.stage_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        "method_layer_alpha_curve.csv",
        "calibrated_tau_table.csv",
        "method_layer_tau_summary.csv",
        "record_onsets_by_tau.csv",
        "THRESHOLD_REANALYSIS.md",
        "threshold_reanalysis_config.json",
    ]
    if any((out_dir / name).exists() for name in output_files) and not args.overwrite:
        raise FileExistsError(f"Output exists in {out_dir}. Use --overwrite.")
    if args.overwrite:
        for name in output_files:
            (out_dir / name).unlink(missing_ok=True)

    all_rows: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
    for method, layer, layer_dir in method_layer_dirs(stage_root):
        all_rows[(method, layer)] = read_csv(layer_dir / "per_record.csv")

    null_rows, path_tau, global_tau = collect_random_null(
        all_rows, args.random_method_name, args.tiny_alpha_max
    )

    alpha_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    onset_rows: List[Dict[str, Any]] = []
    fixed_taus = [(f"fixed_{tau:g}", tau) for tau in args.fixed_taus]

    for (method, layer), rows in sorted(all_rows.items()):
        alpha_rows.extend(alpha_curve_rows(method, layer, rows))
        tau_specs = list(fixed_taus)
        if global_tau["random_abs_delta_q95"] > 0:
            tau_specs.append(("global_random_q95", global_tau["random_abs_delta_q95"]))
        if global_tau["random_abs_delta_median_plus_2mad"] > 0:
            tau_specs.append(("global_random_median_plus_2mad", global_tau["random_abs_delta_median_plus_2mad"]))
        target_path_tau = path_tau.get(layer, {}).get("target")
        if target_path_tau:
            tau_specs.append((f"layer_{layer}_target_random_q95", target_path_tau))
        path_summary, path_onsets = tau_summary_rows(method, layer, rows, tau_specs)
        summary_rows.extend(path_summary)
        onset_rows.extend(path_onsets)

    config = {
        "stage_root": str(stage_root),
        "output_dir": str(out_dir),
        "fixed_taus": args.fixed_taus,
        "random_method_name": args.random_method_name,
        "tiny_alpha_max": args.tiny_alpha_max,
        "n_method_layer_paths": len(all_rows),
        "outputs": output_files,
    }
    (out_dir / "threshold_reanalysis_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "method_layer_alpha_curve.csv", alpha_rows)
    write_csv(out_dir / "calibrated_tau_table.csv", null_rows)
    write_csv(out_dir / "method_layer_tau_summary.csv", summary_rows)
    write_csv(out_dir / "record_onsets_by_tau.csv", onset_rows)
    write_report(out_dir / "THRESHOLD_REANALYSIS.md", args, summary_rows, null_rows)

    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"Wrote threshold reanalysis to {out_dir}")


if __name__ == "__main__":
    main()
