from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-record adaptive alpha plans from gradient-response slopes."
    )
    parser.add_argument("--gradient-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tau", type=float, default=0.1522890289624531)
    parser.add_argument("--min-alpha", type=float, default=0.01)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument("--max-points-per-side", type=int, default=6)
    parser.add_argument(
        "--fallback-abs-alphas",
        nargs="*",
        type=float,
        default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--multipliers",
        nargs="*",
        type=float,
        default=[0.5, 0.8, 1.0, 1.25, 1.6],
        help="Multipliers around predicted onset used to form per-record grids.",
    )
    return parser.parse_args()


def method_layer_from_path(path: Path) -> tuple[str, str]:
    return path.parts[-3], path.parts[-2]


def finite_positive(value: Any, min_alpha: float, max_alpha: float) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    x = abs(x)
    if x < min_alpha or x > max_alpha:
        return None
    return x


def compact_abs_grid(values: Sequence[float], args: argparse.Namespace) -> List[float]:
    clipped = [
        min(max(abs(float(value)), args.min_alpha), args.max_alpha)
        for value in values
        if math.isfinite(float(value))
    ]
    rounded = sorted({round(value, 4) for value in clipped})
    if len(rounded) <= args.max_points_per_side:
        return rounded
    idxs = sorted({round(i * (len(rounded) - 1) / (args.max_points_per_side - 1)) for i in range(args.max_points_per_side)})
    return [rounded[int(idx)] for idx in idxs]


def onset_candidates(slopes: Sequence[float], args: argparse.Namespace) -> List[float]:
    candidates: List[float] = []
    for slope in slopes:
        if slope == 0 or not math.isfinite(float(slope)):
            continue
        onset = finite_positive(args.tau / abs(float(slope)), args.min_alpha, args.max_alpha)
        if onset is None:
            continue
        for multiplier in args.multipliers:
            candidates.append(onset * multiplier)
    return candidates


def plan_for_record(group: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    target = group[group["eval_type"] == "target"]
    neighbor = group[group["eval_type"] == "neighbor"]
    capability = group[group["eval_type"] == "capability"]

    target_slopes = [float(x) for x in target["directional_slope"].dropna()]
    side_slopes = [float(x) for x in neighbor["directional_slope"].dropna()] + [
        float(x) for x in capability["directional_slope"].dropna()
    ]
    candidates = list(args.fallback_abs_alphas)
    candidates.extend(onset_candidates(target_slopes, args))
    candidates.extend(onset_candidates(side_slopes, args))
    abs_grid = compact_abs_grid(candidates, args)
    alphas = [-x for x in reversed(abs_grid) if x > 0] + [0.0] + [x for x in abs_grid if x > 0]
    return {
        "alphas": alphas,
        "abs_grid": abs_grid,
        "n_alpha": len(alphas),
        "target_slope_mean": float(pd.Series(target_slopes).mean()) if target_slopes else None,
        "neighbor_slope_mean": float(neighbor["directional_slope"].mean()) if len(neighbor) else None,
        "capability_slope_mean": float(capability["directional_slope"].mean()) if len(capability) else None,
    }


def main() -> None:
    args = parse_args()
    gradient_root = Path(args.gradient_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for csv_path in sorted(gradient_root.glob("*/*/per_eval_gradient.csv")):
        method, layer_name = method_layer_from_path(csv_path)
        df = pd.read_csv(csv_path)
        group_dir = out_dir / method / layer_name
        group_dir.mkdir(parents=True, exist_ok=True)
        plan_rows: List[Dict[str, Any]] = []
        for record_id, group in df.groupby("record_id"):
            plan = plan_for_record(group, args)
            plan_rows.append(
                {
                    "record_id": str(record_id),
                    "method": method,
                    "layer": layer_name,
                    **plan,
                }
            )
        plan_path = group_dir / "adaptive_alpha_plan.jsonl"
        with plan_path.open("w", encoding="utf-8") as f:
            for row in plan_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "method": method,
            "layer": layer_name,
            "n_records": len(plan_rows),
            "plan_path": str(plan_path),
            "mean_n_alpha": float(pd.Series([row["n_alpha"] for row in plan_rows]).mean()) if plan_rows else 0.0,
            "min_n_alpha": min([row["n_alpha"] for row in plan_rows]) if plan_rows else 0,
            "max_n_alpha": max([row["n_alpha"] for row in plan_rows]) if plan_rows else 0,
        }
        summaries.append(summary)

    if not summaries:
        raise ValueError(f"No per_eval_gradient.csv files found under {gradient_root}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "adaptive_alpha_plan_summary.csv", index=False)
    (out_dir / "adaptive_alpha_plan_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Adaptive Alpha Plan",
        "",
        f"- gradient_root: `{gradient_root}`",
        f"- tau: `{args.tau}`",
        f"- max_points_per_side: `{args.max_points_per_side}`",
        "",
        "| method | layer | records | mean n_alpha | min | max | plan |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        report.append(
            f"| `{row['method']}` | `{row['layer']}` | {row['n_records']} | {row['mean_n_alpha']:.2f} | {row['min_n_alpha']} | {row['max_n_alpha']} | `{row['plan_path']}` |"
        )
    (out_dir / "ADAPTIVE_ALPHA_PLAN.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"gradient_root": str(gradient_root), "output_dir": str(out_dir), "n_groups": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
