from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propose reduced alpha grids from gradient-response predicted onset statistics."
    )
    parser.add_argument("--gradient-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tau", type=float, default=0.1522890289624531)
    parser.add_argument("--min-alpha", type=float, default=0.01)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument(
        "--quantiles",
        nargs="*",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
        help="Onset quantiles used to form the reduced grid.",
    )
    parser.add_argument(
        "--anchors",
        nargs="*",
        type=float,
        default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
        help="Safety anchors always considered before clipping/dedup.",
    )
    parser.add_argument("--max-points-per-side", type=int, default=7)
    return parser.parse_args()


def method_layer_from_path(path: Path) -> tuple[str, str]:
    return path.parts[-3], path.parts[-2]


def finite_positive(values: Sequence[Any], min_alpha: float, max_alpha: float) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and min_alpha <= x <= max_alpha:
            out.append(x)
    return out


def rounded_grid(values: Sequence[float], max_points: int, min_alpha: float, max_alpha: float) -> List[float]:
    clipped = [min(max(abs(float(v)), min_alpha), max_alpha) for v in values if math.isfinite(float(v))]
    rounded = sorted({round(v, 4) for v in clipped})
    if len(rounded) <= max_points:
        return rounded
    # Keep endpoints and evenly spaced interior values by rank.
    idxs = sorted({round(i * (len(rounded) - 1) / (max_points - 1)) for i in range(max_points)})
    return [rounded[int(i)] for i in idxs]


def propose_for_group(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    target = df[df["eval_type"] == "target"].copy()
    neighbor = df[df["eval_type"] == "neighbor"].copy()
    capability = df[df["eval_type"] == "capability"].copy()

    target_slopes = target.groupby("record_id")["directional_slope"].mean()
    neighbor_slopes = neighbor.groupby("record_id")["directional_slope"].mean()
    capability_slopes = capability.groupby("record_id")["directional_slope"].mean()

    suppression_onsets = finite_positive(
        [args.tau / s for s in target_slopes if s > 0],
        args.min_alpha,
        args.max_alpha,
    )
    enhancement_onsets = finite_positive(
        [args.tau / s for s in target_slopes if s > 0],
        args.min_alpha,
        args.max_alpha,
    )
    negative_damage_onsets = finite_positive(
        [args.tau / s for s in list(neighbor_slopes[neighbor_slopes > 0]) + list(capability_slopes[capability_slopes > 0])],
        args.min_alpha,
        args.max_alpha,
    )
    positive_damage_onsets = finite_positive(
        [args.tau / abs(s) for s in list(neighbor_slopes[neighbor_slopes < 0]) + list(capability_slopes[capability_slopes < 0])],
        args.min_alpha,
        args.max_alpha,
    )

    def q(values: List[float]) -> List[float]:
        if not values:
            return []
        s = pd.Series(values)
        return [float(s.quantile(quantile)) for quantile in args.quantiles]

    negative_candidates = list(args.anchors) + q(suppression_onsets) + q(negative_damage_onsets)
    positive_candidates = list(args.anchors) + q(enhancement_onsets) + q(positive_damage_onsets)
    negative_abs_grid = rounded_grid(negative_candidates, args.max_points_per_side, args.min_alpha, args.max_alpha)
    positive_abs_grid = rounded_grid(positive_candidates, args.max_points_per_side, args.min_alpha, args.max_alpha)
    negative_grid = [-x for x in reversed(negative_abs_grid) if x > 0]
    positive_grid = [x for x in positive_abs_grid if x > 0]
    combined = negative_grid + [0.0] + positive_grid

    return {
        "n_records": int(df["record_id"].nunique()),
        "n_eval_rows": int(len(df)),
        "target_slope_positive_rate": float((target_slopes > 0).mean()) if len(target_slopes) else None,
        "target_slope_negative_rate": float((target_slopes < 0).mean()) if len(target_slopes) else None,
        "suppression_onset_n": len(suppression_onsets),
        "enhancement_onset_n": len(enhancement_onsets),
        "negative_damage_onset_n": len(negative_damage_onsets),
        "positive_damage_onset_n": len(positive_damage_onsets),
        "negative_abs_grid": negative_abs_grid,
        "positive_abs_grid": positive_abs_grid,
        "combined_alpha_grid": combined,
        "n_combined_alpha": len(combined),
    }


def main() -> None:
    args = parse_args()
    gradient_root = Path(args.gradient_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    shell_lines = [
        "# Suggested reduced alpha grids from gradient response.",
        "# Copy one ALPHAS value into the reduced-scan runner for a specific method/layer.",
        "",
    ]
    for path in sorted(gradient_root.glob("*/*/per_eval_gradient.csv")):
        method, layer_name = method_layer_from_path(path)
        df = pd.read_csv(path)
        summary = propose_for_group(df, args)
        summary.update({"method": method, "layer": layer_name, "source_csv": str(path)})
        rows.append(summary)
        alphas = " ".join(str(x) for x in summary["combined_alpha_grid"])
        shell_lines.append(f"# method={method} layer={layer_name} n_alpha={summary['n_combined_alpha']}")
        shell_lines.append(f"ALPHAS_{method}_{layer_name}=\"{alphas}\"")
        shell_lines.append("")

    if not rows:
        raise ValueError(f"No per_eval_gradient.csv files found under {gradient_root}")

    pd.DataFrame(rows).to_csv(out_dir / "reduced_alpha_grid_proposals.csv", index=False)
    (out_dir / "reduced_alpha_grid_proposals.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "reduced_alpha_grid_proposals.sh").write_text("\n".join(shell_lines), encoding="utf-8")
    report_lines = [
        "# Reduced Alpha Grid Proposals",
        "",
        f"- gradient_root: `{gradient_root}`",
        f"- tau: `{args.tau}`",
        f"- max_points_per_side: `{args.max_points_per_side}`",
        "",
        "Use these grids only after gradient-vs-dense validation is acceptable.",
        "",
        "| method | layer | n_records | n_alpha | alpha_grid |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        report_lines.append(
            f"| `{row['method']}` | `{row['layer']}` | {row['n_records']} | {row['n_combined_alpha']} | `{row['combined_alpha_grid']}` |"
        )
    (out_dir / "REDUCED_ALPHA_GRID_PROPOSALS.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"gradient_root": str(gradient_root), "output_dir": str(out_dir), "n_groups": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
