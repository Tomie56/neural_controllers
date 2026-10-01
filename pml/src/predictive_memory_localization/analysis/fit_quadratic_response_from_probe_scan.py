from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from predictive_memory_localization.analysis.compare_gradient_response_to_dense_curve import (
    dataframe_to_markdown,
    safe_corr,
    summarize_eval_metrics,
    summarize_path_labels,
)


DEFAULT_ALPHAS = [
    -1.0,
    -0.75,
    -0.5,
    -0.35,
    -0.25,
    -0.2,
    -0.15,
    -0.1,
    -0.08,
    -0.05,
    -0.03,
    -0.02,
    -0.01,
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.1,
    0.15,
    0.2,
    0.25,
    0.35,
    0.5,
    0.75,
    1.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit quadratic alpha-response curves from a small probe scan and optionally compare to dense curves."
    )
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dense-root", default=None)
    parser.add_argument("--predict-alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--tau", type=float, default=0.1522890289624531)
    parser.add_argument("--min-probe-points", type=int, default=3)
    return parser.parse_args()


def method_layer_from_path(path: Path) -> tuple[str, str]:
    return path.parts[-3], path.parts[-2]


def fit_coefficients(group: pd.DataFrame, min_probe_points: int) -> Dict[str, Any]:
    work = group[["alpha", "delta_margin", "base_margin"]].dropna().copy()
    work["alpha"] = work["alpha"].astype(float)
    work = work.drop_duplicates("alpha", keep="first")
    base_margin = float(work["base_margin"].iloc[0]) if len(work) else 0.0
    if len(work) < min_probe_points or work["alpha"].nunique() < min_probe_points:
        return {
            "fit_ok": False,
            "linear_coef": None,
            "quadratic_coef": None,
            "intercept": 0.0,
            "base_margin": base_margin,
            "n_probe_points": int(len(work)),
        }
    alpha = work["alpha"].to_numpy(dtype=float)
    delta = work["delta_margin"].to_numpy(dtype=float)
    # Force the fitted delta through 0 at alpha=0: delta ~= b*alpha + a*alpha^2.
    design = np.stack([alpha, alpha**2], axis=1)
    coef, *_ = np.linalg.lstsq(design, delta, rcond=None)
    pred = design @ coef
    rmse = float(np.sqrt(np.mean((pred - delta) ** 2)))
    return {
        "fit_ok": True,
        "linear_coef": float(coef[0]),
        "quadratic_coef": float(coef[1]),
        "intercept": 0.0,
        "base_margin": base_margin,
        "n_probe_points": int(len(work)),
        "probe_fit_rmse": rmse,
    }


def predict_group_rows(group: pd.DataFrame, coeff: Dict[str, Any], predict_alphas: List[float]) -> List[Dict[str, Any]]:
    first = group.iloc[0].to_dict()
    rows: List[Dict[str, Any]] = []
    linear = float(coeff["linear_coef"] or 0.0)
    quadratic = float(coeff["quadratic_coef"] or 0.0)
    base_margin = float(coeff["base_margin"])
    for alpha in predict_alphas:
        alpha = float(alpha)
        predicted_delta = linear * alpha + quadratic * alpha * alpha
        row = {
            "record_id": first["record_id"],
            "eval_id": first["eval_id"],
            "eval_type": first["eval_type"],
            "assignment_index": first["assignment_index"],
            "alpha": alpha,
            "method": first["method"],
            "layer": first["layer"],
            "base_margin": base_margin,
            "predicted_delta_margin": predicted_delta,
            "predicted_margin": base_margin + predicted_delta,
            "linear_coef": coeff["linear_coef"],
            "quadratic_coef": coeff["quadratic_coef"],
            "fit_ok": coeff["fit_ok"],
            "n_probe_points": coeff["n_probe_points"],
            "probe_fit_rmse": coeff.get("probe_fit_rmse"),
        }
        rows.append(row)
    return rows


def fit_probe_root(probe_root: Path, predict_alphas: List[float], min_probe_points: int) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for probe_path in sorted(probe_root.glob("*/*/per_eval.csv")):
        method, layer_name = method_layer_from_path(probe_path)
        df = pd.read_csv(probe_path)
        df["method"] = method
        df["layer"] = layer_name
        rows: List[Dict[str, Any]] = []
        group_keys = ["record_id", "eval_id", "eval_type", "assignment_index", "method", "layer"]
        for _, group in df.groupby(group_keys, dropna=False):
            coeff = fit_coefficients(group, min_probe_points)
            rows.extend(predict_group_rows(group, coeff, predict_alphas))
        if rows:
            parts.append(pd.DataFrame(rows))
    if not parts:
        raise ValueError(f"No probe per_eval.csv files found under {probe_root}")
    return pd.concat(parts, ignore_index=True)


def compare_to_dense(predicted: pd.DataFrame, dense_root: Path, tau: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dense_parts: List[pd.DataFrame] = []
    for (method, layer), pred_group in predicted.groupby(["method", "layer"], dropna=False):
        dense_path = dense_root / str(method) / str(layer) / "per_eval.csv"
        if not dense_path.exists():
            continue
        dense = pd.read_csv(dense_path)
        dense["method"] = method
        dense["layer"] = layer
        dense["alpha"] = dense["alpha"].astype(float)
        dense_parts.append(dense)
    if not dense_parts:
        raise ValueError(f"No matching dense per_eval.csv files found under {dense_root}")
    dense_all = pd.concat(dense_parts, ignore_index=True)
    predicted = predicted.copy()
    predicted["alpha"] = predicted["alpha"].astype(float)
    keys = ["record_id", "eval_id", "eval_type", "assignment_index", "alpha", "method", "layer"]
    merged = predicted[
        keys + ["base_margin", "predicted_delta_margin", "predicted_margin", "linear_coef", "quadratic_coef", "fit_ok"]
    ].merge(
        dense_all[keys + ["delta_margin", "margin"]],
        on=keys,
        how="inner",
    )
    metric_rows: List[Dict[str, Any]] = []
    for (method, layer), group in merged.groupby(["method", "layer"], dropna=False):
        metric_rows.append({"method": method, "layer": layer, **summarize_eval_metrics(group, tau)})
    metric_rows.append({"method": "__all__", "layer": "__all__", **summarize_eval_metrics(merged, tau)})
    metrics = pd.DataFrame(metric_rows)

    observed_paths = summarize_path_labels(merged, tau, "observed")
    predicted_paths = summarize_path_labels(merged, tau, "predicted")
    path_compare = observed_paths.merge(predicted_paths, on=["record_id", "method", "layer"], how="inner")
    label_rows: List[Dict[str, Any]] = []
    for label in [
        "target_any",
        "damage_any",
        "clean_suppression_window_exists",
        "clean_enhancement_window_exists",
        "bidirectional_clean_control",
    ]:
        observed = path_compare[f"observed_{label}"].astype(bool)
        pred = path_compare[f"predicted_{label}"].astype(bool)
        label_rows.append(
            {
                "label": label,
                "n_paths": int(len(path_compare)),
                "observed_rate": float(observed.mean()) if len(observed) else None,
                "predicted_rate": float(pred.mean()) if len(pred) else None,
                "accuracy": float((observed == pred).mean()) if len(path_compare) else None,
                "recall": float((observed & pred).sum() / observed.sum()) if observed.sum() else None,
                "false_positive_rate": float(((~observed) & pred).sum() / (~observed).sum()) if (~observed).sum() else None,
            }
        )
    return merged, metrics, pd.DataFrame(label_rows)


def main() -> None:
    args = parse_args()
    probe_root = Path(args.probe_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predicted = fit_probe_root(probe_root, [float(x) for x in args.predict_alphas], args.min_probe_points)
    predicted.to_csv(out_dir / "quadratic_predicted_per_eval.csv", index=False)
    fit_summary = predicted.groupby(["method", "layer"], dropna=False).agg(
        n_rows=("record_id", "size"),
        n_records=("record_id", "nunique"),
        fit_ok_rate=("fit_ok", "mean"),
        linear_coef_mean=("linear_coef", "mean"),
        quadratic_coef_mean=("quadratic_coef", "mean"),
    )
    fit_summary.reset_index().to_csv(out_dir / "quadratic_fit_summary.csv", index=False)

    report_lines = [
        "# Quadratic Response from Small Probe Scan",
        "",
        f"- probe_root: `{probe_root}`",
        f"- predicted rows: `{len(predicted)}`",
        f"- predicted alphas: `{sorted(float(x) for x in args.predict_alphas)}`",
        "",
        "## Fit Summary",
        "",
        dataframe_to_markdown(fit_summary.reset_index()),
        "",
    ]

    manifest: Dict[str, Any] = {
        "probe_root": str(probe_root),
        "output_dir": str(out_dir),
        "predict_alphas": [float(x) for x in args.predict_alphas],
        "tau": args.tau,
        "n_predicted_rows": int(len(predicted)),
    }

    if args.dense_root:
        merged, metrics, labels = compare_to_dense(predicted, Path(args.dense_root), args.tau)
        merged.to_csv(out_dir / "quadratic_vs_dense_per_eval.csv", index=False)
        metrics.to_csv(out_dir / "quadratic_vs_dense_metrics.csv", index=False)
        labels.to_csv(out_dir / "quadratic_vs_dense_label_metrics.csv", index=False)
        report_lines.extend(
            [
                "## Dense-Curve Metrics",
                "",
                dataframe_to_markdown(metrics),
                "",
                "## Path Label Metrics",
                "",
                dataframe_to_markdown(labels),
                "",
            ]
        )
        manifest["dense_root"] = str(Path(args.dense_root))
        manifest["matched_dense_rows"] = int(len(merged))

    (out_dir / "QUADRATIC_RESPONSE_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    (out_dir / "quadratic_response_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
