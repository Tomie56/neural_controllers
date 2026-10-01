from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DEFAULT_DENSE_ROOT = "/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare alpha=0 gradient response predictions with dense alpha curves.")
    parser.add_argument("--gradient-root", required=True)
    parser.add_argument("--dense-root", default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tau", type=float, default=0.1522890289624531)
    return parser.parse_args()


def method_layer_from_path(path: Path) -> tuple[str, str]:
    return path.parts[-3], path.parts[-2]


def safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float | None:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.corr(right, method=method)
    return None if pd.isna(value) else float(value)


def bool_rate(series: pd.Series) -> float:
    return float(series.astype(bool).mean()) if len(series) else 0.0


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_eval_metrics(df: pd.DataFrame, tau: float) -> Dict[str, Any]:
    err = df["predicted_delta_margin"] - df["delta_margin"]
    out: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_records": int(df["record_id"].nunique()),
        "n_eval_ids": int(df["eval_id"].nunique()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "pearson": safe_corr(df["predicted_delta_margin"], df["delta_margin"], "pearson"),
        "spearman": safe_corr(df["predicted_delta_margin"], df["delta_margin"], "spearman"),
        "sign_accuracy_nonzero": None,
        "effective_agreement": None,
        "observed_effective_rate": None,
        "predicted_effective_rate": None,
    }
    nonzero = df[df["alpha"].astype(float) != 0].copy()
    if not nonzero.empty:
        out["sign_accuracy_nonzero"] = float(
            (
                nonzero["predicted_delta_margin"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                == nonzero["delta_margin"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            ).mean()
        )
        observed_effective = nonzero["delta_margin"].abs() >= tau
        predicted_effective = nonzero["predicted_delta_margin"].abs() >= tau
        out["effective_agreement"] = float((observed_effective == predicted_effective).mean())
        out["observed_effective_rate"] = bool_rate(observed_effective)
        out["predicted_effective_rate"] = bool_rate(predicted_effective)
    return out


def first_onset(rows: pd.DataFrame, column: str, sign: str) -> float | None:
    if sign == "negative":
        rows = rows[rows["alpha"].astype(float) < 0]
    else:
        rows = rows[rows["alpha"].astype(float) > 0]
    rows = rows[rows[column].astype(bool)].copy()
    if rows.empty:
        return None
    rows["abs_alpha"] = rows["alpha"].astype(float).abs()
    return float(rows.sort_values("abs_alpha").iloc[0]["alpha"])


def summarize_path_labels(df: pd.DataFrame, tau: float, prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    delta_col = f"{prefix}_delta"
    work = df.copy()
    work[delta_col] = work["predicted_delta_margin"] if prefix == "predicted" else work["delta_margin"]
    for (record_id, method, layer), group in work.groupby(["record_id", "method", "layer"], dropna=False):
        record_rows: List[Dict[str, Any]] = []
        for alpha, alpha_group in group.groupby("alpha"):
            alpha = float(alpha)
            target = alpha_group[alpha_group["eval_type"] == "target"]
            neighbor = alpha_group[alpha_group["eval_type"] == "neighbor"]
            capability = alpha_group[alpha_group["eval_type"] == "capability"]
            if alpha < 0:
                target_success_rate = (target[delta_col] <= -tau).mean() if len(target) else 0.0
            elif alpha > 0:
                target_success_rate = (target[delta_col] >= tau).mean() if len(target) else 0.0
            else:
                target_success_rate = 0.0
            neighbor_damage_rate = (neighbor[delta_col] <= -tau).mean() if len(neighbor) else 0.0
            capability_damage_rate = (capability[delta_col] <= -tau).mean() if len(capability) else 0.0
            record_rows.append(
                {
                    "record_id": record_id,
                    "method": method,
                    "layer": layer,
                    "alpha": alpha,
                    "target_success": bool(target_success_rate >= 0.5 and alpha != 0),
                    "suppression_success": bool(alpha < 0 and target_success_rate >= 0.5),
                    "enhancement_success": bool(alpha > 0 and target_success_rate >= 0.5),
                    "neighbor_damaged": bool(neighbor_damage_rate >= 0.5),
                    "capability_damaged": bool(capability_damage_rate >= 0.2),
                }
            )
        rec = pd.DataFrame(record_rows)
        supp = first_onset(rec, "suppression_success", "negative")
        enh = first_onset(rec, "enhancement_success", "positive")
        rec["any_damage"] = rec["neighbor_damaged"] | rec["capability_damaged"]
        neg_damage = first_onset(rec, "any_damage", "negative")
        pos_damage = first_onset(rec, "any_damage", "positive")
        clean_supp = rec[
            (rec["alpha"] < 0) & rec["suppression_success"] & ~rec["neighbor_damaged"] & ~rec["capability_damaged"]
        ]
        clean_enh = rec[
            (rec["alpha"] > 0) & rec["enhancement_success"] & ~rec["neighbor_damaged"] & ~rec["capability_damaged"]
        ]
        rows.append(
            {
                "record_id": record_id,
                "method": method,
                "layer": layer,
                f"{prefix}_suppression_onset_alpha": supp,
                f"{prefix}_enhancement_onset_alpha": enh,
                f"{prefix}_negative_damage_onset_alpha": neg_damage,
                f"{prefix}_positive_damage_onset_alpha": pos_damage,
                f"{prefix}_clean_suppression_window_exists": bool(len(clean_supp)),
                f"{prefix}_clean_enhancement_window_exists": bool(len(clean_enh)),
                f"{prefix}_bidirectional_clean_control": bool(len(clean_supp) and len(clean_enh)),
                f"{prefix}_target_any": bool(supp is not None or enh is not None),
                f"{prefix}_damage_any": bool(neg_damage is not None or pos_damage is not None),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    gradient_root = Path(args.gradient_root)
    dense_root = Path(args.dense_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    merged_parts: List[pd.DataFrame] = []
    for pred_path in sorted(gradient_root.glob("*/*/per_eval_gradient.csv")):
        method, layer_name = method_layer_from_path(pred_path)
        dense_path = dense_root / method / layer_name / "per_eval.csv"
        if not dense_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        dense = pd.read_csv(dense_path)
        pred["method"] = method
        pred["layer"] = layer_name
        dense["method"] = method
        dense["layer"] = layer_name
        pred["alpha"] = pred["alpha"].astype(float)
        dense["alpha"] = dense["alpha"].astype(float)
        keys = ["record_id", "eval_id", "eval_type", "assignment_index", "alpha", "method", "layer"]
        merged = pred[
            keys + ["base_margin", "directional_slope", "predicted_delta_margin", "predicted_margin"]
        ].merge(
            dense[keys + ["delta_margin", "margin"]],
            on=keys,
            how="inner",
            suffixes=("_pred", "_observed"),
        )
        if merged.empty:
            continue
        merged_parts.append(merged)
        metric_rows.append({"method": method, "layer": layer_name, **summarize_eval_metrics(merged, args.tau)})

    if not merged_parts:
        raise ValueError(f"No matching gradient/dense per_eval files found under {gradient_root} and {dense_root}")

    merged_all = pd.concat(merged_parts, ignore_index=True)
    metric_rows.append({"method": "__all__", "layer": "__all__", **summarize_eval_metrics(merged_all, args.tau)})
    metrics = pd.DataFrame(metric_rows)

    observed_paths = summarize_path_labels(merged_all, args.tau, "observed")
    predicted_paths = summarize_path_labels(merged_all, args.tau, "predicted")
    path_keys = ["record_id", "method", "layer"]
    path_compare = observed_paths.merge(predicted_paths, on=path_keys, how="inner")
    label_rows: List[Dict[str, Any]] = []
    for label in [
        "target_any",
        "damage_any",
        "clean_suppression_window_exists",
        "clean_enhancement_window_exists",
        "bidirectional_clean_control",
    ]:
        observed = path_compare[f"observed_{label}"].astype(bool)
        predicted = path_compare[f"predicted_{label}"].astype(bool)
        label_rows.append(
            {
                "label": label,
                "n_paths": int(len(path_compare)),
                "observed_rate": bool_rate(observed),
                "predicted_rate": bool_rate(predicted),
                "accuracy": float((observed == predicted).mean()) if len(path_compare) else None,
                "true_positive_rate": float((observed & predicted).sum() / observed.sum()) if observed.sum() else None,
                "false_positive_rate": float(((~observed) & predicted).sum() / (~observed).sum()) if (~observed).sum() else None,
            }
        )

    merged_all.to_csv(out_dir / "gradient_vs_dense_per_eval.csv", index=False)
    metrics.to_csv(out_dir / "gradient_vs_dense_metrics.csv", index=False)
    path_compare.to_csv(out_dir / "gradient_vs_dense_path_labels.csv", index=False)
    labels = pd.DataFrame(label_rows)
    labels.to_csv(out_dir / "gradient_vs_dense_label_metrics.csv", index=False)

    report_lines = [
        "# Gradient Response vs Dense Alpha Curve",
        "",
        f"- gradient_root: `{gradient_root}`",
        f"- dense_root: `{dense_root}`",
        f"- tau: `{args.tau}`",
        f"- matched per-eval rows: `{len(merged_all)}`",
        f"- matched paths: `{len(path_compare)}`",
        "",
        "## Curve Metrics",
        "",
        dataframe_to_markdown(metrics),
        "",
        "## Path Label Metrics",
        "",
        dataframe_to_markdown(labels),
        "",
        "## Interpretation",
        "",
        "- Good evidence that the structural estimator can reduce alpha sweeps requires high sign/effective agreement and useful path-label accuracy.",
        "- If small-alpha rows work but large-alpha rows fail, use this estimator for candidate pruning and then scan a reduced alpha set.",
        "- If target labels work but damage labels fail, use gradient response only for target-onset screening.",
        "",
    ]
    (out_dir / "GRADIENT_RESPONSE_VALIDATION_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "gradient_root": str(gradient_root),
        "dense_root": str(dense_root),
        "output_dir": str(out_dir),
        "tau": args.tau,
        "matched_per_eval_rows": int(len(merged_all)),
        "matched_paths": int(len(path_compare)),
    }
    (out_dir / "gradient_vs_dense_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
