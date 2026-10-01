from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DEFAULT_DENSE_ROOT = "/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reduced alpha scans against full dense alpha curves.")
    parser.add_argument("--reduced-root", required=True)
    parser.add_argument("--dense-root", default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tau", type=float, default=0.1522890289624531)
    return parser.parse_args()


def method_layer_from_path(path: Path) -> tuple[str, str]:
    return path.parts[-3], path.parts[-2]


def bool_rate(series: pd.Series) -> float:
    return float(series.astype(bool).mean()) if len(series) else 0.0


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


def path_labels(df: pd.DataFrame, tau: float, prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (record_id, method, layer), group in df.groupby(["record_id", "method", "layer"], dropna=False):
        alpha_rows: List[Dict[str, Any]] = []
        for alpha, alpha_group in group.groupby("alpha"):
            alpha = float(alpha)
            target = alpha_group[alpha_group["eval_type"] == "target"]
            neighbor = alpha_group[alpha_group["eval_type"] == "neighbor"]
            capability = alpha_group[alpha_group["eval_type"] == "capability"]
            if alpha < 0:
                target_success_rate = (target["delta_margin"] <= -tau).mean() if len(target) else 0.0
            elif alpha > 0:
                target_success_rate = (target["delta_margin"] >= tau).mean() if len(target) else 0.0
            else:
                target_success_rate = 0.0
            neighbor_damage_rate = (neighbor["delta_margin"] <= -tau).mean() if len(neighbor) else 0.0
            capability_damage_rate = (capability["delta_margin"] <= -tau).mean() if len(capability) else 0.0
            alpha_rows.append(
                {
                    "record_id": record_id,
                    "method": method,
                    "layer": layer,
                    "alpha": alpha,
                    "suppression_success": bool(alpha < 0 and target_success_rate >= 0.5),
                    "enhancement_success": bool(alpha > 0 and target_success_rate >= 0.5),
                    "neighbor_damaged": bool(neighbor_damage_rate >= 0.5),
                    "capability_damaged": bool(capability_damage_rate >= 0.2),
                }
            )
        rec = pd.DataFrame(alpha_rows)
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
                f"{prefix}_target_any": bool(supp is not None or enh is not None),
                f"{prefix}_damage_any": bool(neg_damage is not None or pos_damage is not None),
                f"{prefix}_clean_suppression_window_exists": bool(len(clean_supp)),
                f"{prefix}_clean_enhancement_window_exists": bool(len(clean_enh)),
                f"{prefix}_bidirectional_clean_control": bool(len(clean_supp) and len(clean_enh)),
            }
        )
    return pd.DataFrame(rows)


def onset_abs_error(left: pd.Series, right: pd.Series) -> float | None:
    pairs = pd.DataFrame({"left": left, "right": right}).dropna()
    if pairs.empty:
        return None
    return float((pairs["left"].abs() - pairs["right"].abs()).abs().mean())


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(str(col) for col in df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, (list, tuple, dict)):
                values.append(str(value))
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reduced_root = Path(args.reduced_root)
    dense_root = Path(args.dense_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reduced_parts: List[pd.DataFrame] = []
    dense_parts: List[pd.DataFrame] = []
    group_rows: List[Dict[str, Any]] = []
    for reduced_path in sorted(reduced_root.glob("*/*/per_eval.csv")):
        method, layer_name = method_layer_from_path(reduced_path)
        dense_path = dense_root / method / layer_name / "per_eval.csv"
        if not dense_path.exists():
            continue
        reduced = pd.read_csv(reduced_path)
        dense = pd.read_csv(dense_path)
        reduced["method"] = method
        reduced["layer"] = layer_name
        dense["method"] = method
        dense["layer"] = layer_name
        reduced_parts.append(reduced)
        dense_parts.append(dense)
        group_rows.append(
            {
                "method": method,
                "layer": layer_name,
                "reduced_records": int(reduced["record_id"].nunique()),
                "dense_records": int(dense["record_id"].nunique()),
                "reduced_alphas": sorted(float(x) for x in reduced["alpha"].unique()),
                "n_reduced_alphas": int(reduced["alpha"].nunique()),
                "n_dense_alphas": int(dense["alpha"].nunique()),
            }
        )

    if not reduced_parts:
        raise ValueError(f"No matching reduced/dense per_eval.csv files found under {reduced_root} and {dense_root}")

    reduced_all = pd.concat(reduced_parts, ignore_index=True)
    dense_all = pd.concat(dense_parts, ignore_index=True)
    reduced_labels = path_labels(reduced_all, args.tau, "reduced")
    dense_labels = path_labels(dense_all, args.tau, "dense")
    keys = ["record_id", "method", "layer"]
    label_compare = dense_labels.merge(reduced_labels, on=keys, how="inner")

    label_metrics: List[Dict[str, Any]] = []
    for label in [
        "target_any",
        "damage_any",
        "clean_suppression_window_exists",
        "clean_enhancement_window_exists",
        "bidirectional_clean_control",
    ]:
        dense = label_compare[f"dense_{label}"].astype(bool)
        reduced = label_compare[f"reduced_{label}"].astype(bool)
        label_metrics.append(
            {
                "label": label,
                "n_paths": int(len(label_compare)),
                "dense_rate": bool_rate(dense),
                "reduced_rate": bool_rate(reduced),
                "accuracy": float((dense == reduced).mean()) if len(label_compare) else None,
                "recall": float((dense & reduced).sum() / dense.sum()) if dense.sum() else None,
                "false_positive_rate": float(((~dense) & reduced).sum() / (~dense).sum()) if (~dense).sum() else None,
            }
        )

    onset_metrics = pd.DataFrame(
        [
            {
                "onset": "suppression",
                "mean_abs_alpha_error": onset_abs_error(
                    label_compare["dense_suppression_onset_alpha"],
                    label_compare["reduced_suppression_onset_alpha"],
                ),
            },
            {
                "onset": "enhancement",
                "mean_abs_alpha_error": onset_abs_error(
                    label_compare["dense_enhancement_onset_alpha"],
                    label_compare["reduced_enhancement_onset_alpha"],
                ),
            },
            {
                "onset": "negative_damage",
                "mean_abs_alpha_error": onset_abs_error(
                    label_compare["dense_negative_damage_onset_alpha"],
                    label_compare["reduced_negative_damage_onset_alpha"],
                ),
            },
            {
                "onset": "positive_damage",
                "mean_abs_alpha_error": onset_abs_error(
                    label_compare["dense_positive_damage_onset_alpha"],
                    label_compare["reduced_positive_damage_onset_alpha"],
                ),
            },
        ]
    )
    group_summary = pd.DataFrame(group_rows)
    label_metrics_df = pd.DataFrame(label_metrics)

    group_summary.to_csv(out_dir / "reduced_scan_group_summary.csv", index=False)
    label_compare.to_csv(out_dir / "reduced_vs_dense_path_labels.csv", index=False)
    label_metrics_df.to_csv(out_dir / "reduced_vs_dense_label_metrics.csv", index=False)
    onset_metrics.to_csv(out_dir / "reduced_vs_dense_onset_metrics.csv", index=False)

    report = [
        "# Reduced Alpha Scan vs Dense Curve",
        "",
        f"- reduced_root: `{reduced_root}`",
        f"- dense_root: `{dense_root}`",
        f"- tau: `{args.tau}`",
        f"- matched paths: `{len(label_compare)}`",
        "",
        "## Groups",
        "",
        dataframe_to_markdown(group_summary),
        "",
        "## Label Metrics",
        "",
        dataframe_to_markdown(label_metrics_df),
        "",
        "## Onset Metrics",
        "",
        dataframe_to_markdown(onset_metrics),
        "",
    ]
    (out_dir / "REDUCED_SCAN_VALIDATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "reduced_root": str(reduced_root),
        "dense_root": str(dense_root),
        "output_dir": str(out_dir),
        "tau": args.tau,
        "matched_paths": int(len(label_compare)),
        "n_groups": len(group_rows),
    }
    (out_dir / "reduced_vs_dense_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
