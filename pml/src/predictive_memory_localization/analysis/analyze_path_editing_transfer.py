from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join activation path-geometry rows with ROME/superficial-editing outcomes "
            "and test whether path metrics transfer to editing robustness prediction."
        )
    )
    parser.add_argument("--path-rows-csv", required=True)
    parser.add_argument("--editing-rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-label", default="heldout_144_agop_direct")
    parser.add_argument("--methods", nargs="+", default=["agop_top1", "agop_topk_project"])
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def to_bool(value: Any) -> Optional[bool]:
    if value in {None, ""}:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def finite_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def rate(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [to_bool(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return mean([1.0 if v else 0.0 for v in vals])


def median_or_none(values: List[float]) -> Optional[float]:
    return median(values) if values else None


def pivot_path_rows(path_rows: List[Dict[str, str]], methods: List[str], dataset_label: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = defaultdict(dict)
    keep_keys = [
        "path_type",
        "target_onset_strength",
        "damage_onset_strength",
        "damage_minus_target_onset",
        "clean_window_exists",
        "clean_window_width",
        "target_success_any",
        "neighbor_damage_any",
        "collapse_flag",
        "target_auc",
        "neighbor_damage_auc",
        "pareto_score",
        "target_min_delta",
        "neighbor_min_delta",
        "target_monotonicity_error",
        "neighbor_monotonicity_error",
        "target_smoothness",
        "neighbor_smoothness",
    ]
    for row in path_rows:
        if row.get("dataset_label") != dataset_label:
            continue
        method = row.get("control_method", "")
        if method not in methods:
            continue
        rid = row.get("id", "")
        if not rid:
            continue
        prefix = f"path_{method}_"
        out[rid]["id"] = rid
        for key in keep_keys:
            out[rid][prefix + key] = row.get(key, "")
    return out


def join_rows(path_rows: List[Dict[str, str]], editing_rows: List[Dict[str, str]], methods: List[str], dataset_label: str) -> List[Dict[str, Any]]:
    path_by_id = pivot_path_rows(path_rows, methods, dataset_label)
    joined = []
    for row in editing_rows:
        rid = row.get("id", "")
        if rid not in path_by_id:
            continue
        item: Dict[str, Any] = dict(row)
        item.update(path_by_id[rid])
        joined.append(item)
    return joined


def group_summary(joined: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    rows = []
    outcomes = [
        "standard_success_bool",
        "robust_success_bool",
        "fragile_success_bool",
        "locality_damage_bool",
        "weak_post_margin_bool",
        "weak_rephrase_bool",
    ]
    numeric_outcomes = ["rewrite_margin_delta", "rephrase_margin_delta", "locality_logprob_delta", "superficial_risk_score"]
    for method in methods:
        key = f"path_{method}_path_type"
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in joined:
            groups[str(row.get(key, ""))].append(row)
        for path_type, group in sorted(groups.items()):
            if not path_type:
                continue
            out: Dict[str, Any] = {
                "control_method": method,
                "path_type": path_type,
                "n_records": len(group),
                "target_onset_median": median_or_none(finite_values(group, f"path_{method}_target_onset_strength")),
                "damage_onset_median": median_or_none(finite_values(group, f"path_{method}_damage_onset_strength")),
                "clean_window_width_median": median_or_none(finite_values(group, f"path_{method}_clean_window_width")),
                "damage_minus_target_median": median_or_none(finite_values(group, f"path_{method}_damage_minus_target_onset")),
            }
            for outcome in outcomes:
                out[f"{outcome}_rate"] = rate(group, outcome)
            for outcome in numeric_outcomes:
                vals = finite_values(group, outcome)
                out[f"{outcome}_mean"] = mean(vals) if vals else None
                out[f"{outcome}_median"] = median_or_none(vals)
            rows.append(out)
    return rows


def one_hot_path_type(value: str, path_type: str) -> float:
    return 1.0 if value == path_type else 0.0


DIAGNOSTIC_FEATURES = [
    "pre_margin",
    "pre_rewrite_logprob",
    "pre_rewrite_mean_logprob",
    "pre_old_logprob",
    "pre_old_mean_logprob",
    "pre_rewrite_num_tokens",
    "pre_old_num_tokens",
    "prompt_chars",
    "prompt_words",
    "subject_chars",
    "subject_words",
    "subject_prompt_pos",
    "subject_prompt_char_ratio",
    "target_new_chars",
    "target_new_words",
    "ground_truth_chars",
    "ground_truth_words",
    "num_paraphrase_prompts",
    "num_locality_prompts",
]


def feature_set_parts(methods: List[str], feature_set: str) -> Tuple[bool, List[str]]:
    if feature_set == "diagnostic":
        return True, []
    if feature_set == "both":
        return False, methods
    if feature_set == "diagnostic+both":
        return True, methods
    if feature_set.startswith("diagnostic+"):
        method = feature_set.split("+", 1)[1]
        return True, [method]
    return False, [feature_set]


def build_feature_matrix(rows: List[Dict[str, Any]], methods: List[str], feature_set: str) -> Tuple[List[List[float]], List[str]]:
    include_diagnostic, method_list = feature_set_parts(methods, feature_set)
    base_numeric = [
        "target_onset_strength",
        "damage_onset_strength",
        "damage_minus_target_onset",
        "clean_window_width",
        "target_auc",
        "neighbor_damage_auc",
        "pareto_score",
        "target_min_delta",
        "neighbor_min_delta",
        "target_monotonicity_error",
        "neighbor_monotonicity_error",
        "target_smoothness",
        "neighbor_smoothness",
    ]
    path_types = ["clean-window", "damage-first", "no-effect", "collapse", "unstable"]
    names = []
    if include_diagnostic:
        names.extend(DIAGNOSTIC_FEATURES)
    for method in method_list:
        for key in base_numeric:
            names.append(f"path_{method}_{key}")
        for path_type in path_types:
            names.append(f"path_{method}_is_{path_type}")
        names.extend(
            [
                f"path_{method}_clean_window_exists",
                f"path_{method}_target_success_any",
                f"path_{method}_neighbor_damage_any",
                f"path_{method}_collapse_flag",
            ]
        )

    matrix = []
    for row in rows:
        feats = []
        if include_diagnostic:
            for key in DIAGNOSTIC_FEATURES:
                value = to_float(row.get(key))
                feats.append(0.0 if value is None else value)
        for method in method_list:
            for key in base_numeric:
                value = to_float(row.get(f"path_{method}_{key}"))
                feats.append(0.0 if value is None else value)
            path_type_value = str(row.get(f"path_{method}_path_type", ""))
            for path_type in path_types:
                feats.append(one_hot_path_type(path_type_value, path_type))
            for key in ["clean_window_exists", "target_success_any", "neighbor_damage_any", "collapse_flag"]:
                value = to_bool(row.get(f"path_{method}_{key}"))
                feats.append(1.0 if value else 0.0)
        matrix.append(feats)
    return matrix, names


def classification_report(joined: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return [{"error": f"sklearn unavailable: {exc}"}]

    labels = [
        "standard_success_bool",
        "robust_success_bool",
        "fragile_success_bool",
        "locality_damage_bool",
        "weak_post_margin_bool",
        "weak_rephrase_bool",
    ]
    feature_sets = (
        ["diagnostic"]
        + methods
        + ["both"]
        + [f"diagnostic+{method}" for method in methods]
        + ["diagnostic+both"]
    )
    reports: List[Dict[str, Any]] = []
    for label in labels:
        valid_rows = []
        y = []
        for row in joined:
            value = to_bool(row.get(label))
            if value is None:
                continue
            valid_rows.append(row)
            y.append(1 if value else 0)
        if len(valid_rows) < 20 or len(set(y)) < 2 or min(Counter(y).values()) < 5:
            reports.append({"outcome": label, "available": False, "reason": "insufficient_labels", "n": len(valid_rows), "positive_rate": mean(y) if y else None})
            continue
        for feature_set in feature_sets:
            x, names = build_feature_matrix(valid_rows, methods, feature_set)
            x_arr = np.array(x, dtype=float)
            y_arr = np.array(y, dtype=int)
            folds = min(5, min(Counter(y).values()))
            if folds < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
            )
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            probs = cross_val_predict(model, x_arr, y_arr, cv=cv, method="predict_proba")[:, 1]
            reports.append(
                {
                    "outcome": label,
                    "feature_set": feature_set,
                    "available": True,
                    "n": len(valid_rows),
                    "positive_rate": float(y_arr.mean()),
                    "n_features": len(names),
                    "cv_folds": folds,
                    "cv_auroc": float(roc_auc_score(y_arr, probs)),
                    "cv_average_precision": float(average_precision_score(y_arr, probs)),
                }
            )
    return reports


def controlled_deltas(clf_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {
        (row.get("outcome"), row.get("feature_set")): row
        for row in clf_rows
        if row.get("available")
    }
    comparisons = []
    for outcome in sorted({row.get("outcome") for row in clf_rows if row.get("feature_set") == "diagnostic"}):
        base_row = by_key.get((outcome, "diagnostic"))
        if not base_row:
            continue
        for feature_set in sorted(fs for (label, fs) in by_key if label == outcome and str(fs).startswith("diagnostic+")):
            plus = by_key[(outcome, feature_set)]
            comparisons.append(
                {
                    "outcome": outcome,
                    "comparison": f"{feature_set}_over_diagnostic",
                    "n": plus.get("n"),
                    "base_auroc": base_row.get("cv_auroc"),
                    "plus_auroc": plus.get("cv_auroc"),
                    "delta_auroc": float(plus.get("cv_auroc", 0.0)) - float(base_row.get("cv_auroc", 0.0)),
                    "base_ap": base_row.get("cv_average_precision"),
                    "plus_ap": plus.get("cv_average_precision"),
                    "delta_ap": float(plus.get("cv_average_precision", 0.0)) - float(base_row.get("cv_average_precision", 0.0)),
                }
            )
    return comparisons


def numeric_associations(joined: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    try:
        from scipy.stats import spearmanr
    except Exception as exc:
        return [{"error": f"scipy unavailable: {exc}"}]
    numeric_outcomes = ["rewrite_margin_delta", "rephrase_margin_delta", "locality_logprob_delta", "superficial_risk_score"]
    path_metrics = ["clean_window_width", "damage_minus_target_onset", "target_auc", "neighbor_damage_auc", "pareto_score"]
    out = []
    for method in methods:
        for metric in path_metrics:
            x_key = f"path_{method}_{metric}"
            for outcome in numeric_outcomes:
                pairs = []
                for row in joined:
                    x = to_float(row.get(x_key))
                    y = to_float(row.get(outcome))
                    if x is not None and y is not None:
                        pairs.append((x, y))
                if len(pairs) < 10:
                    continue
                xs, ys = zip(*pairs)
                rho, pvalue = spearmanr(xs, ys)
                out.append(
                    {
                        "control_method": method,
                        "path_metric": metric,
                        "editing_outcome": outcome,
                        "n": len(pairs),
                        "spearman_rho": float(rho),
                        "pvalue": float(pvalue),
                    }
                )
    return out


def build_markdown(
    summary: Dict[str, Any],
    group_rows: List[Dict[str, Any]],
    clf_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
    assoc_rows: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Path Geometry to Editing Transfer Analysis",
        "",
        "This analysis joins held-out activation path metrics with ROME / superficial-editing outcomes by record id.",
        "",
        "## Inputs",
        "",
        f"- path rows: `{summary['path_rows_csv']}`",
        f"- editing rows: `{summary['editing_rows_csv']}`",
        f"- dataset_label: `{summary['dataset_label']}`",
        f"- joined records: `{summary['n_joined']}`",
        "",
        "## Path Type Editing Summary",
        "",
        "| Method | Path type | n | standard success | robust success | fragile success | locality damage | rewrite margin mean | superficial risk mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            "| {control_method} | {path_type} | {n_records} | {standard} | {robust} | {fragile} | {locality} | {rewrite} | {risk} |".format(
                control_method=row.get("control_method", ""),
                path_type=row.get("path_type", ""),
                n_records=row.get("n_records", ""),
                standard=f"{row['standard_success_bool_rate']:.4f}" if row.get("standard_success_bool_rate") is not None else "",
                robust=f"{row['robust_success_bool_rate']:.4f}" if row.get("robust_success_bool_rate") is not None else "",
                fragile=f"{row['fragile_success_bool_rate']:.4f}" if row.get("fragile_success_bool_rate") is not None else "",
                locality=f"{row['locality_damage_bool_rate']:.4f}" if row.get("locality_damage_bool_rate") is not None else "",
                rewrite=f"{row['rewrite_margin_delta_mean']:.4f}" if row.get("rewrite_margin_delta_mean") is not None else "",
                risk=f"{row['superficial_risk_score_mean']:.4f}" if row.get("superficial_risk_score_mean") is not None else "",
            )
        )
    lines.extend(["", "## Prediction Summary", "", "| Outcome | Feature set | n | positive rate | AUROC | AP |", "|---|---|---:|---:|---:|---:|"])
    for row in clf_rows:
        if not row.get("available"):
            continue
        lines.append(
            "| {outcome} | {feature_set} | {n} | {positive_rate:.4f} | {cv_auroc:.4f} | {cv_average_precision:.4f} |".format(**row)
        )
    lines.extend(["", "## Controlled Gain Over Diagnostic Features", "", "| Outcome | Comparison | n | base AUROC | plus AUROC | delta AUROC | base AP | plus AP | delta AP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in delta_rows:
        lines.append(
            "| {outcome} | {comparison} | {n} | {base_auroc:.4f} | {plus_auroc:.4f} | {delta_auroc:.4f} | {base_ap:.4f} | {plus_ap:.4f} | {delta_ap:.4f} |".format(
                outcome=row["outcome"],
                comparison=row["comparison"],
                n=row.get("n", ""),
                base_auroc=float(row["base_auroc"]),
                plus_auroc=float(row["plus_auroc"]),
                delta_auroc=float(row["delta_auroc"]),
                base_ap=float(row["base_ap"]),
                plus_ap=float(row["plus_ap"]),
                delta_ap=float(row["delta_ap"]),
            )
        )
    lines.extend(["", "## Strongest Numeric Associations", "", "| Method | Path metric | Editing outcome | n | Spearman rho | p-value |", "|---|---|---|---:|---:|---:|"])
    for row in sorted(
        [r for r in assoc_rows if "spearman_rho" in r],
        key=lambda r: abs(float(r["spearman_rho"])),
        reverse=True,
    )[:20]:
        lines.append(
            "| {control_method} | {path_metric} | {editing_outcome} | {n} | {spearman_rho:.4f} | {pvalue:.4g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This is a transfer analysis: activation path metrics are measured before editing, while outcomes come from ROME.",
            "- Small path-type groups should be read as diagnostics, not definitive population estimates.",
            "- Strong prediction here would support the claim that intervention path geometry transfers to parameter editing robustness.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_rows = read_csv(Path(args.path_rows_csv))
    editing_rows = read_csv(Path(args.editing_rows_csv))
    joined = join_rows(path_rows, editing_rows, args.methods, args.dataset_label)
    group_rows = group_summary(joined, args.methods)
    clf_rows = classification_report(joined, args.methods)
    delta_rows = controlled_deltas(clf_rows)
    assoc_rows = numeric_associations(joined, args.methods)
    write_csv(output_dir / "path_editing_joined_rows.csv", joined)
    write_csv(output_dir / "path_type_editing_group_summary.csv", group_rows)
    write_csv(output_dir / "path_editing_prediction_rows.csv", clf_rows)
    write_csv(output_dir / "path_editing_controlled_deltas.csv", delta_rows)
    write_csv(output_dir / "path_numeric_associations.csv", assoc_rows)
    summary = {
        "path_rows_csv": args.path_rows_csv,
        "editing_rows_csv": args.editing_rows_csv,
        "dataset_label": args.dataset_label,
        "methods": args.methods,
        "n_path_rows": len(path_rows),
        "n_editing_rows": len(editing_rows),
        "n_joined": len(joined),
        "joined_ids": len({row.get("id", "") for row in joined}),
        "path_type_counts": {
            method: dict(Counter(row.get(f"path_{method}_path_type", "") for row in joined))
            for method in args.methods
        },
        "classification": clf_rows,
        "controlled_deltas": delta_rows,
        "numeric_associations": assoc_rows,
    }
    (output_dir / "path_editing_transfer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "PATH_EDITING_TRANSFER_SUMMARY.md").write_text(
        build_markdown(summary, group_rows, clf_rows, delta_rows, assoc_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "n_joined": len(joined)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
