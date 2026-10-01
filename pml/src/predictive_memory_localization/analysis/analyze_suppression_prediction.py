from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze which localization features predict suppression outcomes."
    )
    parser.add_argument("--per-record-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-delta-threshold",
        type=float,
        default=-0.05,
        help="Target success threshold, used only if target_success is absent.",
    )
    parser.add_argument(
        "--neighbor-damage-threshold",
        type=float,
        default=-0.05,
        help="Neighbor damage threshold, used only if neighbor_damaged is absent.",
    )
    parser.add_argument(
        "--include-method-coef",
        action="store_true",
        help="Include one-hot control method and numeric coef as predictors in the logistic model.",
    )
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def load_rows(path: str, target_threshold: float, damage_threshold: float) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if "target_success" in row and row["target_success"] != "":
            row["target_success_bool"] = parse_bool(row["target_success"])
        else:
            row["target_success_bool"] = (to_float(row.get("target_mean_delta")) or 0.0) <= target_threshold
        if "neighbor_damaged" in row and row["neighbor_damaged"] != "":
            row["neighbor_damaged_bool"] = parse_bool(row["neighbor_damaged"])
        else:
            row["neighbor_damaged_bool"] = (to_float(row.get("neighbor_mean_delta")) or 0.0) <= damage_threshold
    return rows


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique_values = np.unique(values)
    for value in unique_values:
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    total = float(len(pos) * len(neg))
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / total


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = 0
    precisions: List[float] = []
    for idx, label in enumerate(sorted_labels, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / idx)
    return float(np.mean(precisions)) if precisions else 0.0


def feature_columns(rows: List[Dict[str, Any]]) -> List[str]:
    blocked = {
        "feature_top_saliency_layer",
    }
    candidates = sorted({key for row in rows for key in row if key.startswith("feature_")})
    out = []
    for key in candidates:
        if key in blocked:
            continue
        values = [to_float(row.get(key)) for row in rows]
        values = [v for v in values if v is not None]
        if len(values) >= 3 and len(set(values)) > 1:
            out.append(key)
    return out


def predictor_columns(rows: List[Dict[str, Any]], include_method_coef: bool) -> List[str]:
    columns = feature_columns(rows)
    if include_method_coef:
        columns.append("coef")
        for method in sorted({row.get("control_method", "") for row in rows}):
            columns.append(f"method={method}")
    return columns


def predictor_value(row: Dict[str, Any], column: str) -> Optional[float]:
    if column == "coef":
        return to_float(row.get("coef"))
    if column.startswith("method="):
        return 1.0 if row.get("control_method", "") == column.split("=", 1)[1] else 0.0
    return to_float(row.get(column))


def single_feature_report(rows: List[Dict[str, Any]], label_key: str, outcome_key: str) -> List[Dict[str, Any]]:
    labels = np.array([1 if row[label_key] else 0 for row in rows], dtype=int)
    outcome = np.array([to_float(row.get(outcome_key)) or 0.0 for row in rows], dtype=float)
    reports: List[Dict[str, Any]] = []
    for key in feature_columns(rows):
        pairs = [(to_float(row.get(key)), idx) for idx, row in enumerate(rows)]
        pairs = [(value, idx) for value, idx in pairs if value is not None]
        if len(pairs) < 3:
            continue
        values = np.array([value for value, _ in pairs], dtype=float)
        sub_labels = np.array([labels[idx] for _, idx in pairs], dtype=int)
        sub_outcome = np.array([outcome[idx] for _, idx in pairs], dtype=float)
        reports.append(
            {
                "feature": key,
                "n": int(len(values)),
                "pearson_with_delta": pearson(values, sub_outcome),
                "spearman_with_delta": spearman(values, sub_outcome),
                "auroc_for_label": auroc(values, sub_labels),
                "average_precision_for_label": average_precision(values, sub_labels),
                "positive_mean": float(values[sub_labels == 1].mean()) if int(sub_labels.sum()) else 0.0,
                "negative_mean": float(values[sub_labels == 0].mean()) if int((sub_labels == 0).sum()) else 0.0,
            }
        )
    return sorted(reports, key=lambda row: abs(row["spearman_with_delta"]), reverse=True)


def fit_logistic_report(
    rows: List[Dict[str, Any]],
    label_key: str,
    *,
    include_method_coef: bool = False,
) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False, "reason": "scikit-learn is not installed"}

    features = predictor_columns(rows, include_method_coef)
    complete_rows = []
    for row in rows:
        values = [predictor_value(row, key) for key in features]
        if all(v is not None for v in values):
            complete_rows.append((values, 1 if row[label_key] else 0))

    if len(complete_rows) < 10:
        return {"available": False, "reason": "fewer than 10 complete rows"}
    y = np.array([label for _, label in complete_rows], dtype=int)
    if len(np.unique(y)) < 2:
        return {"available": False, "reason": "only one class is present"}

    X = np.array([values for values, _ in complete_rows], dtype=float)
    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        return {"available": False, "reason": "not enough examples per class for cross-validation"}

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    model.fit(X, y)
    clf = model.named_steps["logisticregression"]
    coefs = [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in zip(features, clf.coef_[0])
    ]
    coefs.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    return {
        "available": True,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "cv_splits": int(n_splits),
        "cv_auroc": auroc(probs, y),
        "cv_average_precision": average_precision(probs, y),
        "top_coefficients": coefs[:12],
    }


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_deltas = [to_float(row.get("target_mean_delta")) or 0.0 for row in rows]
    neighbor_deltas = [to_float(row.get("neighbor_mean_delta")) or 0.0 for row in rows]
    return {
        "n_rows": len(rows),
        "n_records": len({row.get("id") for row in rows}),
        "target_success_rate": float(np.mean([row["target_success_bool"] for row in rows])) if rows else 0.0,
        "neighbor_damage_rate": float(np.mean([row["neighbor_damaged_bool"] for row in rows])) if rows else 0.0,
        "target_mean_delta": float(np.mean(target_deltas)) if target_deltas else 0.0,
        "neighbor_mean_delta": float(np.mean(neighbor_deltas)) if neighbor_deltas else 0.0,
    }


def subgroup_reports(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, Any] = {}
    for method in sorted({row.get("control_method", "") for row in rows}):
        group = [row for row in rows if row.get("control_method", "") == method]
        by_method[method] = {
            **summarize_group(group),
            "target_logistic_model": fit_logistic_report(group, "target_success_bool"),
            "neighbor_damage_logistic_model": fit_logistic_report(group, "neighbor_damaged_bool"),
        }

    by_method_coef: List[Dict[str, Any]] = []
    keys = sorted(
        {
            (row.get("control_method", ""), to_float(row.get("coef")) or 0.0)
            for row in rows
        }
    )
    for method, coef in keys:
        group = [
            row
            for row in rows
            if row.get("control_method", "") == method
            and (to_float(row.get("coef")) or 0.0) == coef
        ]
        entry = {
            "control_method": method,
            "coef": coef,
            **summarize_group(group),
            "target_logistic_model": fit_logistic_report(group, "target_success_bool"),
            "neighbor_damage_logistic_model": fit_logistic_report(group, "neighbor_damaged_bool"),
        }
        by_method_coef.append(entry)

    return {"by_method": by_method, "by_method_coef": by_method_coef}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_subgroup_models(subgroups: Dict[str, Any], outcome_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, report in subgroups["by_method"].items():
        model = report[outcome_key]
        rows.append(
            {
                "group_type": "method",
                "control_method": method,
                "coef": "",
                "n_rows": report["n_rows"],
                "n_records": report["n_records"],
                "positive_rate": model.get("positive_rate", ""),
                "cv_auroc": model.get("cv_auroc", ""),
                "cv_average_precision": model.get("cv_average_precision", ""),
                "available": model.get("available", False),
                "reason": model.get("reason", ""),
            }
        )
    for report in subgroups["by_method_coef"]:
        model = report[outcome_key]
        rows.append(
            {
                "group_type": "method_coef",
                "control_method": report["control_method"],
                "coef": report["coef"],
                "n_rows": report["n_rows"],
                "n_records": report["n_records"],
                "positive_rate": model.get("positive_rate", ""),
                "cv_auroc": model.get("cv_auroc", ""),
                "cv_average_precision": model.get("cv_average_precision", ""),
                "available": model.get("available", False),
                "reason": model.get("reason", ""),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.per_record_csv, args.target_delta_threshold, args.neighbor_damage_threshold)
    target_report = single_feature_report(rows, "target_success_bool", "target_mean_delta")
    damage_report = single_feature_report(rows, "neighbor_damaged_bool", "neighbor_mean_delta")
    subgroups = subgroup_reports(rows)

    summary = {
        "n_rows": len(rows),
        "n_records": len({row.get("id") for row in rows}),
        "control_methods": sorted({row.get("control_method") for row in rows}),
        "coefs": sorted({float(row.get("coef", 0.0)) for row in rows}),
        "target_success_rate": float(np.mean([row["target_success_bool"] for row in rows])) if rows else 0.0,
        "neighbor_damage_rate": float(np.mean([row["neighbor_damaged_bool"] for row in rows])) if rows else 0.0,
        "target_logistic_model": fit_logistic_report(rows, "target_success_bool"),
        "neighbor_damage_logistic_model": fit_logistic_report(rows, "neighbor_damaged_bool"),
        "target_logistic_model_with_method_coef": fit_logistic_report(
            rows,
            "target_success_bool",
            include_method_coef=True,
        ),
        "neighbor_damage_logistic_model_with_method_coef": fit_logistic_report(
            rows,
            "neighbor_damaged_bool",
            include_method_coef=True,
        ),
        "subgroups": subgroups,
    }

    (out_dir / "prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "target_feature_report.csv", target_report)
    write_csv(out_dir / "neighbor_damage_feature_report.csv", damage_report)
    write_csv(
        out_dir / "target_subgroup_model_report.csv",
        flatten_subgroup_models(subgroups, "target_logistic_model"),
    )
    write_csv(
        out_dir / "neighbor_damage_subgroup_model_report.csv",
        flatten_subgroup_models(subgroups, "neighbor_damage_logistic_model"),
    )

    print(f"Wrote prediction summary to {out_dir / 'prediction_summary.json'}")
    print(f"Wrote target feature report to {out_dir / 'target_feature_report.csv'}")
    print(f"Wrote neighbor damage feature report to {out_dir / 'neighbor_damage_feature_report.csv'}")
    print(f"Wrote target subgroup report to {out_dir / 'target_subgroup_model_report.csv'}")
    print(f"Wrote neighbor damage subgroup report to {out_dir / 'neighbor_damage_subgroup_model_report.csv'}")


if __name__ == "__main__":
    main()
