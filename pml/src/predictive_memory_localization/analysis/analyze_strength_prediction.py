from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.analysis.analyze_suppression_prediction import (
    average_precision,
    auroc,
    parse_bool,
    spearman,
    to_float,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze localization features as predictors of strength/path outcomes."
    )
    parser.add_argument("--path-rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-method",
        action="store_true",
        help="Include one-hot control method features in multivariate models.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help=(
            "Optional group names to analyze, e.g. pooled dataset=heldout_50_capability_onset_probe. "
            "By default all dataset/method groups are analyzed."
        ),
    )
    parser.add_argument(
        "--label-keys",
        nargs="*",
        default=None,
        help="Optional classification outcomes to analyze.",
    )
    parser.add_argument(
        "--numeric-keys",
        nargs="*",
        default=None,
        help="Optional numeric outcomes to analyze.",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="*",
        default=None,
        help="Optional feature groups to analyze: all, baseline, agop, agop_spectrum, agop_topvec.",
    )
    parser.add_argument(
        "--single-feature-only",
        action="store_true",
        help="Skip multivariate CV models and only write single-feature reports.",
    )
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def feature_columns(rows: List[Dict[str, Any]], prefix: Optional[str] = None) -> List[str]:
    candidates = sorted({key for row in rows for key in row if key.startswith("feature_")})
    out = []
    for key in candidates:
        if prefix and not key.startswith(prefix):
            continue
        values = [to_float(row.get(key)) for row in rows]
        values = [v for v in values if v is not None]
        if len(values) >= 10 and len(set(values)) > 1:
            out.append(key)
    return out


def feature_groups(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    groups = {
        "all": feature_columns(rows),
        "baseline": [
            key for key in feature_columns(rows) if not key.startswith("feature_agop_")
        ],
        "agop": [
            key for key in feature_columns(rows) if key.startswith("feature_agop_")
        ],
        "agop_spectrum": [
            key
            for key in feature_columns(rows)
            if key.startswith("feature_agop_")
            and any(token in key for token in ["top1", "topk", "entropy", "effective_rank", "eigengap"])
        ],
        "agop_topvec": [
            key
            for key in feature_columns(rows)
            if key.startswith("feature_agop_") and "top_vec" in key
        ],
    }
    return {name: cols for name, cols in groups.items() if cols}


def bool_label(row: Dict[str, Any], key: str) -> bool:
    if key == "clean_window_label":
        return parse_bool(row.get("clean_window_exists"))
    if key == "damage_first_label":
        return str(row.get("path_type")) == "damage-first"
    if key == "capability_first_label":
        return str(row.get("path_type")) == "capability-first"
    if key == "no_effect_label":
        return str(row.get("path_type")) == "no-effect"
    if key == "usable_path_label":
        return str(row.get("path_type")) == "clean-window"
    return parse_bool(row.get(key))


def numeric_outcome(row: Dict[str, Any], key: str) -> Optional[float]:
    value = to_float(row.get(key))
    if value is None:
        return None
    return value


def single_feature_reports(
    rows: List[Dict[str, Any]],
    label_keys: List[str],
    numeric_keys: List[str],
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for feature in feature_columns(rows):
        pairs = [(to_float(row.get(feature)), idx) for idx, row in enumerate(rows)]
        pairs = [(value, idx) for value, idx in pairs if value is not None]
        if len(pairs) < 10:
            continue
        values = np.array([value for value, _ in pairs], dtype=float)
        for label_key in label_keys:
            labels = np.array([1 if bool_label(rows[idx], label_key) else 0 for _, idx in pairs], dtype=int)
            if len(np.unique(labels)) > 1:
                reports.append(
                    {
                        "feature": feature,
                        "outcome": label_key,
                        "kind": "classification",
                        "n": int(len(values)),
                        "positive_rate": float(labels.mean()),
                        "auroc": auroc(values, labels),
                        "average_precision": average_precision(values, labels),
                    }
                )
        for numeric_key in numeric_keys:
            numeric_pairs = [
                (value, numeric_outcome(rows[idx], numeric_key))
                for value, idx in pairs
            ]
            numeric_pairs = [(value, target) for value, target in numeric_pairs if target is not None]
            if len(numeric_pairs) >= 10:
                x = np.array([value for value, _ in numeric_pairs], dtype=float)
                y = np.array([target for _, target in numeric_pairs], dtype=float)
                if np.std(y) > 0 and np.std(x) > 0:
                    reports.append(
                        {
                            "feature": feature,
                            "outcome": numeric_key,
                            "kind": "regression",
                            "n": int(len(x)),
                            "spearman": spearman(x, y),
                            "pearson": float(np.corrcoef(x, y)[0, 1]),
                        }
                    )
    return sorted(
        reports,
        key=lambda row: max(
            abs(float(row.get("spearman", 0.0) or 0.0)),
            abs(float(row.get("auroc", 0.5) or 0.5) - 0.5),
        ),
        reverse=True,
    )


def complete_matrix(
    rows: List[Dict[str, Any]],
    cols: List[str],
    *,
    include_method: bool,
    outcome_key: str,
    classification: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    methods = sorted({str(row.get("control_method", "")) for row in rows}) if include_method else []
    feature_names = list(cols) + [f"method={method}" for method in methods]
    X_rows = []
    y_rows = []
    for row in rows:
        values = [to_float(row.get(col)) for col in cols]
        if any(value is None for value in values):
            continue
        values = [float(value) for value in values if value is not None]
        values.extend([1.0 if row.get("control_method") == method else 0.0 for method in methods])
        if classification:
            y = 1 if bool_label(row, outcome_key) else 0
        else:
            target = numeric_outcome(row, outcome_key)
            if target is None:
                continue
            y = float(target)
        X_rows.append(values)
        y_rows.append(y)
    return np.array(X_rows, dtype=float), np.array(y_rows), feature_names


def logistic_cv(rows: List[Dict[str, Any]], cols: List[str], label_key: str, include_method: bool) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False, "reason": "scikit-learn unavailable"}

    X, y, feature_names = complete_matrix(
        rows,
        cols,
        include_method=include_method,
        outcome_key=label_key,
        classification=True,
    )
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {"available": False, "reason": "not enough labeled rows", "n": int(len(y))}
    min_class = int(np.bincount(y.astype(int)).min())
    if min_class < 2:
        return {"available": False, "reason": "not enough rows per class", "n": int(len(y))}
    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    probs = cross_val_predict(model, X, y.astype(int), cv=cv, method="predict_proba")[:, 1]
    model.fit(X, y.astype(int))
    coefs = model.named_steps["logisticregression"].coef_[0]
    top = sorted(
        [
            {"feature": feature, "coefficient": float(coef)}
            for feature, coef in zip(feature_names, coefs)
        ],
        key=lambda row: abs(row["coefficient"]),
        reverse=True,
    )[:12]
    return {
        "available": True,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "cv_auroc": auroc(probs, y.astype(int)),
        "cv_average_precision": average_precision(probs, y.astype(int)),
        "n_features": len(feature_names),
        "top_coefficients": top,
    }


def ridge_cv(rows: List[Dict[str, Any]], cols: List[str], outcome_key: str, include_method: bool) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False, "reason": "scikit-learn unavailable"}

    X, y, feature_names = complete_matrix(
        rows,
        cols,
        include_method=include_method,
        outcome_key=outcome_key,
        classification=False,
    )
    if len(y) < 20 or np.std(y) == 0:
        return {"available": False, "reason": "not enough numeric rows", "n": int(len(y))}
    cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    preds = cross_val_predict(model, X, y.astype(float), cv=cv)
    mae = float(np.mean(np.abs(preds - y)))
    corr = spearman(preds, y.astype(float)) if np.std(preds) > 0 else 0.0
    return {
        "available": True,
        "n": int(len(y)),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "cv_spearman": corr,
        "cv_mae": mae,
        "n_features": len(feature_names),
    }


def group_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = {"pooled": rows}
    for dataset in sorted({str(row.get("dataset_label", "")) for row in rows}):
        groups[f"dataset={dataset}"] = [row for row in rows if row.get("dataset_label") == dataset]
    for dataset in sorted({str(row.get("dataset_label", "")) for row in rows}):
        subset = [row for row in rows if row.get("dataset_label") == dataset]
        for method in sorted({str(row.get("control_method", "")) for row in subset}):
            groups[f"dataset={dataset}|method={method}"] = [
                row for row in subset if row.get("control_method") == method
            ]
    return groups


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.path_rows_csv)
    default_label_keys = [
        "clean_window_label",
        "usable_path_label",
        "damage_first_label",
        "capability_first_label",
        "no_effect_label",
    ]
    default_numeric_keys = [
        "target_onset_strength",
        "damage_onset_strength",
        "capability_onset_strength",
        "damage_minus_target_onset",
        "capability_minus_target_onset",
        "clean_window_width",
        "target_auc",
        "neighbor_damage_auc",
        "capability_damage_auc",
        "pareto_score",
        "pareto_score_with_capability",
    ]
    label_keys = args.label_keys or default_label_keys
    numeric_keys = args.numeric_keys or default_numeric_keys

    all_reports = []
    model_summaries: Dict[str, Any] = {}
    for group_name, group in group_rows(rows).items():
        if args.groups is not None and group_name not in set(args.groups):
            continue
        if len(group) < 20:
            continue
        group_feature_groups = feature_groups(group)
        if args.feature_groups is not None:
            requested = set(args.feature_groups)
            group_feature_groups = {
                name: cols for name, cols in group_feature_groups.items() if name in requested
            }
        all_reports.extend(
            [
                {"group": group_name, **row}
                for row in single_feature_reports(group, label_keys, numeric_keys)
            ]
        )
        model_summaries[group_name] = {
            "n": len(group),
            "feature_groups": {},
        }
        if args.single_feature_only:
            continue
        for feature_group_name, cols in group_feature_groups.items():
            model_summaries[group_name]["feature_groups"][feature_group_name] = {
                "n_features": len(cols),
                "classification": {
                    key: logistic_cv(group, cols, key, include_method=args.include_method)
                    for key in label_keys
                },
                "regression": {
                    key: ridge_cv(group, cols, key, include_method=args.include_method)
                    for key in numeric_keys
                },
            }

    write_csv(out_dir / "strength_single_feature_report.csv", all_reports)
    (out_dir / "strength_prediction_summary.json").write_text(
        json.dumps(
            {
                "path_rows_csv": args.path_rows_csv,
                "n_rows": len(rows),
                "include_method": args.include_method,
                "groups": args.groups,
                "label_keys": label_keys,
                "numeric_keys": numeric_keys,
                "feature_groups": args.feature_groups,
                "single_feature_only": args.single_feature_only,
                "model_summaries": model_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote single-feature report to {out_dir / 'strength_single_feature_report.csv'}")
    print(f"Wrote prediction summary to {out_dir / 'strength_prediction_summary.json'}")


if __name__ == "__main__":
    main()
