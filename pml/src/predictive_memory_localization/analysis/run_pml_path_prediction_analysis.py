from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_SELECTOR_DATASET_DIR = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2_bidirectional_pilot_500/qwen3_1_7b/selector_dataset"
)
DEFAULT_OUTPUT_DIR = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2_bidirectional_pilot_500/qwen3_1_7b/pml_path_prediction"
)


TARGETS = [
    "target_any",
    "clean_any",
    "clean_suppression_window_exists",
    "clean_enhancement_window_exists",
    "bidirectional_clean_control",
    "damage_any",
    "damage_first_any",
    "no_effect",
]


FEATURE_GROUPS = {
    "B": [
        "target_base_margin_mean",
        "target_base_margin_median",
        "target_base_margin_min",
        "target_base_positive_rate",
        "neighbor_base_margin_mean",
        "neighbor_base_margin_median",
        "neighbor_base_positive_rate",
        "capability_base_margin_mean",
        "capability_base_margin_median",
        "capability_base_positive_rate",
        "target_neighbor_base_margin_gap",
        "target_capability_base_margin_gap",
    ],
    "M": ["method", "layer", "dataset", "domain", "freshness_group", "release_year"],
    "L": [
        "feature_max_abs_cohen_d",
        "feature_mean_abs_cohen_d",
        "feature_mean_threshold_accuracy",
        "feature_saliency_concentration",
        "feature_saliency_entropy",
        "feature_top_saliency_layer",
    ],
    "V": ["feature_direction_pairwise_cosine_mean"],
    "R": [
        "early_target_slope_mean",
        "early_neighbor_slope_mean",
        "early_capability_slope_mean",
        "early_target_sign_consistency",
        "early_damage_flag",
        "early_target_delta_neg_0.01",
        "early_target_delta_pos_0.01",
        "early_target_delta_neg_0.02",
        "early_target_delta_pos_0.02",
        "early_target_delta_neg_0.05",
        "early_target_delta_pos_0.05",
        "early_target_delta_neg_0.1",
        "early_target_delta_pos_0.1",
        "early_neighbor_delta_neg_0.1",
        "early_neighbor_delta_pos_0.1",
        "early_capability_delta_neg_0.1",
        "early_capability_delta_pos_0.1",
    ],
}


ABLATIONS = {
    "B": ["B"],
    "B+M": ["B", "M"],
    "B+M+L": ["B", "M", "L"],
    "B+M+L+V": ["B", "M", "L", "V"],
    "B+M+L+V+R": ["B", "M", "L", "V", "R"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PML path outcomes and run localization-as-prediction ablations."
    )
    parser.add_argument("--selector-dataset-dir", default=DEFAULT_SELECTOR_DATASET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--case-examples-per-type", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_abs(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return float("nan")


def mean_for(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def max_for(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
    return float(vals.max()) if len(vals) else float("nan")


def value_at_alpha(df: pd.DataFrame, alpha: float, col: str) -> float:
    sub = df[np.isclose(pd.to_numeric(df["alpha"], errors="coerce"), alpha)]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub[col], errors="coerce").iloc[0])


def enrich_path_outcomes(path_df: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.DataFrame:
    path_df = path_df.copy()
    for col in [
        "clean_suppression_window_exists",
        "clean_enhancement_window_exists",
        "bidirectional_clean_control",
    ]:
        path_df[col] = bool_series(path_df[col])

    path_df["suppression_onset_abs"] = path_df["suppression_onset_alpha"].map(safe_abs)
    path_df["enhancement_onset_abs"] = path_df["enhancement_onset_alpha"].map(safe_abs)
    path_df["negative_damage_onset_abs"] = path_df["negative_damage_onset_alpha"].map(safe_abs)
    path_df["positive_damage_onset_abs"] = path_df["positive_damage_onset_alpha"].map(safe_abs)
    path_df["target_any"] = path_df["suppression_onset_abs"].notna() | path_df["enhancement_onset_abs"].notna()
    path_df["damage_any"] = path_df["negative_damage_onset_abs"].notna() | path_df["positive_damage_onset_abs"].notna()
    path_df["clean_any"] = path_df["clean_suppression_window_exists"] | path_df["clean_enhancement_window_exists"]

    neg_damage_first = (
        path_df["negative_damage_onset_abs"].notna()
        & (
            path_df["suppression_onset_abs"].isna()
            | (path_df["negative_damage_onset_abs"] <= path_df["suppression_onset_abs"])
        )
    )
    pos_damage_first = (
        path_df["positive_damage_onset_abs"].notna()
        & (
            path_df["enhancement_onset_abs"].isna()
            | (path_df["positive_damage_onset_abs"] <= path_df["enhancement_onset_abs"])
        )
    )
    path_df["damage_first_any"] = neg_damage_first | pos_damage_first
    path_df["no_effect"] = (~path_df["target_any"]) & (~path_df["damage_any"])
    path_df["single_side_clean"] = path_df["clean_any"] & (~path_df["bidirectional_clean_control"])

    metrics: List[Dict[str, Any]] = []
    group_cols = ["record_id", "method", "layer"]
    for keys, group in alpha_df.groupby(group_cols, dropna=False):
        record_id, method, layer = keys
        alpha = pd.to_numeric(group["alpha"], errors="coerce")
        neg = alpha < 0
        pos = alpha > 0
        target = pd.to_numeric(group["target_delta"], errors="coerce")
        neighbor = pd.to_numeric(group["neighbor_delta"], errors="coerce")
        capability = pd.to_numeric(group["capability_delta"], errors="coerce")
        metrics.append(
            {
                "record_id": record_id,
                "method": method,
                "layer": layer,
                "target_signed_auc_suppression": mean_for(group.assign(_gain=-target), neg, "_gain"),
                "target_signed_auc_enhancement": mean_for(group.assign(_gain=target), pos, "_gain"),
                "neighbor_damage_auc_negative": mean_for(group.assign(_damage=(-neighbor).clip(lower=0)), neg, "_damage"),
                "neighbor_damage_auc_positive": mean_for(group.assign(_damage=(-neighbor).clip(lower=0)), pos, "_damage"),
                "capability_damage_auc_negative": mean_for(group.assign(_damage=(-capability).clip(lower=0)), neg, "_damage"),
                "capability_damage_auc_positive": mean_for(group.assign(_damage=(-capability).clip(lower=0)), pos, "_damage"),
                "target_delta_alpha_neg1": value_at_alpha(group, -1.0, "target_delta"),
                "target_delta_alpha_pos1": value_at_alpha(group, 1.0, "target_delta"),
                "max_clean_utility": max_for(group[bool_series(group["is_clean"])], pd.Series(True, index=group[bool_series(group["is_clean"])].index), "utility")
                if "is_clean" in group
                else float("nan"),
                "oracle_best_utility": max_for(group, pd.Series(True, index=group.index), "utility"),
            }
        )
    metric_df = pd.DataFrame(metrics)
    out = path_df.merge(metric_df, on=group_cols, how="left")
    return out


def available_features(df: pd.DataFrame, groups: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for group in groups:
        cols.extend(FEATURE_GROUPS[group])
    return [col for col in cols if col in df.columns]


def make_preprocessor(df: pd.DataFrame, features: Sequence[str]) -> ColumnTransformer:
    categorical = [
        col
        for col in features
        if col in {"method", "dataset", "domain", "freshness_group"} or df[col].dtype == object
    ]
    numeric = [col for col in features if col not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def model_specs(seed: int) -> Dict[str, Any]:
    return {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed),
        "gbdt": GradientBoostingClassifier(random_state=seed),
    }


def split_by_record(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = sorted(df["record_id"].dropna().unique())
    train_records, test_records = train_test_split(records, test_size=test_size, random_state=seed)
    train_set = set(train_records)
    return df[df["record_id"].isin(train_set)].copy(), df[~df["record_id"].isin(train_set)].copy()


def evaluate_binary_predictions(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    out = {
        "positive_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "n": int(len(y_true)),
    }
    if len(np.unique(y_true)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, scores))
        out["auprc"] = float(average_precision_score(y_true, scores))
    return out


def run_prediction(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    train_df, test_df = split_by_record(df, args.test_size, args.seed)
    rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        y_train = bool_series(train_df[target]).astype(int).to_numpy()
        y_test = bool_series(test_df[target]).astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            rows.append(
                {
                    "target": target,
                    "model": "skipped",
                    "feature_set": "",
                    "reason": "single_class_train",
                    "train_positive_rate": float(np.mean(y_train)) if len(y_train) else float("nan"),
                    "test_positive_rate": float(np.mean(y_test)) if len(y_test) else float("nan"),
                }
            )
            continue
        for feature_set, groups in ABLATIONS.items():
            features = available_features(df, groups)
            if not features:
                continue
            for model_name, estimator in model_specs(args.seed).items():
                pipeline = Pipeline(
                    [
                        ("preprocess", make_preprocessor(train_df, features)),
                        ("model", estimator),
                    ]
                )
                pipeline.fit(train_df[features], y_train)
                if hasattr(pipeline.named_steps["model"], "predict_proba"):
                    scores = pipeline.predict_proba(test_df[features])[:, 1]
                else:
                    scores = pipeline.decision_function(test_df[features])
                metrics = evaluate_binary_predictions(y_test, scores)
                rows.append(
                    {
                        "target": target,
                        "feature_set": feature_set,
                        "model": model_name,
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "train_positive_rate": float(np.mean(y_train)),
                        "test_positive_rate": metrics["positive_rate"],
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "n_features": len(features),
                        "features": ",".join(features),
                    }
                )
    return pd.DataFrame(rows)


def summarize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "target_any",
        "clean_any",
        "clean_suppression_window_exists",
        "clean_enhancement_window_exists",
        "bidirectional_clean_control",
        "damage_any",
        "damage_first_any",
        "no_effect",
    ]
    rows = []
    for keys, group in df.groupby(["method", "layer"], dropna=False):
        method, layer = keys
        row = {"method": method, "layer": layer, "n": len(group)}
        for col in summary_cols:
            if col in group:
                row[f"{col}_rate"] = float(bool_series(group[col]).mean())
        for col in [
            "target_signed_auc_suppression",
            "target_signed_auc_enhancement",
            "neighbor_damage_auc_negative",
            "neighbor_damage_auc_positive",
            "capability_damage_auc_negative",
            "capability_damage_auc_positive",
            "oracle_best_utility",
        ]:
            if col in group:
                row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "layer"])


def select_case_audit(df: pd.DataFrame, examples_per_type: int) -> pd.DataFrame:
    candidates: List[pd.DataFrame] = []
    specs = [
        ("clean_any", "clean_any", False),
        ("damage_first", "damage_first_any", False),
        ("no_effect", "no_effect", False),
        ("high_leverage_damage", "damage_any", True),
        ("bidirectional_clean", "bidirectional_clean_control", False),
    ]
    for label, col, require_target in specs:
        if col not in df:
            continue
        sub = df[bool_series(df[col])].copy()
        if require_target and "target_any" in sub:
            sub = sub[bool_series(sub["target_any"])]
        if sub.empty:
            continue
        sort_col = "oracle_best_utility" if "oracle_best_utility" in sub else "record_id"
        sub = sub.sort_values(sort_col, ascending=False).head(examples_per_type)
        sub["case_type"] = label
        candidates.append(sub)
    if not candidates:
        return pd.DataFrame()
    cols = [
        "case_type",
        "record_id",
        "method",
        "layer",
        "dataset",
        "domain",
        "target_any",
        "clean_any",
        "damage_any",
        "damage_first_any",
        "no_effect",
        "suppression_onset_alpha",
        "enhancement_onset_alpha",
        "negative_damage_onset_alpha",
        "positive_damage_onset_alpha",
        "target_delta_alpha_neg1",
        "target_delta_alpha_pos1",
        "oracle_best_utility",
        "feature_max_abs_cohen_d",
        "target_base_margin_mean",
        "neighbor_base_margin_mean",
        "capability_base_margin_mean",
    ]
    out = pd.concat(candidates, ignore_index=True)
    return out[[col for col in cols if col in out.columns]]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    outcome_df: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    pred_df: pd.DataFrame,
    cases: pd.DataFrame,
    manifest: Dict[str, Any],
) -> None:
    best_pred = pred_df.dropna(subset=["auroc"]).sort_values(["target", "auroc"], ascending=[True, False])
    best_pred = best_pred.groupby("target", as_index=False).head(3)
    lines = [
        "# PML Path Prediction Analysis",
        "",
        "## Inputs",
        "",
        "```text",
        f"selector_dataset_dir = {manifest['selector_dataset_dir']}",
        f"output_dir = {manifest['output_dir']}",
        "```",
        "",
        "## Dataset",
        "",
        f"- path rows: `{len(outcome_df)}`",
        f"- records: `{outcome_df['record_id'].nunique()}`",
        f"- methods: `{sorted(outcome_df['method'].dropna().unique().tolist())}`",
        f"- layers: `{sorted(pd.to_numeric(outcome_df['layer'], errors='coerce').dropna().astype(int).unique().tolist())}`",
        "",
        "## Method × Layer Outcome Summary",
        "",
        markdown_table(outcome_summary, max_rows=30),
        "",
        "## Best Prediction Rows",
        "",
        markdown_table(
            best_pred[
                [
                    "target",
                    "feature_set",
                    "model",
                    "test_positive_rate",
                    "auroc",
                    "auprc",
                    "n_features",
                ]
            ],
            max_rows=40,
        )
        if not best_pred.empty
        else "_No prediction rows._",
        "",
        "## Case Audit Sample",
        "",
        markdown_table(cases, max_rows=40),
        "",
        "## Interpretation",
        "",
        "- This analysis treats localization as a prediction problem: pre/intervention-cheap features are used to predict path outcomes.",
        "- Strong evidence for the PML claim requires B+M+L or B+M+L+V to improve over B/B+M for clean/damage/no-effect outcomes.",
        "- If only B+M+L+V+R improves, the conclusion should be that static localization is insufficient and early-response probing is needed.",
        "- These results should be read with random/null calibrated tau from the selector dataset manifest.",
        "",
    ]
    (out_dir / "PML_PATH_PREDICTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selector_dir = Path(args.selector_dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        "pml_path_outcome_dataset.csv",
        "pml_path_outcome_summary.csv",
        "pml_path_prediction_metrics.csv",
        "pml_path_case_audit.csv",
        "PML_PATH_PREDICTION_REPORT.md",
        "pml_path_prediction_manifest.json",
    ]
    if any((out_dir / name).exists() for name in expected) and not args.overwrite:
        raise FileExistsError(f"Outputs exist in {out_dir}. Use --overwrite.")

    path_path = selector_dir / "path_level_selector_dataset.csv"
    alpha_path = selector_dir / "alpha_level_selector_dataset.csv"
    if not path_path.exists() or not alpha_path.exists():
        raise FileNotFoundError(f"Missing selector dataset files in {selector_dir}")

    path_df = pd.read_csv(path_path)
    alpha_df = pd.read_csv(alpha_path)
    outcome_df = enrich_path_outcomes(path_df, alpha_df)
    outcome_summary = summarize_outcomes(outcome_df)
    prediction_metrics = run_prediction(outcome_df, args)
    case_audit = select_case_audit(outcome_df, args.case_examples_per_type)

    outcome_df.to_csv(out_dir / "pml_path_outcome_dataset.csv", index=False)
    outcome_summary.to_csv(out_dir / "pml_path_outcome_summary.csv", index=False)
    prediction_metrics.to_csv(out_dir / "pml_path_prediction_metrics.csv", index=False)
    case_audit.to_csv(out_dir / "pml_path_case_audit.csv", index=False)
    manifest = {
        "selector_dataset_dir": str(selector_dir),
        "output_dir": str(out_dir),
        "test_size": args.test_size,
        "seed": args.seed,
        "n_path_rows": int(len(outcome_df)),
        "n_records": int(outcome_df["record_id"].nunique()),
        "targets": TARGETS,
        "feature_groups": FEATURE_GROUPS,
        "ablations": ABLATIONS,
    }
    (out_dir / "pml_path_prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir, outcome_df, outcome_summary, prediction_metrics, case_audit, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
