from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_memory_localization.common import append_jsonl, load_jsonl


DEFAULT_DATASET_DIR = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b/prediction_dataset"
)
DEFAULT_OUTPUT_DIR = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b/prediction_analysis"
)

DEFAULT_TARGETS = [
    "suppression_success_medium",
    "enhancement_success_medium",
    "clean_suppression_medium",
    "clean_enhancement_medium",
    "target_any_path",
    "neighbor_damage_any_path",
    "capability_damage_any_path",
    "clean_any_path",
    "no_effect_path",
]
DEFAULT_SPLITS = ["record", "dataset", "domain"]
DEFAULT_COHORTS = ["all", "learned", "rfm"]
DEFAULT_MODELS = ["logistic", "random_forest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable grouped prediction ablations for the Activation-Space PML main experiment."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    parser.add_argument("--cohorts", nargs="*", default=DEFAULT_COHORTS)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--feature-sets", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def read_manifest(dataset_dir: Path) -> Dict[str, Any]:
    path = dataset_dir / "prediction_dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cohort_frame(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == "all":
        return frame[~frame["method"].eq("matched_norm_random")]
    if cohort == "learned":
        return frame[~frame["method"].isin(["random", "matched_norm_random"])]
    if cohort == "simple":
        return frame[frame["method"].isin(["mean_difference", "logistic", "linear"])]
    if cohort == "rfm":
        return frame[frame["method"].astype(str).str.startswith("rfm_agop_")]
    raise ValueError(f"Unknown cohort: {cohort}")


def default_feature_sets(cohort: str) -> List[str]:
    if cohort == "rfm":
        return ["B", "B+M", "B+M+L", "B+M+L+G", "B+M+L+G+R"]
    return ["B", "B+M", "B+M+L", "B+M+L+R"]


def columns_for_feature_set(
    feature_set: str, feature_groups: Dict[str, List[str]], columns: Sequence[str]
) -> List[str]:
    available = set(columns)
    selected: List[str] = []
    for group in feature_set.split("+"):
        selected.extend(column for column in feature_groups.get(group, []) if column in available)
    return list(dict.fromkeys(selected))


def categorical_columns(features: Sequence[str]) -> List[str]:
    categorical = {"method", "layer", "dataset", "domain", "freshness_group", "release_year"}
    return [feature for feature in features if feature in categorical]


def make_pipeline(
    frame: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    seed: int,
    rf_n_jobs: int,
) -> Pipeline:
    categorical = categorical_columns(features)
    numeric = [feature for feature in features if feature not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    if model_name == "logistic":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=rf_n_jobs,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def split_groups(frame: pd.DataFrame, split_name: str) -> pd.Series:
    if split_name == "record":
        return frame["record_id"].astype(str)
    if split_name in {"dataset", "domain", "method", "layer"}:
        return frame[split_name].astype(str)
    raise ValueError(f"Unknown split: {split_name}")


def prediction_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    predictions = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, scores)),
    }


def task_key(row: Dict[str, Any]) -> str:
    return "|".join(
        str(row[key])
        for key in ["cohort", "split", "target", "feature_set", "model", "fold"]
    )


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {task_key(row) for row in load_jsonl(path)}


def valid_target(frame: pd.DataFrame, target: str, min_positive: int) -> bool:
    if target not in frame.columns:
        return False
    values = pd.to_numeric(frame[target], errors="coerce").dropna().astype(int)
    if values.nunique() < 2:
        return False
    positives = int(values.sum())
    negatives = int(len(values) - positives)
    return positives >= min_positive and negatives >= min_positive


def run_tasks(frame: pd.DataFrame, manifest: Dict[str, Any], args: argparse.Namespace, results_path: Path) -> None:
    done = existing_keys(results_path)
    feature_groups = manifest["feature_groups"]
    for cohort in args.cohorts:
        cohort_df = cohort_frame(frame, cohort).copy()
        if cohort_df.empty:
            print(f"[skip] empty cohort={cohort}")
            continue
        requested_feature_sets = args.feature_sets or default_feature_sets(cohort)
        for split_name in args.splits:
            groups = split_groups(cohort_df, split_name)
            unique_groups = int(groups.nunique())
            if unique_groups < 2:
                print(f"[skip] cohort={cohort} split={split_name} unique_groups={unique_groups}")
                continue
            n_splits = min(args.n_splits, unique_groups)
            splitter = GroupKFold(n_splits=n_splits)
            for target in args.targets:
                if not valid_target(cohort_df, target, args.min_positive):
                    print(f"[skip] cohort={cohort} target={target} has insufficient class support")
                    continue
                target_values = pd.to_numeric(cohort_df[target], errors="coerce")
                valid_mask = target_values.notna()
                task_df = cohort_df.loc[valid_mask].reset_index(drop=True)
                y = target_values.loc[valid_mask].astype(int).to_numpy()
                task_groups = split_groups(task_df, split_name).to_numpy()
                fold_indices = list(splitter.split(task_df, y, task_groups))
                for feature_set in requested_feature_sets:
                    features = columns_for_feature_set(feature_set, feature_groups, task_df.columns)
                    if not features:
                        continue
                    for model_name in args.models:
                        for fold, (train_idx, test_idx) in enumerate(fold_indices):
                            identity = {
                                "cohort": cohort,
                                "split": split_name,
                                "target": target,
                                "feature_set": feature_set,
                                "model": model_name,
                                "fold": fold,
                            }
                            key = task_key(identity)
                            if key in done:
                                continue
                            y_train = y[train_idx]
                            y_test = y[test_idx]
                            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                                append_jsonl(
                                    results_path,
                                    {
                                        **identity,
                                        "status": "skipped_single_class_fold",
                                        "n_train": int(len(train_idx)),
                                        "n_test": int(len(test_idx)),
                                        "train_positive_rate": float(y_train.mean()),
                                        "test_positive_rate": float(y_test.mean()),
                                    },
                                )
                                done.add(key)
                                continue
                            pipeline = make_pipeline(
                                task_df,
                                features,
                                model_name,
                                args.seed + fold,
                                args.rf_n_jobs,
                            )
                            pipeline.fit(task_df.iloc[train_idx][features], y_train)
                            scores = pipeline.predict_proba(task_df.iloc[test_idx][features])[:, 1]
                            metrics = prediction_metrics(y_test, scores)
                            result = {
                                **identity,
                                "status": "completed",
                                "n_train": int(len(train_idx)),
                                "n_test": int(len(test_idx)),
                                "n_features": len(features),
                                "features": features,
                                "train_positive_rate": float(y_train.mean()),
                                "test_positive_rate": float(y_test.mean()),
                                **metrics,
                            }
                            append_jsonl(results_path, result)
                            done.add(key)
                            print(
                                f"[done] {key} auroc={metrics['auroc']:.4f} "
                                f"ap={metrics['average_precision']:.4f}"
                            )


def bootstrap_mean_ci(values: Sequence[float], seed: int, reps: int = 2000) -> tuple[float, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(reps, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_results(results_path: Path, output_dir: Path, seed: int) -> pd.DataFrame:
    rows = [row for row in load_jsonl(results_path) if row.get("status") == "completed"]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    group_columns = ["cohort", "split", "target", "feature_set", "model"]
    summaries: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_folds"] = int(len(group))
        row["n_test_total"] = int(group["n_test"].sum())
        row["positive_rate_mean"] = float(group["test_positive_rate"].mean())
        for metric in ["auroc", "average_precision", "balanced_accuracy", "f1", "brier"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            row[f"{metric}_mean"] = float(np.mean(values)) if values else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            low, high = bootstrap_mean_ci(values, seed)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    baseline = summary[summary["feature_set"] == "B+M"][
        ["cohort", "split", "target", "model", "auroc_mean", "average_precision_mean"]
    ].rename(
        columns={
            "auroc_mean": "baseline_auroc_mean",
            "average_precision_mean": "baseline_average_precision_mean",
        }
    )
    summary = summary.merge(baseline, on=["cohort", "split", "target", "model"], how="left")
    summary["delta_auroc_vs_BM"] = summary["auroc_mean"] - summary["baseline_auroc_mean"]
    summary["delta_ap_vs_BM"] = (
        summary["average_precision_mean"] - summary["baseline_average_precision_mean"]
    )
    summary = summary.sort_values(group_columns).reset_index(drop=True)
    summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    return summary


def outcome_summary(
    frame: pd.DataFrame, output_dir: Path, requested_targets: Sequence[str]
) -> pd.DataFrame:
    targets = [target for target in requested_targets if target in frame.columns]
    rows = []
    for (method, layer), group in frame.groupby(["method", "layer"]):
        row: Dict[str, Any] = {"method": method, "layer": layer, "n_rows": int(len(group))}
        for target in targets:
            row[f"{target}_rate"] = float(pd.to_numeric(group[target], errors="coerce").mean())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["method", "layer"])
    summary.to_csv(output_dir / "outcome_summary.csv", index=False)
    return summary


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], max_rows: int = 80) -> str:
    subset = frame[[column for column in columns if column in frame.columns]].head(max_rows)
    if subset.empty:
        return "_No rows._"
    headers = subset.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in subset.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    frame: pd.DataFrame,
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: Dict[str, Any],
) -> None:
    primary = summary[
        (summary["split"] == "record")
        & (summary["model"] == "logistic")
        & (
            summary["feature_set"].isin(
                ["M+R", "B+M", "B+M+R", "B+M+L", "B+M+L+G", "B+M+L+R", "B+M+L+G+R"]
            )
        )
    ].copy()
    primary = primary.sort_values(["cohort", "target", "feature_set"])
    strict_later = bool(manifest.get("later_alphas"))
    title = (
        "# Activation-Space PML Strict Later-Outcome Prediction Report"
        if strict_later
        else "# Activation-Space PML Main Prediction Report"
    )
    lines = [
        title,
        "",
        f"- rows: `{len(frame)}`",
        f"- unique records: `{frame['record_id'].nunique()}`",
        f"- methods: `{sorted(frame['method'].unique().tolist())}`",
        f"- layers: `{sorted(int(value) for value in frame['layer'].unique())}`",
        *(
            [
                f"- early response alpha: `{manifest.get('early_alpha')}`",
                f"- strict later label alphas: `{manifest.get('later_alphas')}`",
                "- leakage guard: later labels exclude the early response strength",
            ]
            if strict_later
            else []
        ),
        "",
        "## Outcome Distribution",
        "",
        markdown_table(outcome, ["method", "layer", "n_rows"] + [column for column in outcome if column.endswith("_rate")], 40),
        "",
        "## Primary Record-Held-Out Ablation",
        "",
        markdown_table(
            primary,
            [
                "cohort",
                "target",
                "feature_set",
                "n_folds",
                "positive_rate_mean",
                "auroc_mean",
                "auroc_ci_low",
                "auroc_ci_high",
                "average_precision_mean",
                "delta_auroc_vs_BM",
                "delta_ap_vs_BM",
            ],
            160,
        ),
        "",
        "## Interpretation Rules",
        "",
        "- `B+M+L > B+M`: localization features add held-out predictive information.",
        "- `B+M+L+G > B+M+L` inside the RFM cohort: AGOP geometry adds within-method information.",
        "- `+R` is a secondary diagnostic using small-strength response; primary PML claims should use pre-intervention groups.",
        "- Dataset/domain grouped results test source transfer and should be read with their class prevalence.",
        "",
    ]
    (output_dir / "PML_MAIN_PREDICTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def validate_resume_config(path: Path, config: Dict[str, Any], overwrite: bool) -> None:
    if overwrite or not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    fixed_keys = [
        "dataset_manifest",
        "targets",
        "splits",
        "cohorts",
        "models",
        "feature_sets",
        "n_splits",
        "rf_n_jobs",
        "seed",
        "min_positive",
        "max_records",
    ]
    mismatches = {
        key: {"previous": previous.get(key), "current": config.get(key)}
        for key in fixed_keys
        if previous.get(key) != config.get(key)
    }
    if mismatches:
        raise ValueError(
            "Prediction resume configuration mismatch. Use a new output directory or --overwrite. "
            f"Mismatches: {mismatches}"
        )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "fold_results.jsonl"
    if args.overwrite:
        for name in [
            "fold_results.jsonl",
            "prediction_summary.csv",
            "outcome_summary.csv",
            "PML_MAIN_PREDICTION_REPORT.md",
            "prediction_run_config.json",
        ]:
            (output_dir / name).unlink(missing_ok=True)
    manifest = read_manifest(dataset_dir)
    frame = pd.read_csv(dataset_dir / "pml_path_prediction_dataset.csv")
    if args.max_records is not None and frame["record_id"].nunique() > args.max_records:
        rng = np.random.default_rng(args.seed)
        record_ids = frame["record_id"].drop_duplicates().to_numpy()
        selected = set(rng.choice(record_ids, size=args.max_records, replace=False).tolist())
        frame = frame[frame["record_id"].isin(selected)].reset_index(drop=True)
    config = {
        **vars(args),
        "dataset_manifest": str(dataset_dir / "prediction_dataset_manifest.json"),
        "n_rows": int(len(frame)),
        "n_records": int(frame["record_id"].nunique()),
    }
    validate_resume_config(output_dir / "prediction_run_config.json", config, args.overwrite)
    (output_dir / "prediction_run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_tasks(frame, manifest, args, results_path)
    outcome = outcome_summary(frame, output_dir, args.targets)
    summary = summarize_results(results_path, output_dir, args.seed)
    if not summary.empty:
        write_report(output_dir, frame, outcome, summary, manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_rows": int(len(frame)),
                "n_records": int(frame["record_id"].nunique()),
                "n_completed_folds": sum(1 for row in load_jsonl(results_path) if row.get("status") == "completed"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
