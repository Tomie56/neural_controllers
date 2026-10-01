from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


BOOLEAN_COLUMNS = ["is_clean", "target_success", "neighbor_damage", "capability_damage", "any_damage"]
METADATA_COLUMNS = ["method", "layer", "dataset", "domain", "freshness_group", "release_year"]
ALPHA_COLUMNS = ["alpha", "abs_alpha", "alpha_sign", "alpha_squared"]
EARLY_COLUMNS = [
    "early_target_delta_neg_0.1",
    "early_target_delta_pos_0.1",
    "early_neighbor_delta_neg_0.1",
    "early_neighbor_delta_pos_0.1",
    "early_capability_delta_neg_0.1",
    "early_capability_delta_pos_0.1",
    "early_target_slope_mean",
    "early_neighbor_slope_mean",
    "early_capability_slope_mean",
    "early_target_sign_consistency",
    "early_damage_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train resumable multi-fidelity dense-grid strength predictors.")
    parser.add_argument("--dense-train-dataset-dir", required=True)
    parser.add_argument("--multifidelity-train-dataset-dir", required=True)
    parser.add_argument("--validation-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-sets", nargs="*", default=["BMA", "MAR", "BMAR", "BMLAR"])
    parser.add_argument("--models", nargs="*", default=["linear", "hist_gbdt"])
    parser.add_argument("--clean-bonus", type=float, default=0.1)
    parser.add_argument("--alpha-penalty", type=float, default=0.01)
    parser.add_argument("--abstain-score", type=float, default=0.0)
    parser.add_argument("--linear-n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_dataset(directory: Path) -> pd.DataFrame:
    path = directory / "alpha_level_selector_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame = frame[frame["direction"].isin(["suppression", "enhancement"])].copy()
    frame["record_id"] = frame["record_id"].astype(str)
    for column in BOOLEAN_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.lower().isin(["true", "1"])
    if "source_weight" not in frame.columns:
        frame["source_weight"] = 1.0
    return frame


def feature_groups(frame: pd.DataFrame) -> Dict[str, List[str]]:
    baseline = [
        column
        for column in frame.columns
        if column.startswith("target_base_margin_")
        or column.startswith("neighbor_base_margin_")
        or column.startswith("capability_base_margin_")
        or column.endswith("_base_positive_rate")
        or column in {"target_neighbor_base_margin_gap", "target_capability_base_margin_gap"}
    ]
    geometry = [column for column in frame.columns if column.startswith("feature_agop_")]
    localization = [
        column
        for column in frame.columns
        if column.startswith("feature_") and not column.startswith("feature_agop_")
    ]
    available = set(frame.columns)
    return {
        "M": [column for column in METADATA_COLUMNS if column in available],
        "B": sorted(baseline),
        "L": sorted(localization),
        "G": sorted(geometry),
        "A": [column for column in ALPHA_COLUMNS if column in available],
        "R": [column for column in EARLY_COLUMNS if column in available],
    }


def columns_for_feature_set(frame: pd.DataFrame, feature_set: str) -> List[str]:
    groups = feature_groups(frame)
    mapping = {
        "BMA": ["B", "M", "A"],
        "MAR": ["M", "A", "R"],
        "BMAR": ["B", "M", "A", "R"],
        "BMLAR": ["B", "M", "L", "A", "R"],
    }
    if feature_set not in mapping:
        raise ValueError(f"Unknown feature set: {feature_set}")
    columns: List[str] = []
    for group in mapping[feature_set]:
        columns.extend(groups[group])
    return list(dict.fromkeys(columns))


def categorical_columns(frame: pd.DataFrame, features: Sequence[str]) -> List[str]:
    categorical_names = {"method", "layer", "dataset", "domain", "freshness_group", "release_year"}
    return [column for column in features if column in categorical_names or frame[column].dtype == object]


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_models(
    frame: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    seed: int,
    linear_n_jobs: int,
) -> tuple[Pipeline, Pipeline]:
    categorical = categorical_columns(frame, features)
    numeric = [column for column in features if column not in categorical]
    if model_name == "linear":
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]),
                    categorical,
                ),
            ]
        )
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight=None,
            solver="saga",
            n_jobs=linear_n_jobs,
            random_state=seed,
        )
        regressor = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=1e-4,
            max_iter=3000,
            random_state=seed,
        )
    elif model_name == "hist_gbdt":
        preprocessor = ColumnTransformer(
            [
                ("numeric", SimpleImputer(strategy="median"), numeric),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "ordinal",
                                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            ),
                        ]
                    ),
                    categorical,
                ),
            ],
            sparse_threshold=0.0,
        )
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed,
        )
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("model", classifier)]), Pipeline(
        [("preprocess", clone(preprocessor)), ("model", regressor)]
    )


def classifier_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["is_clean"].astype(int).to_numpy()
    positives = max(int(labels.sum()), 1)
    negatives = max(int(len(labels) - labels.sum()), 1)
    class_weights = np.where(labels == 1, len(labels) / (2 * positives), len(labels) / (2 * negatives))
    return frame["source_weight"].astype(float).to_numpy() * class_weights


def safe_metric(metric: str, labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    if metric == "average_precision":
        return float(average_precision_score(labels, scores))
    raise ValueError(metric)


def choose_candidates(
    frame: pd.DataFrame,
    task_label: str,
    clean_bonus: float,
    alpha_penalty: float,
    abstain_score: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["selection_score"] = (
        frame["predicted_utility"]
        + clean_bonus * frame["predicted_clean_probability"]
        - alpha_penalty * frame["abs_alpha"].astype(float)
    )
    selected_rows: List[Dict[str, Any]] = []
    groups = frame.groupby(["record_id", "method", "layer", "direction"], dropna=False)
    for key, group in groups:
        selected = group.sort_values(["selection_score", "abs_alpha"], ascending=[False, True]).iloc[0]
        oracle = group.sort_values(["utility", "abs_alpha"], ascending=[False, True]).iloc[0]
        abstain = float(selected["selection_score"]) <= abstain_score
        oracle_abstain = float(oracle["utility"]) <= 0.0
        selected_rows.append(
            {
                "task": task_label,
                "record_id": key[0],
                "method": key[1],
                "layer": key[2],
                "direction": key[3],
                "selected_alpha": 0.0 if abstain else float(selected["alpha"]),
                "selected_abs_alpha": 0.0 if abstain else abs(float(selected["alpha"])),
                "selected_clean": False if abstain else bool(selected["is_clean"]),
                "selected_target_success": False if abstain else bool(selected["target_success"]),
                "selected_neighbor_damage": False if abstain else bool(selected["neighbor_damage"]),
                "selected_capability_damage": False if abstain else bool(selected["capability_damage"]),
                "selected_utility": 0.0 if abstain else float(selected["utility"]),
                "selected_score": float(selected["selection_score"]),
                "abstained": abstain,
                "oracle_alpha": 0.0 if oracle_abstain else float(oracle["alpha"]),
                "oracle_clean": False if oracle_abstain else bool(oracle["is_clean"]),
                "oracle_utility": 0.0 if oracle_abstain else float(oracle["utility"]),
                "regret": (0.0 if oracle_abstain else float(oracle["utility"]))
                - (0.0 if abstain else float(selected["utility"])),
            }
        )
    return pd.DataFrame(selected_rows)


def summarize_selected(selected: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for direction, group in selected.groupby("direction"):
        rows.append(
            {
                **metadata,
                "direction": direction,
                "n_groups": len(group),
                "clean_rate": float(group["selected_clean"].mean()),
                "target_success_rate": float(group["selected_target_success"].mean()),
                "neighbor_damage_rate": float(group["selected_neighbor_damage"].mean()),
                "capability_damage_rate": float(group["selected_capability_damage"].mean()),
                "utility_mean": float(group["selected_utility"].mean()),
                "regret_mean": float(group["regret"].mean()),
                "abstention_rate": float(group["abstained"].mean()),
                "mean_abs_alpha": float(group["selected_abs_alpha"].mean()),
                "oracle_clean_rate": float(group["oracle_clean"].mean()),
                "oracle_utility_mean": float(group["oracle_utility"].mean()),
            }
        )
    return rows


def fixed_baselines(validation: pd.DataFrame) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for direction, signed_values in {"suppression": [-0.25, -0.5, -1.0], "enhancement": [0.25, 0.5, 1.0]}.items():
        direction_frame = validation[validation["direction"] == direction]
        for alpha in signed_values:
            rows = direction_frame[np.isclose(direction_frame["alpha"].astype(float), alpha)].copy()
            if rows.empty:
                continue
            summary.append(
                {
                    "regime": "fixed",
                    "feature_set": "none",
                    "model": f"fixed_{alpha:g}",
                    "direction": direction,
                    "n_groups": len(rows),
                    "clean_rate": float(rows["is_clean"].mean()),
                    "target_success_rate": float(rows["target_success"].mean()),
                    "neighbor_damage_rate": float(rows["neighbor_damage"].mean()),
                    "capability_damage_rate": float(rows["capability_damage"].mean()),
                    "utility_mean": float(rows["utility"].mean()),
                    "regret_mean": None,
                    "abstention_rate": 0.0,
                    "mean_abs_alpha": abs(alpha),
                    "oracle_clean_rate": None,
                    "oracle_utility_mean": None,
                }
            )
    return summary


def write_report(path: Path, summary: pd.DataFrame, config: Dict[str, Any]) -> None:
    learned = summary[summary["regime"].isin(["dense_only", "multifidelity"])].copy()
    learned = learned.sort_values(["utility_mean", "clean_rate"], ascending=False)
    lines = [
        "# Multi-Fidelity Dense Strength Predictor",
        "",
        f"- dense train records: `{config['dense_train_records']}`",
        f"- multifidelity train records: `{config['multifidelity_train_records']}`",
        f"- dense held-out validation records: `{config['validation_records']}`",
        f"- validation overlap: `{config['validation_overlap_count']}`",
        "",
        "## Best Validation Configurations",
        "",
        "| regime | feature set | model | direction | clean rate | utility | regret | abstention | mean abs alpha |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in learned.head(16).iterrows():
        lines.append(
            f"| {row['regime']} | {row['feature_set']} | {row['model']} | {row['direction']} | "
            f"{row['clean_rate']:.4f} | {row['utility_mean']:.4f} | {row['regret_mean']:.4f} | "
            f"{row['abstention_rate']:.4f} | {row['mean_abs_alpha']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The held-out validation set is never included in dense or sparse training.",
            "Sparse auxiliary trajectories are reported separately from dense scans.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tasks_dir = output_dir / "tasks"
    if args.overwrite and output_dir.exists():
        if tasks_dir.exists():
            shutil.rmtree(tasks_dir)
        for name in [
            "strength_selector_summary.csv",
            "selected_alpha_rows.csv",
            "run_request.json",
            "run_config.json",
            "STRENGTH_PREDICTOR_REPORT.md",
        ]:
            (output_dir / name).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    request_config = {key: value for key, value in vars(args).items() if key != "overwrite"}
    request_path = output_dir / "run_request.json"
    if request_path.exists() and not args.overwrite:
        previous_request = json.loads(request_path.read_text(encoding="utf-8"))
        if previous_request != request_config:
            raise ValueError(
                f"Resume configuration differs from {request_path}. "
                "Use the original arguments or pass --overwrite."
            )
    else:
        request_path.write_text(json.dumps(request_config, ensure_ascii=False, indent=2), encoding="utf-8")

    dense_train = read_dataset(Path(args.dense_train_dataset_dir))
    multifidelity_train = read_dataset(Path(args.multifidelity_train_dataset_dir))
    validation = read_dataset(Path(args.validation_dataset_dir))
    validation_ids = set(validation["record_id"])
    overlap = set(multifidelity_train["record_id"]) & validation_ids
    if overlap:
        raise ValueError(f"Validation record leakage detected: {len(overlap)} records")

    regimes = {"dense_only": dense_train, "multifidelity": multifidelity_train}
    for regime, train in regimes.items():
        for feature_set in args.feature_sets:
            for model_name in args.models:
                task_label = f"{regime}__{feature_set}__{model_name}"
                task_dir = tasks_dir / task_label
                metrics_path = task_dir / "metrics.json"
                selected_path = task_dir / "selected_alpha_rows.csv"
                clean_model_path = task_dir / "clean_model.joblib"
                utility_model_path = task_dir / "utility_model.joblib"
                task_outputs = [metrics_path, selected_path, clean_model_path, utility_model_path]
                if all(path.exists() for path in task_outputs) and not args.overwrite:
                    print(f"[skip] {task_label}")
                    continue
                task_dir.mkdir(parents=True, exist_ok=True)
                for path in task_outputs:
                    path.unlink(missing_ok=True)
                features = columns_for_feature_set(train, feature_set)
                if not features:
                    raise ValueError(f"No usable features for {task_label}")
                missing_validation = [column for column in features if column not in validation.columns]
                if missing_validation:
                    raise ValueError(f"Validation missing features for {task_label}: {missing_validation}")
                clean_model, utility_model = make_models(train, features, model_name, args.seed, args.linear_n_jobs)
                clean_model.fit(
                    train[features],
                    train["is_clean"].astype(int),
                    model__sample_weight=classifier_weights(train),
                )
                utility_model.fit(
                    train[features],
                    train["utility"].astype(float),
                    model__sample_weight=train["source_weight"].astype(float).to_numpy(),
                )
                predicted = validation.copy()
                predicted["predicted_clean_probability"] = clean_model.predict_proba(validation[features])[:, 1]
                predicted["predicted_utility"] = utility_model.predict(validation[features])
                selected = choose_candidates(
                    predicted,
                    task_label,
                    args.clean_bonus,
                    args.alpha_penalty,
                    args.abstain_score,
                )
                selected.to_csv(selected_path, index=False)
                labels = validation["is_clean"].astype(int).to_numpy()
                metrics = {
                    "task": task_label,
                    "regime": regime,
                    "feature_set": feature_set,
                    "model": model_name,
                    "n_train_rows": len(train),
                    "n_train_records": int(train["record_id"].nunique()),
                    "n_validation_rows": len(validation),
                    "n_validation_records": int(validation["record_id"].nunique()),
                    "n_features": len(features),
                    "features": features,
                    "clean_auroc": safe_metric(
                        "auroc", labels, predicted["predicted_clean_probability"].to_numpy()
                    ),
                    "clean_average_precision": safe_metric(
                        "average_precision", labels, predicted["predicted_clean_probability"].to_numpy()
                    ),
                    "utility_mae": float(
                        mean_absolute_error(validation["utility"], predicted["predicted_utility"])
                    ),
                    "utility_r2": float(r2_score(validation["utility"], predicted["predicted_utility"])),
                    "selection_summary": summarize_selected(
                        selected,
                        {"regime": regime, "feature_set": feature_set, "model": model_name},
                    ),
                }
                joblib.dump(clean_model, clean_model_path)
                joblib.dump(utility_model, utility_model_path)
                metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[done] {task_label}")

    summary_rows = fixed_baselines(validation)
    selected_frames: List[pd.DataFrame] = []
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        metrics_path = task_dir / "metrics.json"
        selected_path = task_dir / "selected_alpha_rows.csv"
        if not metrics_path.exists() or not selected_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary_rows.extend(metrics["selection_summary"])
        selected_frames.append(pd.read_csv(selected_path))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "strength_selector_summary.csv", index=False)
    if selected_frames:
        pd.concat(selected_frames, ignore_index=True).to_csv(output_dir / "selected_alpha_rows.csv", index=False)

    config = {
        **vars(args),
        "dense_train_records": int(dense_train["record_id"].nunique()),
        "multifidelity_train_records": int(multifidelity_train["record_id"].nunique()),
        "validation_records": int(validation["record_id"].nunique()),
        "validation_overlap_count": len(overlap),
        "dense_train_rows": len(dense_train),
        "multifidelity_train_rows": len(multifidelity_train),
        "validation_rows": len(validation),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "STRENGTH_PREDICTOR_REPORT.md", summary, config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
