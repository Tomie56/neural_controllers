from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TRAIN_DATASET = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2_bidirectional_pilot_500/qwen3_1_7b/selector_dataset"
)
DEFAULT_OUTPUT_DIR = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2_strength_selector_models/qwen3_1_7b"
)


FEATURE_SETS = {
    "L": [
        "method",
        "layer",
        "feature_max_abs_cohen_d",
        "feature_mean_abs_cohen_d",
        "feature_mean_threshold_accuracy",
        "feature_saliency_concentration",
        "feature_saliency_entropy",
        "feature_top_saliency_layer",
    ],
    "LB": [
        "method",
        "layer",
        "dataset",
        "domain",
        "freshness_group",
        "release_year",
        "feature_max_abs_cohen_d",
        "feature_mean_abs_cohen_d",
        "feature_mean_threshold_accuracy",
        "feature_saliency_concentration",
        "feature_saliency_entropy",
        "feature_top_saliency_layer",
        "target_base_margin_mean",
        "target_base_margin_median",
        "target_base_margin_min",
        "neighbor_base_margin_mean",
        "neighbor_base_margin_median",
        "capability_base_margin_mean",
        "capability_base_margin_median",
        "target_base_positive_rate",
        "neighbor_base_positive_rate",
        "capability_base_positive_rate",
        "target_neighbor_base_margin_gap",
        "target_capability_base_margin_gap",
    ],
    "LBA": [
        "method",
        "layer",
        "dataset",
        "domain",
        "freshness_group",
        "release_year",
        "feature_max_abs_cohen_d",
        "feature_mean_abs_cohen_d",
        "feature_mean_threshold_accuracy",
        "feature_saliency_concentration",
        "feature_saliency_entropy",
        "feature_top_saliency_layer",
        "target_base_margin_mean",
        "target_base_margin_median",
        "target_base_margin_min",
        "neighbor_base_margin_mean",
        "neighbor_base_margin_median",
        "capability_base_margin_mean",
        "capability_base_margin_median",
        "target_base_positive_rate",
        "neighbor_base_positive_rate",
        "capability_base_positive_rate",
        "target_neighbor_base_margin_gap",
        "target_capability_base_margin_gap",
        "alpha",
        "abs_alpha",
        "alpha_sign",
        "alpha_squared",
    ],
}

EARLY_FEATURE_PREFIXES = (
    "early_target_delta_",
    "early_neighbor_delta_",
    "early_capability_delta_",
    "early_target_slope",
    "early_neighbor_slope",
    "early_capability_slope",
    "early_target_sign_consistency",
    "early_damage_flag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight PML strength selectors.")
    parser.add_argument("--train-dataset-dir", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--validation-dataset-dir", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-sets", nargs="*", default=["L", "LB", "LBA", "LBAR"])
    parser.add_argument(
        "--models",
        nargs="*",
        default=["shared_logistic", "shared_gbdt", "per_layer_logistic", "per_layer_gbdt"],
    )
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_alpha_dataset(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "alpha_level_selector_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["direction"].isin(["suppression", "enhancement"])].copy()
    for col in ["is_clean", "target_success", "neighbor_damage", "capability_damage", "any_damage"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    return df


def feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    if feature_set == "LBAR":
        cols = list(FEATURE_SETS["LBA"])
        cols.extend([c for c in df.columns if c.startswith(EARLY_FEATURE_PREFIXES)])
    else:
        cols = list(FEATURE_SETS[feature_set])
    return [c for c in cols if c in df.columns]


def make_pipeline(df: pd.DataFrame, features: Sequence[str], model_name: str) -> Pipeline:
    categorical = [c for c in features if df[c].dtype == object or c in {"method", "dataset", "domain", "freshness_group"}]
    numeric = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )
    if model_name.endswith("logistic"):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
    elif model_name.endswith("gbdt"):
        clf = GradientBoostingClassifier(random_state=113)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def split_by_record(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = sorted(df["record_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    train_ids = set(records[: int(0.70 * n)])
    dev_ids = set(records[int(0.70 * n) : int(0.85 * n)])
    test_ids = set(records[int(0.85 * n) :])
    return (
        df[df["record_id"].isin(train_ids)].copy(),
        df[df["record_id"].isin(dev_ids)].copy(),
        df[df["record_id"].isin(test_ids)].copy(),
    )


def binary_metrics(y_true: Sequence[bool], score: Sequence[float]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    out: Dict[str, Any] = {"n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else 0.0}
    if len(set(y.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def candidate_eval(df: pd.DataFrame, score_col: str, model_label: str, split_label: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    groups = df.groupby(["record_id", "method", "layer", "direction"], dropna=False)
    for key, group in groups:
        if group.empty:
            continue
        selected = group.sort_values(score_col, ascending=False).iloc[0]
        oracle = group.sort_values("utility", ascending=False).iloc[0]
        selected_rows.append(
            {
                "model": model_label,
                "split": split_label,
                "record_id": key[0],
                "method": key[1],
                "layer": key[2],
                "direction": key[3],
                "selected_alpha": selected["alpha"],
                "selected_score": selected[score_col],
                "selected_is_clean": bool(selected["is_clean"]),
                "selected_target_success": bool(selected["target_success"]),
                "selected_neighbor_damage": bool(selected["neighbor_damage"]),
                "selected_capability_damage": bool(selected["capability_damage"]),
                "selected_utility": selected["utility"],
                "oracle_alpha": oracle["alpha"],
                "oracle_is_clean": bool(oracle["is_clean"]),
                "oracle_utility": oracle["utility"],
                "regret": oracle["utility"] - selected["utility"],
            }
        )
    if selected_rows:
        for direction in ["suppression", "enhancement"]:
            rows = [r for r in selected_rows if r["direction"] == direction]
            if not rows:
                continue
            summary_rows.append(
                {
                    "model": model_label,
                    "split": split_label,
                    "direction": direction,
                    "n_groups": len(rows),
                    "selected_clean_rate": float(np.mean([r["selected_is_clean"] for r in rows])),
                    "selected_target_success_rate": float(np.mean([r["selected_target_success"] for r in rows])),
                    "selected_neighbor_damage_rate": float(np.mean([r["selected_neighbor_damage"] for r in rows])),
                    "selected_capability_damage_rate": float(np.mean([r["selected_capability_damage"] for r in rows])),
                    "selected_utility_mean": float(np.mean([r["selected_utility"] for r in rows])),
                    "oracle_clean_rate": float(np.mean([r["oracle_is_clean"] for r in rows])),
                    "oracle_utility_mean": float(np.mean([r["oracle_utility"] for r in rows])),
                    "regret_mean": float(np.mean([r["regret"] for r in rows])),
                }
            )
    return selected_rows, summary_rows


def fixed_alpha_baselines(df: pd.DataFrame, split_label: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    fixed = {"suppression": [-0.1, -0.5, -1.0], "enhancement": [0.1, 0.5, 1.0]}
    groups = df.groupby(["record_id", "method", "layer", "direction"], dropna=False)
    for fixed_label, alpha_values in fixed.items():
        for alpha in alpha_values:
            selected = []
            for key, group in groups:
                if key[3] != fixed_label:
                    continue
                match = group[np.isclose(group["alpha"].astype(float), alpha)]
                if match.empty:
                    continue
                row = match.iloc[0]
                oracle = group.sort_values("utility", ascending=False).iloc[0]
                selected.append(
                    {
                        "model": f"fixed_alpha_{alpha:g}",
                        "split": split_label,
                        "record_id": key[0],
                        "method": key[1],
                        "layer": key[2],
                        "direction": key[3],
                        "selected_alpha": row["alpha"],
                        "selected_is_clean": bool(row["is_clean"]),
                        "selected_target_success": bool(row["target_success"]),
                        "selected_neighbor_damage": bool(row["neighbor_damage"]),
                        "selected_capability_damage": bool(row["capability_damage"]),
                        "selected_utility": row["utility"],
                        "oracle_alpha": oracle["alpha"],
                        "oracle_is_clean": bool(oracle["is_clean"]),
                        "oracle_utility": oracle["utility"],
                        "regret": oracle["utility"] - row["utility"],
                    }
                )
            rows.extend(selected)
            if selected:
                summaries.append(
                    {
                        "model": f"fixed_alpha_{alpha:g}",
                        "split": split_label,
                        "direction": fixed_label,
                        "n_groups": len(selected),
                        "selected_clean_rate": float(np.mean([r["selected_is_clean"] for r in selected])),
                        "selected_target_success_rate": float(np.mean([r["selected_target_success"] for r in selected])),
                        "selected_neighbor_damage_rate": float(np.mean([r["selected_neighbor_damage"] for r in selected])),
                        "selected_capability_damage_rate": float(np.mean([r["selected_capability_damage"] for r in selected])),
                        "selected_utility_mean": float(np.mean([r["selected_utility"] for r in selected])),
                        "oracle_clean_rate": float(np.mean([r["oracle_is_clean"] for r in selected])),
                        "oracle_utility_mean": float(np.mean([r["oracle_utility"] for r in selected])),
                        "regret_mean": float(np.mean([r["regret"] for r in selected])),
                    }
                )
    return rows, summaries


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_and_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    split_label: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_pred_rows: List[Dict[str, Any]] = []
    all_selected: List[Dict[str, Any]] = []
    all_summary: List[Dict[str, Any]] = []
    if model_name.startswith("per_layer_"):
        layers = sorted(train_df["layer"].unique())
        for layer in layers:
            tr = train_df[train_df["layer"] == layer].copy()
            ev = eval_df[eval_df["layer"] == layer].copy()
            if tr.empty or ev.empty or tr["is_clean"].nunique() < 2:
                continue
            features = feature_columns(tr, feature_set)
            pipeline = make_pipeline(tr, features, model_name)
            pipeline.fit(tr[features], tr["is_clean"].astype(bool))
            ev = ev.copy()
            ev["selector_score"] = pipeline.predict_proba(ev[features])[:, 1]
            label = f"{model_name}_{feature_set}_layer_{layer}"
            metric = binary_metrics(ev["is_clean"], ev["selector_score"])
            metric.update({"model": label, "split": split_label, "feature_set": feature_set, "layer": layer})
            all_summary.append(metric)
            selected, selected_summary = candidate_eval(ev, "selector_score", label, split_label)
            all_selected.extend(selected)
            all_summary.extend(selected_summary)
    else:
        if train_df["is_clean"].nunique() < 2:
            return all_pred_rows, all_selected, all_summary
        features = feature_columns(train_df, feature_set)
        pipeline = make_pipeline(train_df, features, model_name)
        pipeline.fit(train_df[features], train_df["is_clean"].astype(bool))
        ev = eval_df.copy()
        ev["selector_score"] = pipeline.predict_proba(ev[features])[:, 1]
        label = f"{model_name}_{feature_set}"
        metric = binary_metrics(ev["is_clean"], ev["selector_score"])
        metric.update({"model": label, "split": split_label, "feature_set": feature_set, "layer": "shared"})
        all_summary.append(metric)
        selected, selected_summary = candidate_eval(ev, "selector_score", label, split_label)
        all_selected.extend(selected)
        all_summary.extend(selected_summary)
    return all_pred_rows, all_selected, all_summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = ["selector_model_summary.csv", "selector_selected_alpha_rows.csv", "selector_train_config.json"]
    if any((out_dir / name).exists() for name in outputs) and not args.overwrite:
        raise FileExistsError(f"Outputs exist in {out_dir}. Use --overwrite.")
    if args.overwrite:
        for name in outputs:
            (out_dir / name).unlink(missing_ok=True)

    train_all = read_alpha_dataset(Path(args.train_dataset_dir))
    train_df, dev_df, internal_test_df = split_by_record(train_all, args.seed)
    eval_splits = [("dev", dev_df), ("internal_test", internal_test_df)]
    if args.validation_dataset_dir:
        validation_df = read_alpha_dataset(Path(args.validation_dataset_dir))
        eval_splits.append(("external_validation", validation_df))

    summary_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    for split_label, eval_df in eval_splits:
        baseline_selected, baseline_summary = fixed_alpha_baselines(eval_df, split_label)
        selected_rows.extend(baseline_selected)
        summary_rows.extend(baseline_summary)
        for feature_set in args.feature_sets:
            if feature_set not in {"L", "LB", "LBA", "LBAR"}:
                raise ValueError(f"Unknown feature set: {feature_set}")
            for model_name in args.models:
                _, selected, summary = fit_and_eval(train_df, eval_df, feature_set, model_name, split_label)
                selected_rows.extend(selected)
                summary_rows.extend(summary)

    write_csv(out_dir / "selector_model_summary.csv", summary_rows)
    write_csv(out_dir / "selector_selected_alpha_rows.csv", selected_rows)
    config = {
        "train_dataset_dir": args.train_dataset_dir,
        "validation_dataset_dir": args.validation_dataset_dir,
        "output_dir": str(out_dir),
        "feature_sets": args.feature_sets,
        "models": args.models,
        "seed": args.seed,
        "n_train_rows": len(train_df),
        "n_dev_rows": len(dev_df),
        "n_internal_test_rows": len(internal_test_df),
        "n_summary_rows": len(summary_rows),
        "n_selected_rows": len(selected_rows),
    }
    (out_dir / "selector_train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
