from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

from predictive_memory_localization.analysis.train_multifidelity_dense_strength_predictor import (
    columns_for_feature_set,
    make_models,
    read_dataset,
)


DEFAULT_MAIN_ROOT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b"
)
DEFAULT_DENSE_TRAIN = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2_bidirectional_pilot_500/qwen3_1_7b/selector_dataset_tau_main_q95"
)
DEFAULT_VALIDATION = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2b_selector_validation_100/qwen3_1_7b/selector_dataset_tau_main_q95"
)
SCENARIOS = [
    ("lower_damage", 0.5, 0.0, 0.1, 0.01),
    ("no_clean_bonus", 1.0, 0.0, 0.0, 0.01),
    ("no_strength_penalty", 1.0, 0.0, 0.1, 0.0),
    ("declared", 1.0, 0.0, 0.1, 0.01),
    ("higher_clean_bonus", 1.0, 0.0, 0.2, 0.01),
    ("higher_strength_penalty", 1.0, 0.0, 0.1, 0.02),
    ("higher_damage", 2.0, 0.0, 0.1, 0.01),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out selector sensitivity to declared utility weights."
    )
    parser.add_argument(
        "--multifidelity-train-dataset-dir",
        default=str(DEFAULT_MAIN_ROOT / "strength_selector_multifidelity_dataset"),
    )
    parser.add_argument("--dense-train-dataset-dir", default=str(DEFAULT_DENSE_TRAIN))
    parser.add_argument("--validation-dataset-dir", default=str(DEFAULT_VALIDATION))
    parser.add_argument(
        "--selector-output-dir",
        default=str(DEFAULT_MAIN_ROOT / "strength_predictor_multifidelity_dense"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MAIN_ROOT / "strength_utility_weight_sensitivity"),
    )
    parser.add_argument("--feature-set", default="MAR")
    parser.add_argument("--model", default="hist_gbdt")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def weighted_utility(
    frame: pd.DataFrame,
    damage_weight: float,
    capability_extra_weight: float,
) -> pd.Series:
    sign = frame["direction"].map({"suppression": -1.0, "enhancement": 1.0})
    if sign.isna().any():
        raise ValueError("Unexpected direction in selector dataset")
    target = sign * frame["target_delta"].astype(float)
    neighbor_damage = np.maximum(0.0, -frame["neighbor_delta"].astype(float))
    capability_damage = np.maximum(0.0, -frame["capability_delta"].astype(float))
    return (
        target
        - damage_weight * neighbor_damage
        - (damage_weight + capability_extra_weight) * capability_damage
    )


def bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples: List[np.ndarray] = []
    completed = 0
    while completed < reps:
        current = min(500, reps - completed)
        indices = rng.integers(0, len(values), size=(current, len(values)))
        samples.append(values[indices].mean(axis=1))
        completed += current
    joined = np.concatenate(samples)
    return float(np.quantile(joined, 0.025)), float(np.quantile(joined, 0.975))


def select_rows(
    frame: pd.DataFrame,
    clean_bonus: float,
    alpha_penalty: float,
) -> pd.DataFrame:
    candidates = frame.copy()
    candidates["selection_score"] = (
        candidates["predicted_utility"]
        + clean_bonus * candidates["predicted_clean_probability"]
        - alpha_penalty * candidates["abs_alpha"].astype(float)
    )
    selected = (
        candidates.sort_values(
            ["record_id", "method", "layer", "direction", "selection_score", "abs_alpha"],
            ascending=[True, True, True, True, False, True],
        )
        .groupby(["record_id", "method", "layer", "direction"], as_index=False)
        .first()
    )
    selected["abstained"] = selected["selection_score"] <= 0.0
    selected.loc[selected["abstained"], "weighted_utility"] = 0.0
    for column in ["target_success", "is_clean", "neighbor_damage", "capability_damage"]:
        selected.loc[selected["abstained"], column] = False
    return selected


def fixed_rows(
    dense_train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    alphas: Dict[str, float] = {}
    for direction in ["suppression", "enhancement"]:
        train = dense_train[dense_train["direction"].eq(direction)]
        means = train[train["alpha"].ne(0.0)].groupby("alpha")["weighted_utility"].mean()
        alpha = float(means.idxmax())
        alphas[direction] = alpha
        selected = validation[
            validation["direction"].eq(direction) & np.isclose(validation["alpha"], alpha)
        ].copy()
        selected["abstained"] = False
        rows.append(selected)
    return pd.concat(rows, ignore_index=True), alphas


def record_paired_difference(
    selector: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    keys = ["record_id", "method", "layer", "direction"]
    merged = selector[keys + [metric]].merge(
        baseline[keys + [metric]], on=keys, suffixes=("_selector", "_baseline"), validate="one_to_one"
    )
    merged["difference"] = (
        merged[f"{metric}_selector"].astype(float)
        - merged[f"{metric}_baseline"].astype(float)
    )
    values = merged.groupby("record_id")["difference"].mean().to_numpy()
    low, high = bootstrap_ci(values, reps, seed)
    return float(merged["difference"].mean()), low, high


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "utility_weight_sensitivity.csv"
    if summary_path.exists() and not args.overwrite:
        print((output_dir / "UTILITY_WEIGHT_SENSITIVITY.md").read_text())
        return

    train = read_dataset(Path(args.multifidelity_train_dataset_dir))
    dense_train = read_dataset(Path(args.dense_train_dataset_dir))
    validation = read_dataset(Path(args.validation_dataset_dir))
    if set(train["record_id"]) & set(validation["record_id"]):
        raise ValueError("Validation records overlap multi-fidelity training records")
    features = columns_for_feature_set(train, args.feature_set)
    missing = [column for column in features if column not in validation.columns]
    if missing:
        raise ValueError(f"Validation missing features: {missing}")
    clean_model_path = (
        Path(args.selector_output_dir)
        / "tasks"
        / f"multifidelity__{args.feature_set}__{args.model}"
        / "clean_model.joblib"
    )
    clean_model = joblib.load(clean_model_path)
    clean_probability = clean_model.predict_proba(validation[features])[:, 1]

    rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []
    for scenario_index, (
        scenario,
        damage_weight,
        capability_extra_weight,
        clean_bonus,
        alpha_penalty,
    ) in enumerate(SCENARIOS):
        train_scenario = train.copy()
        dense_scenario = dense_train.copy()
        validation_scenario = validation.copy()
        for frame in [train_scenario, dense_scenario, validation_scenario]:
            frame["weighted_utility"] = weighted_utility(
                frame, damage_weight, capability_extra_weight
            )

        _, utility_model = make_models(
            train_scenario,
            features,
            args.model,
            args.seed,
            linear_n_jobs=1,
        )
        utility_model.fit(
            train_scenario[features],
            train_scenario["weighted_utility"],
            model__sample_weight=train_scenario["source_weight"].astype(float).to_numpy(),
        )
        validation_scenario["predicted_clean_probability"] = clean_probability
        validation_scenario["predicted_utility"] = utility_model.predict(
            validation_scenario[features]
        )
        selector = select_rows(validation_scenario, clean_bonus, alpha_penalty)
        fixed, fixed_alphas = fixed_rows(dense_scenario, validation_scenario)
        zero = selector[["record_id", "method", "layer", "direction"]].copy()
        for column in [
            "weighted_utility",
            "target_success",
            "is_clean",
            "neighbor_damage",
            "capability_damage",
        ]:
            zero[column] = 0.0
        zero["abstained"] = True

        for direction_index, direction in enumerate(["suppression", "enhancement"]):
            selected_direction = selector[selector["direction"].eq(direction)].copy()
            fixed_direction = fixed[fixed["direction"].eq(direction)].copy()
            zero_direction = zero[zero["direction"].eq(direction)].copy()
            utility_fixed = record_paired_difference(
                selected_direction,
                fixed_direction,
                "weighted_utility",
                args.bootstrap_reps,
                args.seed + 100 * scenario_index + direction_index,
            )
            utility_zero = record_paired_difference(
                selected_direction,
                zero_direction,
                "weighted_utility",
                args.bootstrap_reps,
                args.seed + 1000 + 100 * scenario_index + direction_index,
            )
            neighbor_fixed = record_paired_difference(
                selected_direction,
                fixed_direction,
                "neighbor_damage",
                args.bootstrap_reps,
                args.seed + 2000 + 100 * scenario_index + direction_index,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "direction": direction,
                    "damage_weight": damage_weight,
                    "capability_extra_weight": capability_extra_weight,
                    "clean_bonus": clean_bonus,
                    "alpha_penalty": alpha_penalty,
                    "fixed_alpha": fixed_alphas[direction],
                    "selector_utility": float(selected_direction["weighted_utility"].mean()),
                    "utility_vs_fixed": utility_fixed[0],
                    "utility_vs_fixed_ci_low": utility_fixed[1],
                    "utility_vs_fixed_ci_high": utility_fixed[2],
                    "utility_vs_zero": utility_zero[0],
                    "utility_vs_zero_ci_low": utility_zero[1],
                    "utility_vs_zero_ci_high": utility_zero[2],
                    "neighbor_damage_change_vs_fixed": neighbor_fixed[0],
                    "abstention_rate": float(selected_direction["abstained"].mean()),
                    "target_success_rate": float(selected_direction["target_success"].mean()),
                    "clean_rate": float(selected_direction["is_clean"].mean()),
                }
            )
        selector.insert(0, "scenario", scenario)
        detail_frames.append(selector)

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    pd.concat(detail_frames, ignore_index=True).to_csv(
        output_dir / "utility_weight_selected_rows.csv", index=False
    )
    report = [
        "# Strength-Selector Utility-Weight Sensitivity",
        "",
        "Each scenario retrains the utility regressor on the frozen multi-fidelity training "
        "records and evaluates on the same 100 disjoint dense-grid validation records. The "
        "clean classifier is unchanged because clean labels do not depend on utility weights. "
        "Only one declared weight is varied at a time around the paper configuration.",
        "",
        markdown_table(
            summary,
            [
                "scenario",
                "direction",
                "damage_weight",
                "clean_bonus",
                "alpha_penalty",
                "selector_utility",
                "utility_vs_fixed",
                "utility_vs_fixed_ci_low",
                "utility_vs_fixed_ci_high",
                "utility_vs_zero",
                "utility_vs_zero_ci_low",
                "utility_vs_zero_ci_high",
                "neighbor_damage_change_vs_fixed",
                "abstention_rate",
            ],
        ),
        "",
        "The sensitivity grid is pre-specified and is not used to select a replacement main-paper "
        "configuration.",
        "",
    ]
    (output_dir / "UTILITY_WEIGHT_SENSITIVITY.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    (output_dir / "utility_weight_sensitivity_manifest.json").write_text(
        json.dumps(
            {
                "multifidelity_train_dataset_dir": str(
                    Path(args.multifidelity_train_dataset_dir).resolve()
                ),
                "dense_train_dataset_dir": str(Path(args.dense_train_dataset_dir).resolve()),
                "validation_dataset_dir": str(Path(args.validation_dataset_dir).resolve()),
                "validation_records": int(validation["record_id"].nunique()),
                "feature_set": args.feature_set,
                "model": args.model,
                "scenarios": [scenario[0] for scenario in SCENARIOS],
                "bootstrap_reps": args.bootstrap_reps,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((output_dir / "UTILITY_WEIGHT_SENSITIVITY.md").read_text())


if __name__ == "__main__":
    main()
