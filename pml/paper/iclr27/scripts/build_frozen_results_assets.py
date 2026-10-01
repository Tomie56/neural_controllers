#!/usr/bin/env python3
"""Build paper-facing frozen-result data and figures from completed PML outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parents[2]
RESULT_ROOT = (
    REPO_ROOT
    / "pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b"
)
STRICT_ROOT = RESULT_ROOT / "prediction_analysis_strict_later"
R_ABLATION_ROOT = RESULT_ROOT / "prediction_analysis_strict_later_r_ablation"
STRENGTH_ROOT = RESULT_ROOT / "strength_predictor_multifidelity_dense"
SELECTOR_DENSE_ROOT = (
    REPO_ROOT
    / "pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500"
    / "qwen3_1_7b/selector_dataset_tau_main_q95"
)
SELECTOR_VALIDATION_ROOT = (
    REPO_ROOT
    / "pml/results/fresh_multidomain_3000/stage2b_selector_validation_100"
    / "qwen3_1_7b/selector_dataset_tau_main_q95"
)
ENDPOINT_ROOT = (
    REPO_ROOT
    / "pml/results/fresh_multidomain_3000/stage_activation_pml/endpoint_validation_v3"
    / "qwen3_1_7b/formal"
)
FROZEN_DATA_ROOT = REPO_ROOT / "pml/data/pml_fresh_multidomain_3000_v1_frozen"
DATA_DIR = PAPER_ROOT / "data"
FIGURE_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "tables"

DATASET_LABELS = {
    "TIGER-Lab/MMLU-Pro": "MMLU-Pro",
    "edinburgh-dawg/mmlu-redux-2.0": "MMLU-Redux 2.0",
    "allenai/ai2_arc": "AI2 ARC",
    "allenai/openbookqa": "OpenBookQA",
    "allenai/sciq": "SciQ",
    "livebench/reasoning": "LiveBench reasoning",
    "Rowan/hellaswag": "HellaSwag",
    "allenai/qasc": "QASC",
    "livebench/math": "LiveBench math",
}

DATASET_CITATION_KEYS = {
    "TIGER-Lab/MMLU-Pro": "wang2024mmlupro",
    "edinburgh-dawg/mmlu-redux-2.0": "gema2024mmluredux",
    "allenai/ai2_arc": "clark2018arc",
    "allenai/openbookqa": "mihaylov-etal-2018-suit",
    "allenai/sciq": "welbl-etal-2017-crowdsourcing",
    "livebench/reasoning": "white2025livebench",
    "Rowan/hellaswag": "zellers-etal-2019-hellaswag",
    "allenai/qasc": "khot2020qasc",
    "livebench/math": "white2025livebench",
}

TARGET_LABELS = {
    "later_suppression_path": "Suppression",
    "later_enhancement_path": "Enhancement",
    "later_target_any_path": "Target-any",
    "later_neighbor_damage_any_path": "Neighbor damage",
    "later_capability_damage_any_path": "Capability damage",
    "later_clean_suppression_path": "Clean suppression",
    "later_clean_enhancement_path": "Clean enhancement",
    "later_clean_any_path": "Clean-any",
}

METHOD_LABELS = {
    "random": "Random",
    "mean_difference": "Mean difference",
    "linear": "Linear",
    "logistic": "Logistic",
    "rfm_agop_top1": "RFM/AGOP top-1",
}

COLORS = {
    "random": "#777777",
    "mean_difference": "#0072B2",
    "linear": "#009E73",
    "logistic": "#56B4E9",
    "rfm_agop_top1": "#CC79A7",
}

MAIN_FIGURE_ANNOTATION_OFFSETS = {
    ("random", -21): (-18, -12),
    ("random", -17): (4, -2),
    ("mean_difference", -21): (-12, 8),
    ("mean_difference", -17): (-34, 8),
    ("linear", -21): (4, 5),
    ("logistic", -21): (-38, 7),
    ("rfm_agop_top1", -21): (4, 5),
}

MAIN_FIGURE_SHORT_LABELS = {
    "random": "Random",
    "mean_difference": "Mean diff.",
    "linear": "Linear",
    "logistic": "Logistic",
    "rfm_agop_top1": "RFM/AGOP",
}

PAIRED_METRICS = {
    "later_target_any_path": "Target-any",
    "later_clean_any_path": "Clean-any",
    "later_neighbor_damage_any_path": "Neighbor damage",
    "later_capability_damage_any_path": "Capability damage",
}


def build_dataset_composition_snapshot() -> pd.DataFrame:
    validation = json.loads(
        (FROZEN_DATA_ROOT / "validation.json").read_text(encoding="utf-8")
    )
    counts = {str(key): int(value) for key, value in validation["dataset_counts"].items()}
    unknown = set(counts) - set(DATASET_LABELS)
    missing = set(DATASET_LABELS) - set(counts)
    if unknown or missing:
        raise ValueError(
            f"Frozen dataset mapping mismatch: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    rows = [
        {
            "source_key": source_key,
            "source_label": source_label,
            "citation_key": DATASET_CITATION_KEYS[source_key],
            "records": counts[source_key],
        }
        for source_key, source_label in DATASET_LABELS.items()
    ]
    frame = pd.DataFrame(rows)
    expected_total = int(validation["n_records"])
    if int(frame["records"].sum()) != expected_total:
        raise ValueError(
            f"Frozen dataset total mismatch: {frame['records'].sum()} != {expected_total}"
        )
    frame.to_csv(DATA_DIR / "dataset_composition.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Source composition of the frozen 3,000-record benchmark.}",
        r"\label{tab:dataset-composition}",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Source dataset & Records \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append("{} & {:,} ".format(row["source_label"], row["records"]) + r"\\")
    lines.extend(
        [
            r"\midrule",
            "Total & {:,} ".format(expected_total) + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "appendix_dataset_composition.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return frame


def build_paired_bootstrap_snapshot(n_resamples: int = 10_000) -> pd.DataFrame:
    path_rows = pd.read_csv(
        RESULT_ROOT / "prediction_dataset_strict_later/pml_path_prediction_dataset.csv"
    )
    methods = ["mean_difference", "linear", "logistic", "rfm_agop_top1"]
    rows: list[dict[str, object]] = []
    for layer_index, layer in enumerate([-21, -17]):
        random_rows = path_rows.loc[
            path_rows["layer"].eq(layer) & path_rows["method"].eq("random")
        ].set_index("record_id")
        for method_index, method in enumerate(methods):
            method_rows = path_rows.loc[
                path_rows["layer"].eq(layer) & path_rows["method"].eq(method)
            ].set_index("record_id")
            if set(method_rows.index) != set(random_rows.index):
                raise ValueError(f"Paired bootstrap record mismatch for {method}, layer {layer}")
            method_rows = method_rows.loc[random_rows.index]
            for metric_index, (metric, metric_label) in enumerate(PAIRED_METRICS.items()):
                differences = (
                    method_rows[metric].astype(float).to_numpy()
                    - random_rows[metric].astype(float).to_numpy()
                )
                rng = np.random.default_rng(
                    113 + 100 * layer_index + 10 * method_index + metric_index
                )
                bootstrap_means = np.empty(n_resamples, dtype=float)
                batch_size = 100
                for start in range(0, n_resamples, batch_size):
                    stop = min(start + batch_size, n_resamples)
                    indices = rng.integers(
                        0,
                        len(differences),
                        size=(stop - start, len(differences)),
                    )
                    bootstrap_means[start:stop] = differences[indices].mean(axis=1)
                lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
                rows.append(
                    {
                        "method": method,
                        "layer": layer,
                        "metric": metric,
                        "metric_label": metric_label,
                        "n_records": len(differences),
                        "difference": float(differences.mean()),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "n_resamples": n_resamples,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_mean_interval(
    values: np.ndarray,
    seed: int,
    n_resamples: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=float)
    batch_size = 100
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        bootstrap_means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def build_selector_bootstrap_snapshot(n_resamples: int = 10_000) -> pd.DataFrame:
    validation = pd.read_csv(SELECTOR_VALIDATION_ROOT / "alpha_level_selector_dataset.csv")
    selected = pd.read_csv(STRENGTH_ROOT / "selected_alpha_rows.csv")
    policy_manifest = json.loads(
        (STRENGTH_ROOT / "strength_policy_manifest.json").read_text(encoding="utf-8")
    )
    learned = selected.loc[selected["task"].eq(policy_manifest["selector_task"])].copy()
    learned = learned.rename(
        columns={
            "selected_utility": "utility",
            "selected_neighbor_damage": "neighbor_damage",
            "selected_clean": "is_clean",
        }
    )

    rows: list[dict[str, object]] = []
    for direction_index, direction in enumerate(["suppression", "enhancement"]):
        fixed_alpha = float(policy_manifest["fixed_alphas_selected_on_training"][direction])
        fixed = validation.loc[
            validation["direction"].eq(direction) & validation["alpha"].eq(fixed_alpha)
        ]
        learned_direction = learned.loc[learned["direction"].eq(direction)]
        fixed_by_record = fixed.groupby("record_id", sort=True)
        learned_by_record = learned_direction.groupby("record_id", sort=True)
        fixed_index = fixed_by_record.size().index
        learned_index = learned_by_record.size().index
        if not fixed_index.equals(learned_index):
            raise ValueError(f"Selector bootstrap record mismatch for {direction}")

        learned_utility = learned_by_record["utility"].mean().to_numpy(float)
        utility_gain = learned_utility - fixed_by_record["utility"].mean().to_numpy(float)
        neighbor_reduction = (
            fixed_by_record["neighbor_damage"].mean().to_numpy(float)
            - learned_by_record["neighbor_damage"].mean().to_numpy(float)
        )
        clean_gain = (
            learned_by_record["is_clean"].mean().to_numpy(float)
            - fixed_by_record["is_clean"].mean().to_numpy(float)
        )
        for metric_index, (metric, values) in enumerate(
            [
                ("learned_utility", learned_utility),
                ("utility_gain_vs_fixed", utility_gain),
                ("neighbor_damage_reduction", neighbor_reduction),
                ("clean_rate_gain", clean_gain),
            ]
        ):
            estimate, lower, upper = bootstrap_mean_interval(
                values,
                seed=313 + 10 * direction_index + metric_index,
                n_resamples=n_resamples,
            )
            rows.append(
                {
                    "direction": direction,
                    "metric": metric,
                    "n_records": len(values),
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_resamples": n_resamples,
                }
            )
    return pd.DataFrame(rows)


def write_main_selector_table(
    selector_bootstrap: pd.DataFrame,
    policy_source: pd.DataFrame,
) -> None:
    paired_source = pd.read_csv(
        STRENGTH_ROOT / "strength_policy_paired_comparisons.csv"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Held-out decisions on 100 records and 900 paths per objective. "
            r"Utility differences use a paired record bootstrap; N-dmg. is "
            r"fixed$\to$selector.}"
        ),
        r"\label{tab:main-strength-selector}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Objective & $\Delta U$ vs. 0 & $\Delta U$ vs. fixed & N-dmg. & Abstain \\",
        r"\midrule",
    ]
    labels = {"suppression": "Suppression", "enhancement": "Enhancement"}
    for direction in ["suppression", "enhancement"]:
        fixed = policy_source.loc[
            policy_source["policy"].eq("Train-tuned fixed alpha")
            & policy_source["direction"].eq(direction)
        ].iloc[0]
        selected = policy_source.loc[
            policy_source["policy"].eq("Multi-fidelity M+A+R selector")
            & policy_source["direction"].eq(direction)
        ].iloc[0]
        gain = paired_source.loc[
            paired_source["policy"].eq("Multi-fidelity M+A+R selector")
            & paired_source["baseline"].eq("Train-tuned fixed alpha")
            & paired_source["direction"].eq(direction)
            & paired_source["metric"].eq("utility")
        ].iloc[0]

        def compact(value: float) -> str:
            return f"{value:.3f}".replace("-0.", "-.").replace("0.", ".")

        def latex_compact(value: float) -> str:
            rendered = compact(value)
            return f"${rendered}$" if value < 0 else rendered

        versus_zero = paired_source.loc[
            paired_source["policy"].eq("Multi-fidelity M+A+R selector")
            & paired_source["baseline"].eq("No intervention")
            & paired_source["direction"].eq(direction)
            & paired_source["metric"].eq("utility")
        ].iloc[0]
        zero_estimate = compact(float(versus_zero["difference"]))
        if float(versus_zero["ci_low"]) > 0:
            zero_estimate = f"\\textbf{{{zero_estimate}}}"
        zero_interval = (
            f"{zero_estimate} "
            f"[{latex_compact(float(versus_zero['ci_low']))},"
            f"{latex_compact(float(versus_zero['ci_high']))}]"
        )
        utility_interval = (
            f"{compact(float(gain['difference']))} "
            f"[{compact(float(gain['ci_low']))},{compact(float(gain['ci_high']))}]"
        )
        fixed_damage = f"{float(fixed['neighbor_damage_rate']):.3f}".replace("0.", ".")
        selected_damage = f"{float(selected['neighbor_damage_rate']):.3f}".replace("0.", ".")
        lines.append(
            f"{labels[direction]} & {zero_interval} & "
            f"\\textbf{{{utility_interval.split()[0]}}} {utility_interval.split(maxsplit=1)[1]} & "
            f"{fixed_damage}$\\to$\\textbf{{{selected_damage}}} & "
            f"{compact(float(selected['abstention_rate']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "main_strength_selector.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def build_endpoint_snapshot() -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for path in sorted(ENDPOINT_ROOT.glob("*/layer_*/per_eval_endpoint.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No endpoint per-evaluation files found under {ENDPOINT_ROOT}")

    endpoint = pd.concat(frames, ignore_index=True)
    target = endpoint.loc[endpoint["eval_type"].eq("target")].copy()
    path_columns = ["record_id", "control_method", "layer"]
    evaluation_columns = path_columns + ["eval_id"]
    rows: list[dict[str, object]] = []
    for method, method_rows in target.groupby("control_method", sort=True):
        baseline = method_rows.loc[
            method_rows["alpha"].eq(0), evaluation_columns + ["endpoint_correct", "generated"]
        ].rename(
            columns={"endpoint_correct": "baseline_endpoint_correct", "generated": "baseline_generated"}
        )
        nonzero = method_rows.loc[~method_rows["alpha"].eq(0)].merge(
            baseline,
            on=evaluation_columns,
            how="left",
            validate="many_to_one",
        )
        if nonzero["baseline_endpoint_correct"].isna().any():
            raise ValueError(f"Missing alpha=0 endpoint baseline for {method}")

        rows.append(
            {
                "method": method,
                "n_paths": int(method_rows[path_columns].drop_duplicates().shape[0]),
                "n_target_prompts": int(baseline.shape[0]),
                "baseline_target_correct_rate": float(
                    baseline["baseline_endpoint_correct"].mean()
                ),
                "unique_target_gain_count": int(
                    nonzero.loc[
                        nonzero["endpoint_gain_from_base"].astype(bool), evaluation_columns
                    ].drop_duplicates().shape[0]
                ),
                "unique_target_damage_count": int(
                    nonzero.loc[
                        nonzero["endpoint_damage_from_base"].astype(bool), evaluation_columns
                    ].drop_duplicates().shape[0]
                ),
                "target_text_changed_rate": float(
                    nonzero["generated"].fillna("").ne(
                        nonzero["baseline_generated"].fillna("")
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_prediction_snapshot() -> pd.DataFrame:
    strict = pd.read_csv(STRICT_ROOT / "prediction_summary.csv")
    response = pd.read_csv(R_ABLATION_ROOT / "prediction_summary.csv")
    keys = ["cohort", "split", "target", "model"]
    strict_index = strict.set_index(keys + ["feature_set"])
    response_index = response.set_index(keys + ["feature_set"])
    rows: list[dict[str, object]] = []
    for target in TARGET_LABELS:
        row: dict[str, object] = {"target": target, "target_label": TARGET_LABELS[target]}
        for feature_set, source in [
            ("B+M", strict_index),
            ("B+M+L", strict_index),
            ("M+R", response_index),
            ("B+M+R", response_index),
            ("B+M+L+R", response_index),
        ]:
            selected = source.xs(
                ("all", target, "random_forest"),
                level=("cohort", "target", "model"),
            ).xs(feature_set, level="feature_set")
            row[f"{feature_set}_auroc"] = float(selected["auroc_mean"].mean())
            row[f"{feature_set}_ap"] = float(selected["average_precision_mean"].mean())
        row["delta_L_auroc"] = float(row["B+M+L_auroc"]) - float(row["B+M_auroc"])
        row["delta_R_auroc"] = float(row["B+M+R_auroc"]) - float(row["B+M_auroc"])
        row["delta_L_ap"] = float(row["B+M+L_ap"]) - float(row["B+M_ap"])
        row["delta_R_ap"] = float(row["B+M+R_ap"]) - float(row["B+M_ap"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_split_snapshot() -> pd.DataFrame:
    strict = pd.read_csv(STRICT_ROOT / "prediction_summary.csv")
    response = pd.read_csv(R_ABLATION_ROOT / "prediction_summary.csv")
    strict = strict[
        (strict["cohort"] == "all")
        & (strict["model"] == "random_forest")
        & strict["target"].isin(TARGET_LABELS)
        & strict["feature_set"].isin(["B+M", "B+M+L"])
    ]
    response = response[
        (response["cohort"] == "all")
        & (response["model"] == "random_forest")
        & response["target"].isin(TARGET_LABELS)
        & response["feature_set"].isin(["M+R", "B+M+R", "B+M+L+R"])
    ]
    combined = pd.concat([strict, response], ignore_index=True)
    return (
        combined.groupby(["split", "feature_set"], as_index=False)[
            ["auroc_mean", "average_precision_mean", "brier_mean"]
        ]
        .mean()
        .sort_values(["split", "feature_set"])
    )


def build_geometry_ablation_snapshot() -> pd.DataFrame:
    prediction = pd.read_csv(STRICT_ROOT / "prediction_summary.csv")
    subset = prediction.loc[
        prediction["cohort"].eq("rfm")
        & prediction["model"].eq("random_forest")
        & prediction["target"].isin(TARGET_LABELS)
        & prediction["feature_set"].isin(["B+M+L", "B+M+L+G"])
    ]
    rows: list[dict[str, object]] = []
    for target, target_label in TARGET_LABELS.items():
        base = subset.loc[
            subset["target"].eq(target) & subset["feature_set"].eq("B+M+L")
        ]
        geometry = subset.loc[
            subset["target"].eq(target) & subset["feature_set"].eq("B+M+L+G")
        ]
        rows.append(
            {
                "target": target,
                "target_label": target_label,
                "B+M+L_auroc": float(base["auroc_mean"].mean()),
                "B+M+L+G_auroc": float(geometry["auroc_mean"].mean()),
                "delta_auroc": float(
                    geometry["auroc_mean"].mean() - base["auroc_mean"].mean()
                ),
                "delta_ap": float(
                    geometry["average_precision_mean"].mean()
                    - base["average_precision_mean"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def save_scale_manifest() -> None:
    unique_records = 3000
    methods = 6
    layers = [-21, -17]
    strengths = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]
    probes_per_record = {"target": 3, "neighbor": 3, "capability": 4}
    probes = sum(probes_per_record.values())
    method_layer_paths = unique_records * methods * len(layers)
    probe_trajectories = method_layer_paths * probes
    path_strength_evaluations = method_layer_paths * len(strengths)
    probe_strength_observations = path_strength_evaluations * probes
    scale = {
        "unique_records": unique_records,
        "datasets": 9,
        "domains": 14,
        "methods": methods,
        "layers": layers,
        "strengths": strengths,
        "method_layer_paths": method_layer_paths,
        "probe_trajectories": probe_trajectories,
        "path_strength_evaluations": path_strength_evaluations,
        "probe_strength_observations": probe_strength_observations,
        "probe_level_evaluations": probe_strength_observations,
        "continuation_likelihood_evaluations": 2 * probe_strength_observations,
        "probes_per_record": probes_per_record,
        "strict_later_label_strengths": [0.25, 0.5],
        "weak_probe_strength": 0.1,
        "prediction_splits": ["record", "dataset", "domain"],
        "prediction_folds": 5,
        "seed": 113,
        "strength_selection": {
            "dense_train_records": 500,
            "dense_validation_records": 100,
            "sparse_auxiliary_records": 2400,
            "multifidelity_train_records": 2900,
            "dense_train_raw_rows": 243000,
            "dense_validation_raw_rows": 24300,
            "dense_train_nonzero_candidates": 234000,
            "dense_validation_nonzero_candidates": 23400,
            "multifidelity_nonzero_candidates": 320400,
            "validation_overlap": 0,
        },
    }
    (DATA_DIR / "experiment_scale.json").write_text(
        json.dumps(scale, indent=2), encoding="utf-8"
    )


def make_figure(outcomes: pd.DataFrame, prediction: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.7,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.35),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
        constrained_layout=False,
    )

    plotted = outcomes[outcomes["method"].isin(METHOD_LABELS)].copy()
    plotted["collateral_mean"] = 0.5 * (
        plotted["later_neighbor_damage_any_path_rate"]
        + plotted["later_capability_damage_any_path_rate"]
    )
    markers = {-21: "o", -17: "^"}
    for _, row in plotted.iterrows():
        method = str(row["method"])
        layer = int(row["layer"])
        x = 100 * float(row["collateral_mean"])
        y = 100 * float(row["later_target_any_path_rate"])
        axes[0].scatter(
            x,
            y,
            s=42,
            marker=markers[layer],
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        if (method, layer) in MAIN_FIGURE_ANNOTATION_OFFSETS:
            dx, dy = MAIN_FIGURE_ANNOTATION_OFFSETS[(method, layer)]
            label = MAIN_FIGURE_SHORT_LABELS[method]
            if layer == -17:
                label = f"{label} (-17)"
            axes[0].annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.8,
            )
    layer_handles = [
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor="black", markersize=5.0, label="Layer -21"
        ),
        mpl.lines.Line2D(
            [], [], marker="^", linestyle="none", markerfacecolor="white",
            markeredgecolor="black", markersize=5.0, label="Layer -17"
        ),
    ]
    axes[0].legend(handles=layer_handles, frameon=False, loc="lower right")
    axes[0].set_xlim(6.75, 9.35)
    axes[0].set_ylim(8.55, 13.55)
    axes[0].set_xlabel("Mean collateral-damage rate (%)")
    axes[0].set_ylabel("Later target-any rate (%)")
    axes[0].set_title("(a) Leverage versus collateral movement", loc="left", fontweight="bold")
    axes[0].grid(color="#E5E5E5", linewidth=0.5)

    ordered = prediction.sort_values("delta_R_auroc").reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    axes[1].axvline(0, color="#999999", linewidth=0.8)
    for index, row in ordered.iterrows():
        axes[1].plot(
            [100 * row["delta_L_auroc"], 100 * row["delta_R_auroc"]],
            [index, index],
            color="#B8C0C8",
            linewidth=1.2,
            zorder=1,
        )
    axes[1].scatter(
        100 * ordered["delta_L_auroc"],
        y_positions,
        color="#E69F00",
        marker="s",
        s=34,
        edgecolor="black",
        linewidth=0.3,
        label="Static localization $L$",
        zorder=3,
    )
    axes[1].scatter(
        100 * ordered["delta_R_auroc"],
        y_positions,
        color="#0072B2",
        marker="o",
        s=38,
        edgecolor="black",
        linewidth=0.3,
        label="Weak response $R$",
        zorder=3,
    )
    axes[1].set_yticks(y_positions, ordered["target_label"])
    axes[1].set_xlabel("Mean AUROC gain over $B+M$ (points)")
    axes[1].set_title("(b) Static localization versus weak response", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#E5E5E5", linewidth=0.5)
    axes[1].legend(frameon=False, loc="lower right", handletextpad=0.4)
    axes[1].set_xlim(-1.8, 23.5)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.90, wspace=0.42)

    figure_path = FIGURE_DIR / "frozen_confirmatory_results"
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    direct_labels = []
    for method, layer in MAIN_FIGURE_ANNOTATION_OFFSETS:
        label = MAIN_FIGURE_SHORT_LABELS[method]
        if layer == -17:
            label = f"{label} (-17)"
        direct_labels.append({"method": method, "layer": layer, "label": label})
    (DATA_DIR / "frozen_confirmatory_figure_manifest.json").write_text(
        json.dumps(
            {
                "direct_label_count": len(direct_labels),
                "direct_labels": direct_labels,
                "policy": (
                    "All five layer -21 methods plus random and mean-difference "
                    "at layer -17."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outcomes = pd.read_csv(STRICT_ROOT / "outcome_summary.csv")
    prediction = build_prediction_snapshot()
    split_summary = build_split_snapshot()
    geometry_ablation = build_geometry_ablation_snapshot()
    paired_bootstrap = build_paired_bootstrap_snapshot()
    selector_bootstrap = build_selector_bootstrap_snapshot()
    build_dataset_composition_snapshot()
    outcomes.to_csv(DATA_DIR / "frozen_outcome_summary.csv", index=False)
    prediction.to_csv(DATA_DIR / "prediction_ablation_all_rf_split_average.csv", index=False)
    split_summary.to_csv(DATA_DIR / "prediction_split_macro.csv", index=False)
    geometry_ablation.to_csv(DATA_DIR / "geometry_ablation_rfm.csv", index=False)
    paired_bootstrap.to_csv(DATA_DIR / "paired_bootstrap_vs_random.csv", index=False)
    selector_bootstrap.to_csv(DATA_DIR / "selector_paired_bootstrap.csv", index=False)
    strength_summary = pd.read_csv(STRENGTH_ROOT / "strength_selector_summary.csv")
    strength_summary.to_csv(DATA_DIR / "strength_selector_summary.csv", index=False)
    strength_policy = pd.read_csv(STRENGTH_ROOT / "strength_policy_evaluation.csv")
    strength_policy.to_csv(DATA_DIR / "strength_policy_evaluation.csv", index=False)
    write_main_selector_table(selector_bootstrap, strength_policy)
    endpoint_summary = build_endpoint_snapshot()
    endpoint_summary.to_csv(DATA_DIR / "endpoint_validation_summary.csv", index=False)
    strength_policy_manifest = json.loads(
        (STRENGTH_ROOT / "strength_policy_manifest.json").read_text(encoding="utf-8")
    )
    (DATA_DIR / "strength_policy_manifest.json").write_text(
        json.dumps(strength_policy_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    strength_run_config = json.loads(
        (STRENGTH_ROOT / "run_config.json").read_text(encoding="utf-8")
    )
    (DATA_DIR / "strength_run_config.json").write_text(
        json.dumps(strength_run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selector_dataset_manifest = json.loads(
        (SELECTOR_DENSE_ROOT / "selector_dataset_manifest.json").read_text(encoding="utf-8")
    )
    (DATA_DIR / "strength_dataset_manifest.json").write_text(
        json.dumps(selector_dataset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    strength_metrics = []
    for metrics_path in sorted((STRENGTH_ROOT / "tasks").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        strength_metrics.append(
            {
                key: metrics[key]
                for key in [
                    "task",
                    "regime",
                    "feature_set",
                    "model",
                    "n_train_rows",
                    "n_train_records",
                    "n_validation_rows",
                    "n_validation_records",
                    "n_features",
                    "clean_auroc",
                    "clean_average_precision",
                    "utility_mae",
                    "utility_r2",
                ]
            }
        )
    pd.DataFrame(strength_metrics).to_csv(
        DATA_DIR / "strength_predictor_metrics.csv", index=False
    )
    save_scale_manifest()
    make_figure(outcomes, prediction)
    print(f"Wrote paper data to {DATA_DIR}")
    print(f"Wrote figure to {FIGURE_DIR / 'frozen_confirmatory_results.pdf'}")


if __name__ == "__main__":
    main()
