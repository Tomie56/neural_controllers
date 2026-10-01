#!/usr/bin/env python3
"""Build revised paper tables and figures from frozen PML results."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PAPER_ROOT / "data"
TABLE_DIR = PAPER_ROOT / "tables"
FIGURE_DIR = PAPER_ROOT / "figures"
PML_ROOT = PAPER_ROOT.parents[1]
PILOT_CURVE = (
    PML_ROOT
    / "results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b"
    / "threshold_reanalysis/method_layer_alpha_curve.csv"
)

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"

MODEL_DEPTHS = {
    "Qwen3-1.7B-Base": 28,
    "Qwen3.5-2B-Base": 24,
    "Ministral-3-3B-Base-2512": 26,
}
MODEL_KEYS = {
    "Qwen3-1.7B-Base": "qwen3_1_7b",
    "Qwen3.5-2B-Base": "qwen3_5_2b_base",
    "Ministral-3-3B-Base-2512": "ministral_3_3b_base_2512",
}
METHOD_LABELS = {
    "random": "Random",
    "mean_difference": "Mean diff.",
    "logistic": "Logistic",
    "linear": "Linear",
    "rfm_agop_top1": "RFM/AGOP",
}
MODEL_METHODS = {
    "Qwen3-1.7B-Base": ["random", "mean_difference", "logistic", "linear", "rfm_agop_top1"],
    "Qwen3.5-2B-Base": ["random", "mean_difference", "logistic", "rfm_agop_top1"],
    "Ministral-3-3B-Base-2512": ["random", "mean_difference", "logistic", "rfm_agop_top1"],
}
OUTCOME_COLUMNS = [
    ("later_suppression_path_rate", "Supp."),
    ("later_enhancement_path_rate", "Enh."),
    ("later_target_any_path_rate", "Target"),
    ("later_neighbor_damage_any_path_rate", "N-dmg."),
    ("later_capability_damage_any_path_rate", "C-dmg."),
    ("later_clean_suppression_path_rate", "Clean supp."),
    ("later_clean_enhancement_path_rate", "Clean enh."),
    ("later_clean_any_path_rate", "Clean"),
]

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
    }
)


def positive_layer(model_label: str, layer: int) -> int:
    return MODEL_DEPTHS[model_label] + int(layer)


def target_threshold(model_label: str, experiment: str) -> float | None:
    model_key = MODEL_KEYS[model_label]
    if experiment == "primary":
        manifest_path = (
            PML_ROOT
            / "results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b"
            / "prediction_dataset/prediction_dataset_manifest.json"
        )
    else:
        manifest_path = (
            PML_ROOT
            / "results/fresh_multidomain_3000/stage_activation_pml/replication_500_residual_rms_v2"
            / model_key
            / "prediction_dataset/prediction_dataset_manifest.json"
        )
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = (payload.get("thresholds") or {}).get("target")
    return None if value is None else float(value)


def load_model_outcomes(
    model_label: str, experiment: str, cross: pd.DataFrame
) -> pd.DataFrame | None:
    if experiment == "primary":
        frame = pd.read_csv(DATA_DIR / "frozen_outcome_summary.csv")
    else:
        model_rows = cross[cross["model_label"].eq(model_label)]
        if model_rows.empty or not bool(model_rows["complete"].all()):
            return None
        summary_path = Path(str(model_rows["summary_path"].iloc[0]))
        if not summary_path.is_file():
            return None
        frame = pd.read_csv(summary_path)
        if "scope" in frame.columns:
            frame = frame[frame["scope"].eq("method_layer")]
    frame = frame[frame["method"].isin(MODEL_METHODS[model_label])].copy()
    frame["display_layer"] = frame["layer"].map(
        lambda value: positive_layer(model_label, int(value))
    )
    return frame


def write_main_table_legacy() -> None:
    cross = pd.read_csv(DATA_DIR / "cross_model_replication_summary.csv")
    model_order = ["Qwen3-1.7B-Base", "Qwen3.5-2B-Base", "Ministral-3-3B-Base-2512"]

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.45pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\caption{Cross-model confirmatory outcomes at two pre-specified aligned layers. Panel A reports absolute path-outcome rates for every direction evaluated by each frozen protocol: five directions for Qwen3-1.7B-Base and four for each compact replication. Model and sample size define row groups, and each layer is printed once per direction block. Values are percentages of records with suppression (Supp.), enhancement (Enh.), either target effect (Target), semantic-neighbor damage (N-dmg.), capability damage (C-dmg.), clean suppression, clean enhancement, or either clean path (Clean). Within each completed model--layer block, bold marks the largest displayed learned-direction Target and Clean rates; reserved replication cells remain blank until the complete 500-record summary is available. Panel B reports the pre-specified learned-minus-random target and clean differences used for cross-model confirmation.}",
        r"\label{tab:main-results}",
        r"\begin{tabular*}{0.94\textwidth}{@{\extracolsep{\fill}}clrrrrrrrr@{}}",
        r"\toprule",
        r"\multicolumn{10}{c}{\textbf{A. Absolute path-outcome rates for all evaluated directions (\%)}} \\",
        r"\cmidrule(lr){1-10}",
        r"Layer & Direction & Supp. & Enh. & Target & N-dmg. & C-dmg. & Cl. supp. & Cl. enh. & Clean \\",
    ]
    for model_index, model in enumerate(model_order):
        lines.append(r"\midrule")
        model_rows = cross[cross["model_label"].eq(model)]
        records = int(model_rows["records"].iloc[0])
        lines.append(
            rf"\multicolumn{{10}}{{l}}{{\textbf{{{model}}} \quad ($n={records:,}$)}} \\"
        )
        layers = [positive_layer(model, int(value)) for value in model_rows["layer"]]
        outcomes = load_model_outcomes(model, cross)
        for layer_index, layer in enumerate(layers):
            if layer_index:
                lines.append(r"\addlinespace[0.8pt]")
            layer_frame = None if outcomes is None else outcomes[outcomes["display_layer"].eq(layer)]
            learned_frame = None if layer_frame is None else layer_frame[~layer_frame["method"].eq("random")]
            target_best = None if learned_frame is None or learned_frame.empty else round(100 * learned_frame["later_target_any_path_rate"].max(), 1)
            clean_best = None if learned_frame is None or learned_frame.empty else round(100 * learned_frame["later_clean_any_path_rate"].max(), 1)
            for method_index, method in enumerate(MODEL_METHODS[model]):
                values = ["--"] * len(OUTCOME_COLUMNS)
                if layer_frame is not None:
                    matching = layer_frame[layer_frame["method"].eq(method)]
                    if not matching.empty:
                        row = matching.iloc[0]
                        values = [f"{100 * float(row[column]):.1f}" for column, _ in OUTCOME_COLUMNS]
                        if method != "random" and np.isclose(round(100 * float(row["later_target_any_path_rate"]), 1), target_best):
                            values[2] = rf"\textbf{{{values[2]}}}"
                        if method != "random" and np.isclose(round(100 * float(row["later_clean_any_path_rate"]), 1), clean_best):
                            values[-1] = rf"\textbf{{{values[-1]}}}"
                layer_cell = str(layer) if method_index == 0 else ""
                lines.append(
                    f"{layer_cell} & {METHOD_LABELS[method]} & "
                    + " & ".join(values)
                    + r" \\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\vspace{3pt}"])

    lines.extend(
        [
            r"\begin{tabular*}{0.76\textwidth}{@{\extracolsep{\fill}}lcccrr@{}}",
            r"\toprule",
            r"\multicolumn{6}{c}{\textbf{B. Pre-specified learned-minus-random confirmation (points)}} \\",
            r"\cmidrule(lr){1-6}",
            r"Model & $n$ & Layer & Direction & $\Delta$Target & $\Delta$Clean \\",
            r"\midrule",
        ]
    )
    for _, row in cross.iterrows():
        model = str(row["model_label"])
        layer = positive_layer(model, int(row["layer"]))
        direction = METHOD_LABELS[str(row["learned_method"])]
        if bool(row["complete"]):
            delta_target = f"{100 * float(row['target_delta']):+.1f}"
            delta_clean = f"{100 * float(row['clean_delta']):+.1f}"
        else:
            delta_target = delta_clean = "--"
        lines.append(
            f"{model} & {int(row['records']):,} & {layer} & {direction} & {delta_target} & {delta_clean} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    (TABLE_DIR / "main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def write_main_table() -> None:
    cross = pd.read_csv(DATA_DIR / "cross_model_replication_summary.csv")
    blocks = [
        ("A", "Qwen3-1.7B-Base", "primary", 3000, "primary scale"),
        ("B", "Qwen3-1.7B-Base", "rms", 500, "residual-norm matched"),
        ("C", "Qwen3.5-2B-Base", "rms", 500, "residual-norm matched"),
        ("D", "Ministral-3-3B-Base-2512", "rms", 500, "residual-norm matched"),
    ]

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Random-calibrated path incidence (\%) for the 3,000-record primary study and three 500-record confirmations. N/C-dmg. are collateral damage; $\Delta$ columns are learned-minus-random points. Bold marks each block's strongest learned result.}",
        r"\label{tab:main-results}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}clrrrrrrrrrr@{}}",
        r"\toprule",
        r"Layer & Direction & Supp. & Enh. & Target & N-dmg. & C-dmg. & Cl. supp. & Cl. enh. & Clean & $\Delta$T & $\Delta$C \\",
    ]

    for block_label, model, experiment, records, protocol in blocks:
        lines.append(r"\midrule")
        model_rows = cross[cross["model_label"].eq(model)]
        threshold = target_threshold(model, experiment)
        threshold_text = r"\mathrm{pending}" if threshold is None else f"{threshold:.3f}"
        display_model = model.replace("-2512", "")
        lines.append(
            rf"\multicolumn{{12}}{{c}}{{\textbf{{{block_label}. {display_model}}} \quad "
            rf"($n={records:,}$, {protocol}, $\tau_T={threshold_text}$)}} \\"
        )
        lines.append(r"\cmidrule(lr){1-12}")
        layers = (
            [7, 11]
            if experiment == "primary"
            else sorted(model_rows["display_layer"].astype(int).unique().tolist())
        )
        outcomes = load_model_outcomes(model, experiment, cross)

        if outcomes is None:
            for layer in layers:
                lines.append(
                    rf"{layer} & \multicolumn{{11}}{{l}}{{Pending complete frozen 500-record block.}} \\"
                )
            continue

        for layer_index, layer in enumerate(layers):
            if layer_index:
                lines.append(r"\addlinespace[0.35pt]")
            layer_frame = outcomes[outcomes["display_layer"].eq(layer)]
            learned_frame = layer_frame[~layer_frame["method"].eq("random")]
            target_best = round(100 * learned_frame["later_target_any_path_rate"].max(), 1)
            clean_best = round(100 * learned_frame["later_clean_any_path_rate"].max(), 1)

            random_matches = layer_frame[layer_frame["method"].eq("random")]
            random_row = None if random_matches.empty else random_matches.iloc[0]

            target_delta_best = None
            clean_delta_best = None
            if not learned_frame.empty and random_row is not None:
                target_delta_best = max(
                    round(
                        100
                        * (
                            float(candidate["later_target_any_path_rate"])
                            - float(random_row["later_target_any_path_rate"])
                        ),
                        1,
                    )
                    for _, candidate in learned_frame.iterrows()
                )
                clean_delta_best = max(
                    round(
                        100
                        * (
                            float(candidate["later_clean_any_path_rate"])
                            - float(random_row["later_clean_any_path_rate"])
                        ),
                        1,
                    )
                    for _, candidate in learned_frame.iterrows()
                )

            methods = MODEL_METHODS[model] if experiment == "primary" else [
                "random", "mean_difference", "logistic", "rfm_agop_top1"
            ]
            for method in methods:
                values = ["--"] * len(OUTCOME_COLUMNS)
                delta_values = ["--", "--"]
                matching = layer_frame[layer_frame["method"].eq(method)]
                if not matching.empty:
                    row = matching.iloc[0]
                    values = [f"{100 * float(row[column]):.1f}" for column, _ in OUTCOME_COLUMNS]
                    if method != "random" and np.isclose(round(100 * float(row["later_target_any_path_rate"]), 1), target_best):
                        values[2] = rf"\textbf{{{values[2]}}}"
                    if method != "random" and np.isclose(round(100 * float(row["later_clean_any_path_rate"]), 1), clean_best):
                        values[-1] = rf"\textbf{{{values[-1]}}}"
                    if method != "random" and random_row is not None:
                        target_delta = round(
                            100
                            * (
                                float(row["later_target_any_path_rate"])
                                - float(random_row["later_target_any_path_rate"])
                            ),
                            1,
                        )
                        clean_delta = round(
                            100
                            * (
                                float(row["later_clean_any_path_rate"])
                                - float(random_row["later_clean_any_path_rate"])
                            ),
                            1,
                        )
                        delta_values = [f"{target_delta:+.1f}", f"{clean_delta:+.1f}"]
                        if np.isclose(target_delta, target_delta_best):
                            delta_values[0] = rf"\textbf{{{delta_values[0]}}}"
                        if np.isclose(clean_delta, clean_delta_best):
                            delta_values[1] = rf"\textbf{{{delta_values[1]}}}"

                lines.append(
                    f"{layer} & {METHOD_LABELS[method]} & "
                    + " & ".join(values)
                    + " & "
                    + " & ".join(delta_values)
                    + r" \\"
                )

    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    (TABLE_DIR / "main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def build_layer_selection() -> pd.DataFrame:
    curve = pd.read_csv(PILOT_CURVE)
    rows = []
    for layer, group in curve.groupby("layer"):
        endpoints = group[group["alpha"].isin([-1.0, 1.0])]
        learned = endpoints[endpoints["method"].isin(["mean_difference", "logistic"])]
        random = endpoints[endpoints["method"].eq("random")]
        learned_pivot = learned.groupby("alpha")["target_mean_delta"].mean()
        random_pivot = random.groupby("alpha")["target_mean_delta"].mean()
        rows.append(
            {
                "layer": 28 + int(layer),
                "depth_percent": 100 * (28 + int(layer)) / 27,
                "learned_target_span": float(learned_pivot.loc[1.0] - learned_pivot.loc[-1.0]),
                "random_target_span": float(random_pivot.loc[1.0] - random_pivot.loc[-1.0]),
                "learned_signed_rate": float(learned["target_signed_direction_rate"].mean()),
                "selected": 28 + int(layer) in {7, 11},
            }
        )
    result = pd.DataFrame(rows).sort_values("layer")
    result.to_csv(DATA_DIR / "layer_selection_pilot.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\includegraphics[width=0.96\columnwidth]{figures/layer_selection_profile.pdf}",
        r"\caption{Layer selection on a 500-record subset. Target-margin span across relative depth identifies blocks 7 and 11 as the strongest distinct middle-depth blocks; random spans remain near zero.}",
        r"\label{tab:layer-selection}",
        r"\end{table}",
        "",
    ]
    (TABLE_DIR / "layer_selection.tex").write_text("\n".join(lines), encoding="utf-8")
    return result


def write_prediction_table() -> None:
    prediction = pd.read_csv(DATA_DIR / "prediction_ablation_all_rf_split_average.csv")
    feature_sets = ["B+M", "B+M+L", "M+R", "B+M+R", "B+M+L+R"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full later-path prediction performance in the all-method random-forest cohort, averaged over record-, dataset-, and domain-grouped splits. Each cell is AUROC/AP. $B$ denotes identifiers and base margins, $M$ method/layer metadata, $L$ static localization, and $R$ the disjoint response at $|\alpha|=0.1$. Later labels use only $|\alpha|\in\{0.25,0.5\}$.}",
        r"\label{tab:full-prediction-main}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Outcome & $B+M$ & $B+M+L$ & $M+R$ & $B+M+R$ & $B+M+L+R$ \\",
        r"\midrule",
    ]
    for _, row in prediction.iterrows():
        cells = []
        for feature_set in feature_sets:
            text = f"{float(row[f'{feature_set}_auroc']):.3f}/{float(row[f'{feature_set}_ap']):.3f}"
            cells.append(text)
        lines.append(f"{row['target_label']} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (TABLE_DIR / "main_prediction_full.tex").write_text("\n".join(lines), encoding="utf-8")


def make_path_outcome_profiles() -> None:
    outcomes = pd.read_csv(DATA_DIR / "frozen_outcome_summary.csv")
    outcomes = outcomes[outcomes["method"].isin(METHOD_LABELS)].copy()
    outcomes["layer_display"] = outcomes["layer"].map(lambda value: 28 + int(value))
    prediction = pd.read_csv(DATA_DIR / "prediction_ablation_all_rf_split_average.csv")
    learned_methods = ["mean_difference", "logistic", "linear", "rfm_agop_top1"]
    method_colors = {
        "mean_difference": BLUE,
        "logistic": GREEN,
        "linear": ORANGE,
        "rfm_agop_top1": PURPLE,
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.72))
    axis = axes[0]
    for layer, marker in [(7, "o"), (11, "^")]:
        for method in ["random", *learned_methods]:
            row = outcomes[(outcomes["method"].eq(method)) & (outcomes["layer_display"].eq(layer))].iloc[0]
            x_value = 50 * (float(row["later_neighbor_damage_any_path_rate"]) + float(row["later_capability_damage_any_path_rate"]))
            y_value = 100 * float(row["later_target_any_path_rate"])
            color = GRAY if method == "random" else method_colors[method]
            axis.scatter(
                x_value,
                y_value,
                s=42,
                marker=marker,
                facecolor=color,
                edgecolor="#263238",
                linewidth=0.65,
                zorder=4,
            )

    label_offsets = {
        "mean_difference": (-15, 7),
        "logistic": (-26, 4),
        "linear": (4, -10),
        "rfm_agop_top1": (4, 4),
        "random": (-18, 7),
    }
    for method in ["random", *learned_methods]:
        row = outcomes[(outcomes["method"].eq(method)) & (outcomes["layer_display"].eq(7))].iloc[0]
        x_value = 50 * (float(row["later_neighbor_damage_any_path_rate"]) + float(row["later_capability_damage_any_path_rate"]))
        y_value = 100 * float(row["later_target_any_path_rate"])
        axis.annotate(METHOD_LABELS[method], (x_value, y_value), xytext=label_offsets[method], textcoords="offset points", fontsize=6.5, color="#1F2937")

    axis.set_title("(a) Qwen3-1.7B-Base: leverage vs. collateral", loc="left", fontweight="normal", fontsize=8.0)
    axis.set_xlabel("Mean collateral-damage rate (%)")
    axis.set_ylabel("Target-any paths (%)")
    axis.set_xlim(6.72, 9.68)
    axis.set_ylim(8.55, 13.38)
    axis.grid(color="#E5E7EB", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    layer_handles = [
        mpl.lines.Line2D([], [], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="#263238", markersize=4.8, label="Layer 7"),
        mpl.lines.Line2D([], [], marker="^", color="none", markerfacecolor=GRAY, markeredgecolor="#263238", markersize=4.8, label="Layer 11"),
    ]
    legend = axis.legend(
        handles=layer_handles,
        frameon=True,
        ncol=1,
        loc="lower right",
        fontsize=6.5,
        handletextpad=0.35,
        borderpad=0.35,
        labelspacing=0.25,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#9CA3AF")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(0.96)

    short_label = {
        "Capability damage": "Capability damage",
        "Clean suppression": "Clean suppression",
        "Enhancement": "Enhancement",
        "Target-any": "Target-any",
        "Clean-any": "Clean-any",
        "Neighbor damage": "Neighbor damage",
        "Suppression": "Suppression",
        "Clean enhancement": "Clean enhancement",
    }
    outcome_order = [
        "Neighbor damage",
        "Clean enhancement",
        "Enhancement",
        "Capability damage",
        "Suppression",
        "Clean suppression",
        "Target-any",
        "Clean-any",
    ]
    prediction = prediction.set_index("target_label").loc[outcome_order].reset_index()
    y_positions = np.arange(len(prediction))
    for panel_index, (axis, column, title, color, marker, xlim) in enumerate([
        (axes[1], "delta_L_auroc", "(b) Static localization $L$", ORANGE, "s", (-1.15, 0.65)),
        (axes[2], "delta_R_auroc", "(c) Weak probe $R$", BLUE, "o", (0, 23.5)),
    ]):
        values = 100 * prediction[column].to_numpy()
        axis.axvline(0, color="#9CA3AF", linewidth=0.8)
        axis.hlines(
            y_positions,
            np.minimum(values, 0),
            np.maximum(values, 0),
            color=color,
            linewidth=1.15,
            alpha=0.72,
            zorder=2,
        )
        axis.scatter(values, y_positions, color=color, marker=marker, s=30, edgecolor="#263238", linewidth=0.55, zorder=3)
        axis.set_xlim(*xlim)
        axis.set_title(title, loc="left", fontweight="normal", fontsize=8.0)
        axis.set_xlabel("AUROC gain (points)")
        axis.set_yticks(y_positions)
        if panel_index == 0:
            axis.set_yticklabels([short_label[label] for label in prediction["target_label"]])
        else:
            axis.set_yticklabels([])
        axis.set_ylim(len(prediction) - 0.5, -0.5)
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.5)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=2.0, labelsize=6.8)

    axes[0].set_position([0.065, 0.190, 0.285, 0.700])
    axes[1].set_position([0.565, 0.190, 0.130, 0.700])
    axes[2].set_position([0.825, 0.190, 0.155, 0.700])
    output = FIGURE_DIR / "path_outcome_profiles"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_layer_selection_profile(layer_selection: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(3.45, 1.72))
    axis.axhline(0, color="#999999", linewidth=0.7)
    axis.plot(layer_selection["depth_percent"], layer_selection["learned_target_span"], color=BLUE, marker="o", linewidth=1.7, label="Learned")
    axis.plot(layer_selection["depth_percent"], layer_selection["random_target_span"], color=GRAY, marker="s", linewidth=1.4, label="Random")
    selected = layer_selection[layer_selection["selected"]]
    axis.scatter(selected["depth_percent"], selected["learned_target_span"], s=75, facecolor="none", edgecolor=GREEN, linewidth=1.5, zorder=4, label="Selected")
    axis.set_xlabel("Relative model depth (%)")
    axis.set_ylabel(r"Target span $\Delta_{+1}-\Delta_{-1}$")
    axis.set_xticks(layer_selection["depth_percent"], [f"{value:.0f}" for value in layer_selection["depth_percent"]])
    axis.grid(color="#E5E7EB", linewidth=0.55)
    axis.legend(frameon=False, loc="upper right", ncol=1)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.27, top=0.96)
    output = FIGURE_DIR / "layer_selection_profile"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_prediction_heatmaps() -> None:
    prediction = pd.read_csv(DATA_DIR / "prediction_ablation_all_rf_split_average.csv")
    feature_sets = ["B+M", "B+M+L", "M+R", "B+M+R", "B+M+L+R"]
    xlabels = ["$B+M$", "+$L$", "$M+R$", "$B+M+R$", "+$L$ after $R$"]
    ylabels = [str(label).replace(" damage", " dmg.").replace("Clean ", "Clean\n") for label in prediction["target_label"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65))
    for axis, metric, title in [(axes[0], "auroc", "(a) AUROC by outcome"), (axes[1], "ap", "(b) Average precision by outcome")]:
        matrix = prediction[[f"{feature}_{metric}" for feature in feature_sets]].to_numpy()
        axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.60 if metric == "auroc" else 0.08, vmax=0.88 if metric == "auroc" else 0.41)
        axis.set_xticks(np.arange(len(xlabels)), xlabels)
        axis.set_yticks(np.arange(len(ylabels)), ylabels)
        axis.set_title(title, loc="left", fontweight="bold")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                color = "white" if value > (0.76 if metric == "auroc" else 0.25) else "#1F2937"
                axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=7.2, color=color)
        axis.axvline(1.5, color="white", linewidth=1.2)
        axis.tick_params(axis="x", rotation=22)
        axis.tick_params(length=0)
    axes[1].set_yticklabels([])
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.17, top=0.90, wspace=0.20)
    output = FIGURE_DIR / "prediction_heatmaps"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_split_figure() -> None:
    split = pd.read_csv(DATA_DIR / "prediction_split_macro.csv")
    split_order = ["record", "dataset", "domain"]
    labels = ["Record", "Dataset", "Domain"]
    fig, axis = plt.subplots(figsize=(3.35, 2.25))
    for feature_set, color, marker, label in [
        ("B+M", GRAY, "s", "Static prior $B+M$"),
        ("B+M+L", ORANGE, "^", "+ localization $L$"),
        ("B+M+R", BLUE, "o", "+ weak response $R$"),
    ]:
        values = [float(split.loc[split["split"].eq(item) & split["feature_set"].eq(feature_set), "auroc_mean"].iloc[0]) for item in split_order]
        axis.plot(labels, values, color=color, marker=marker, linewidth=1.6, markersize=4.5, label=label)
    axis.set_ylim(0.60, 0.88)
    axis.set_ylabel("Mean AUROC across 8 outcomes")
    axis.set_title("Grouped-transfer robustness", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    axis.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.50))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.88)
    output = FIGURE_DIR / "prediction_split_robustness"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_main_table()
    layer_selection = build_layer_selection()
    write_prediction_table()
    make_path_outcome_profiles()
    make_layer_selection_profile(layer_selection)
    make_prediction_heatmaps()
    make_split_figure()
    print("Revised main-paper tables and figures written")


if __name__ == "__main__":
    main()
