#!/usr/bin/env python3
"""Build outcome-foundation and AUROC-centered manuscript assets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
PML_ROOT = PAPER_ROOT.parents[1]
RESULT_ROOT = PML_ROOT / "results/fresh_multidomain_3000/stage_activation_pml"
PRIMARY_ROOT = RESULT_ROOT / "main_3000/qwen3_1_7b"
REPLICATION_ROOT = RESULT_ROOT / "replication_500_residual_rms_v2"
FROZEN_ROOT = PML_ROOT / "data/pml_fresh_multidomain_3000_v1_frozen"
TABLE_DIR = PAPER_ROOT / "tables"
FIGURE_DIR = PAPER_ROOT / "figures"
DATA_DIR = PAPER_ROOT / "data"

MODEL_SPECS = [
    ("qwen3_1_7b", "Qwen3-1.7B-Base", 28, {-21: 7, -17: 11}),
    ("qwen3_5_2b_base", "Qwen3.5-2B-Base", 24, {-18: 6, -15: 9}),
    ("ministral_3_3b_base_2512", "Ministral-3-3B-Base", 26, {-20: 6, -16: 10}),
]
METHODS = ["random", "mean_difference", "logistic", "rfm_agop_top1"]
PRIMARY_METHODS = ["random", "mean_difference", "logistic", "linear", "rfm_agop_top1"]
METHOD_LABELS = {
    "random": "Random",
    "mean_difference": "Mean diff.",
    "logistic": "Logistic",
    "linear": "Linear",
    "rfm_agop_top1": "RFM/AGOP",
}
MODEL_COLORS = {
    "Qwen3-1.7B-Base": "#0072B2",
    "Qwen3.5-2B-Base": "#E69F00",
    "Ministral-3-3B-Base": "#009E73",
}
METHOD_COLORS = {
    "random": "#4B5563",
    "mean_difference": "#0072B2",
    "logistic": "#D55E00",
    "rfm_agop_top1": "#009E73",
}
METHOD_MARKERS = {
    "random": "s",
    "mean_difference": "o",
    "logistic": "D",
    "rfm_agop_top1": "^",
}
LATER_TARGETS = [
    "later_suppression_path",
    "later_enhancement_path",
    "later_target_any_path",
    "later_neighbor_damage_any_path",
    "later_capability_damage_any_path",
    "later_clean_suppression_path",
    "later_clean_enhancement_path",
    "later_clean_any_path",
]
OUTCOME_LABELS = {
    "later_suppression_path": "Suppression",
    "later_enhancement_path": "Enhancement",
    "later_target_any_path": "Target-any",
    "later_neighbor_damage_any_path": "Neighbor damage",
    "later_capability_damage_any_path": "Capability damage",
    "later_clean_suppression_path": "Clean suppression",
    "later_clean_enhancement_path": "Clean enhancement",
    "later_clean_any_path": "Clean-any",
}

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.0,
    }
)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def load_outcomes(model_key: str) -> pd.DataFrame:
    if model_key == "qwen3_1_7b":
        path = PRIMARY_ROOT / "prediction_analysis_strict_later/outcome_summary.csv"
        frame = pd.read_csv(path)
    else:
        path = (
            REPLICATION_ROOT
            / model_key
            / "outcome_analysis_later_strength/strict_later_outcome_summary.csv"
        )
        frame = pd.read_csv(path)
        frame = frame[frame["scope"].eq("method_layer")].copy()
    frame["layer"] = pd.to_numeric(frame["layer"])
    return frame


def load_replication_outcomes(model_key: str) -> pd.DataFrame:
    path = (
        REPLICATION_ROOT
        / model_key
        / "outcome_analysis_later_strength/strict_later_outcome_summary.csv"
    )
    frame = pd.read_csv(path)
    frame = frame[frame["scope"].eq("method_layer")].copy()
    frame["layer"] = pd.to_numeric(frame["layer"])
    return frame


def write_primary_outcome_table() -> None:
    frame = load_outcomes("qwen3_1_7b")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Primary intervention-path outcomes on 3,000 Qwen3-1.7B-Base records. Entries are record-level path-incidence rates (\%), not predictor scores. Target is the union of later suppression and enhancement; Clean additionally requires neither semantic-neighbor nor capability damage at the effective coefficient. $\Delta$Target and $\Delta$Clean are learned-minus-random percentage-point differences within a layer.}",
        r"\label{tab:main-results}",
        r"\small",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\begin{tabular}{clrrrrrr}",
        r"\toprule",
        r"Layer & Direction & Target & N-dmg. & C-dmg. & Clean & $\Delta$Target & $\Delta$Clean \\",
        r"\midrule",
    ]
    for layer_index, (layer, display_layer) in enumerate([(-21, 7), (-17, 11)]):
        layer_frame = frame[frame["layer"].eq(layer)].set_index("method")
        random_row = layer_frame.loc["random"]
        learned = layer_frame.loc[[method for method in PRIMARY_METHODS if method != "random"]]
        best_target = learned["later_target_any_path_rate"].max()
        best_clean = learned["later_clean_any_path_rate"].max()
        if layer_index:
            lines.append(r"\addlinespace[1.2pt]")
        for method in PRIMARY_METHODS:
            row = layer_frame.loc[method]
            target = 100 * float(row["later_target_any_path_rate"])
            neighbor = 100 * float(row["later_neighbor_damage_any_path_rate"])
            capability = 100 * float(row["later_capability_damage_any_path_rate"])
            clean = 100 * float(row["later_clean_any_path_rate"])
            target_text = f"{target:.1f}"
            clean_text = f"{clean:.1f}"
            delta_target = "--"
            delta_clean = "--"
            if method != "random":
                delta_target_value = 100 * (
                    float(row["later_target_any_path_rate"])
                    - float(random_row["later_target_any_path_rate"])
                )
                delta_clean_value = 100 * (
                    float(row["later_clean_any_path_rate"])
                    - float(random_row["later_clean_any_path_rate"])
                )
                delta_target = f"{delta_target_value:+.1f}"
                delta_clean = f"{delta_clean_value:+.1f}"
                if np.isclose(float(row["later_target_any_path_rate"]), best_target):
                    target_text = rf"\textbf{{{target_text}}}"
                    delta_target = rf"\textbf{{{delta_target}}}"
                if np.isclose(float(row["later_clean_any_path_rate"]), best_clean):
                    clean_text = rf"\textbf{{{clean_text}}}"
                    delta_clean = rf"\textbf{{{delta_clean}}}"
            lines.append(
                f"{display_layer} & {METHOD_LABELS[method]} & {target_text} & "
                f"{neighbor:.1f} & {capability:.1f} & {clean_text} & "
                f"{delta_target} & {delta_clean} " + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (TABLE_DIR / "main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def predictor_macro(path: Path, split: str, feature_set: str) -> tuple[float, float]:
    frame = pd.read_csv(path)
    rows = frame[
        frame["cohort"].eq("all")
        & frame["model"].eq("random_forest")
        & frame["split"].eq(split)
        & frame["feature_set"].eq(feature_set)
        & frame["target"].isin(LATER_TARGETS)
    ]
    if len(rows) != len(LATER_TARGETS):
        raise ValueError(f"Expected eight later outcomes in {path}, got {len(rows)}")
    return float(rows["auroc_mean"].mean()), float(rows["average_precision_mean"].mean())


def prediction_path(model_key: str, primary: bool = False) -> Path:
    if primary:
        return PRIMARY_ROOT / "prediction_analysis_strict_later/prediction_summary.csv"
    return REPLICATION_ROOT / model_key / "prediction_analysis_strict_later/prediction_summary.csv"


def write_predictor_table() -> None:
    rows = [
        ("Qwen3-1.7B primary", 3000, prediction_path("qwen3_1_7b", primary=True)),
        ("Qwen3-1.7B subset", 500, prediction_path("qwen3_1_7b")),
        ("Qwen3.5-2B-Base", 500, prediction_path("qwen3_5_2b_base")),
        ("Ministral-3B-Base", 500, prediction_path("ministral_3_3b_base_2512")),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Macro AUROC across the eight outcomes visualized in the main paper. $B+M$, $\Delta L$, $\Delta R$, and Final use record-held-out evaluation; Dataset and Domain use the final $B+M+L+R$ predictor. The Qwen3-1.7B subset is extracted from the primary cohort.}",
        r"\label{tab:cross-model-prediction}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
        r"\toprule",
        r"Cohort & $B{+}M$ & $\Delta L$ & $\Delta R$ & Final & Dataset & Domain \\",
        r"\midrule",
    ]
    source_rows = []
    for label, records, path in rows:
        bm, _ = predictor_macro(path, "record", "B+M")
        bml, _ = predictor_macro(path, "record", "B+M+L")
        bmlr, ap = predictor_macro(path, "record", "B+M+L+R")
        dataset, _ = predictor_macro(path, "dataset", "B+M+L+R")
        domain, _ = predictor_macro(path, "domain", "B+M+L+R")
        delta_l = 100 * (bml - bm)
        delta_r = 100 * (bmlr - bml)
        compact_label = {
            "Qwen3-1.7B primary": "Qwen3-1.7B primary",
            "Qwen3-1.7B subset": "Qwen3-1.7B subset",
            "Qwen3.5-2B-Base": "Qwen3.5-2B",
            "Ministral-3B-Base": "Ministral-3B",
        }[label]
        lines.append(
            f"{compact_label} & {bm:.3f} & {delta_l:+.1f} & "
            rf"\textbf{{{delta_r:+.1f}}} & \textbf{{{bmlr:.3f}}} & "
            f"{dataset:.3f} & {domain:.3f} " + r"\\"
        )
        source_rows.append(
            {
                "cohort": label,
                "records": records,
                "record_BM_auroc": bm,
                "record_BML_auroc": bml,
                "record_BMLR_auroc": bmlr,
                "record_delta_L_points": delta_l,
                "record_delta_R_points": delta_r,
                "record_BMLR_ap": ap,
                "dataset_BMLR_auroc": dataset,
                "domain_BMLR_auroc": domain,
            }
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])
    (TABLE_DIR / "cross_model_replication.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    pd.DataFrame(source_rows).to_csv(DATA_DIR / "main_predictor_summary.csv", index=False)


def load_jsonl(path: Path, key: str) -> dict[str, dict[str, object]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row[key])] = row
    return rows


def write_worked_record_tables() -> None:
    records = load_jsonl(FROZEN_ROOT / "records.jsonl", "record_id")
    eval_bank = load_jsonl(FROZEN_ROOT / "eval_bank.jsonl", "eval_id")
    assignments = load_jsonl(FROZEN_ROOT / "eval_assignments.jsonl", "record_id")
    assignment = assignments["13c_polarization_relaxation_time"]
    record = records["13c_polarization_relaxation_time"]
    target = eval_bank[assignment["target_eval_ids"][0]]
    neighbor = eval_bank[assignment["neighbor_eval_ids"][0]]
    capability = eval_bank[assignment["capability_eval_ids"][0]]

    def clean_text(value: object) -> str:
        return str(value).strip().strip('"“”')

    lines = [
        r"\begin{center}",
        r"\begin{minipage}{\columnwidth}",
        r"\centering",
        r"\captionof*{table}{\textbf{Worked example.} One frozen record from construction to path label; direction-fitting statements and evaluation probes are disjoint.}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{@{}p{0.20\columnwidth}p{0.74\columnwidth}@{}}",
        r"\toprule",
        r"Object & Frozen example \\",
        r"\midrule",
        rf"Source & MMLU-Redux 2.0 chemistry: {tex_escape(clean_text(record['source']['question']))} \\",
        rf"Fit evidence & Positive statement: {tex_escape(clean_text(record['train_positive'][0]))} Contrast statement: {tex_escape(clean_text(record['train_negative'][0]))} \\",
        rf"Target probe & Prompt: {tex_escape(clean_text(target['prompt']))}; candidate answers: {tex_escape(clean_text(target['correct']))} vs. {tex_escape(clean_text(target['contrast']))}. \\",
        rf"Neighbor probe & Prompt: {tex_escape(clean_text(neighbor['prompt']))}; candidate answers: {tex_escape(clean_text(neighbor['correct']))} vs. {tex_escape(clean_text(neighbor['contrast']))}. \\",
        rf"Capability probe & Prompt: {tex_escape(clean_text(capability['prompt']))}; candidate answers: {tex_escape(clean_text(capability['correct']))} vs. {tex_escape(clean_text(capability['contrast']))}. \\",
        r"Measured path & Mean difference, layer 11: suppression onset $-0.25$; positive neighbor onset $0.5$; no enhancement or capability onset. \\",
        r"Labels & Target=1, N-dmg.=1, C-dmg.=0, Clean suppression=1, Clean=1. \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{minipage}",
        r"\end{center}",
        "",
    ]
    (TABLE_DIR / "main_worked_record.tex").write_text("\n".join(lines), encoding="utf-8")

    paths = pd.read_csv(
        PRIMARY_ROOT / "prediction_dataset_strict_later/pml_path_prediction_dataset.csv"
    )
    examples = [
        ("Mixed clean/damage", "13c_polarization_relaxation_time", "mean_difference", -17),
        ("Clean only", "ferret_beverage_soy_milk", "rfm_agop_top1", -17),
        ("Collateral, no clean", "coal_mines_energy", "rfm_agop_top1", -21),
        ("No effect", "acid_spill_water_01", "logistic", -21),
    ]
    appendix_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Additional frozen-record path examples. Onsets are the first threshold crossings on the negative/positive sides; a dash denotes no crossing. These examples illustrate the label taxonomy and are not selected as quantitative evidence.}",
        r"\label{tab:worked-records-appendix}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Type & Source record & Direction & Layer & Supp. onset & Enh. onset & Damage onset & Clean \\",
        r"\midrule",
    ]
    for category, record_id, method, layer in examples:
        matches = paths[
            paths["record_id"].eq(record_id)
            & paths["method"].eq(method)
            & paths["layer"].eq(layer)
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing worked path {record_id}/{method}/{layer}")
        row = matches.iloc[0]
        record_row = records[record_id]

        def onset(value: object) -> str:
            return "--" if pd.isna(value) else f"{float(value):+.2f}"

        damage_candidates = [
            value
            for value in [row["negative_damage_onset_alpha"], row["positive_damage_onset_alpha"]]
            if not pd.isna(value)
        ]
        damage = "--" if not damage_candidates else "/".join(onset(value) for value in damage_candidates)
        source = f"{record_id} ({record_row['source']['dataset'].split('/')[-1]})"
        appendix_lines.append(
            f"{category} & {tex_escape(source)} & {METHOD_LABELS[method]} & {28 + int(layer)} & "
            f"{onset(row['suppression_onset_alpha'])} & {onset(row['enhancement_onset_alpha'])} & "
            f"{damage} & {int(row['later_clean_any_path'])} " + r"\\"
        )
    appendix_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (TABLE_DIR / "appendix_worked_examples.tex").write_text(
        "\n".join(appendix_lines), encoding="utf-8"
    )


def write_appendix_cross_model_outcomes() -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete residual-norm-matched 500-record outcomes. Rates are percentages. $\Delta$Target and $\Delta$Clean compare each learned direction with random at the same model and layer.}",
        r"\label{tab:cross-model-outcomes-full}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        r"Model/layer & Direction & Supp. & Enh. & Target & N-dmg. & C-dmg. & Cl. supp. & Cl. enh. & Clean & $\Delta$Target & $\Delta$Clean \\",
        r"\midrule",
    ]
    for model_index, (model_key, model_label, _, layer_map) in enumerate(MODEL_SPECS):
        frame = load_replication_outcomes(model_key).set_index(["layer", "method"])
        if model_index:
            lines.append(r"\midrule")
        for layer_index, (layer, display_layer) in enumerate(layer_map.items()):
            random_row = frame.loc[(layer, "random")]
            if layer_index:
                lines.append(r"\addlinespace[0.8pt]")
            for method_index, method in enumerate(METHODS):
                row = frame.loc[(layer, method)]
                values = [
                    100 * float(row[column])
                    for column in [
                        "later_suppression_path_rate",
                        "later_enhancement_path_rate",
                        "later_target_any_path_rate",
                        "later_neighbor_damage_any_path_rate",
                        "later_capability_damage_any_path_rate",
                        "later_clean_suppression_path_rate",
                        "later_clean_enhancement_path_rate",
                        "later_clean_any_path_rate",
                    ]
                ]
                deltas = ["--", "--"]
                if method != "random":
                    deltas = [
                        f"{100 * (float(row['later_target_any_path_rate']) - float(random_row['later_target_any_path_rate'])):+.1f}",
                        f"{100 * (float(row['later_clean_any_path_rate']) - float(random_row['later_clean_any_path_rate'])):+.1f}",
                    ]
                model_cell = f"{model_label}, {display_layer}" if method_index == 0 else ""
                lines.append(
                    f"{model_cell} & {METHOD_LABELS[method]} & "
                    + " & ".join(f"{value:.1f}" for value in values)
                    + " & "
                    + " & ".join(deltas)
                    + r" \\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (TABLE_DIR / "appendix_cross_model_outcomes.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def prediction_gain_frame() -> pd.DataFrame:
    rows = []
    for model_key, model_label, _, _ in MODEL_SPECS:
        path = prediction_path(model_key)
        frame = pd.read_csv(path)
        frame = frame[
            frame["cohort"].eq("all")
            & frame["model"].eq("random_forest")
            & frame["split"].eq("record")
            & frame["target"].isin(LATER_TARGETS)
        ]
        pivot = frame.pivot(index="target", columns="feature_set", values="auroc_mean")
        for target in LATER_TARGETS:
            rows.append(
                {
                    "model": model_label,
                    "target": target,
                    "target_label": OUTCOME_LABELS[target],
                    "delta_L_points": 100 * (pivot.loc[target, "B+M+L"] - pivot.loc[target, "B+M"]),
                    "delta_R_points": 100 * (pivot.loc[target, "B+M+L+R"] - pivot.loc[target, "B+M+L"]),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(DATA_DIR / "figure2_cross_model_prediction_gains.csv", index=False)
    return result


def outcome_tradeoff_frame() -> pd.DataFrame:
    rows = []
    for model_key, model_label, model_depth, layer_map in MODEL_SPECS:
        frame = load_replication_outcomes(model_key).set_index(["layer", "method"])
        for depth_order, (layer, display_layer) in enumerate(layer_map.items()):
            random_row = frame.loc[(layer, "random")]
            for method_order, method in enumerate(METHODS[1:]):
                row = frame.loc[(layer, method)]
                rows.append(
                    {
                        "model": model_label,
                        "layer": display_layer,
                        "relative_depth": 100 * display_layer / (model_depth - 1),
                        "depth_order": depth_order,
                        "method": method,
                        "method_order": method_order,
                        "target_rate_percent": 100
                        * float(row["later_target_any_path_rate"]),
                        "collateral_rate_percent": 50
                        * (
                            float(row["later_neighbor_damage_any_path_rate"])
                            + float(row["later_capability_damage_any_path_rate"])
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(DATA_DIR / "figure2_cross_model_tradeoffs.csv", index=False)
    return result


def make_figure2() -> None:
    outcome = outcome_tradeoff_frame()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.34), sharex=False, sharey=False)
    panel_titles = {
        "Qwen3-1.7B-Base": "(a) Qwen3-1.7B-Base",
        "Qwen3.5-2B-Base": "(b) Qwen3.5-2B-Base",
        "Ministral-3-3B-Base": "(c) Ministral-3-3B-Base",
    }
    for axis, (model_key, model_label, _, layer_map) in zip(axes, MODEL_SPECS):
        group = outcome[outcome["model"].eq(model_label)]
        random_frame = load_replication_outcomes(model_key).set_index(["layer", "method"])
        panel_x = group["collateral_rate_percent"].astype(float).tolist()
        panel_y = group["target_rate_percent"].astype(float).tolist()
        for layer, display_layer in layer_map.items():
            random_row = random_frame.loc[(layer, "random")]
            random_x = 50 * (
                float(random_row["later_neighbor_damage_any_path_rate"])
                + float(random_row["later_capability_damage_any_path_rate"])
            )
            random_y = 100 * float(random_row["later_target_any_path_rate"])
            panel_x.append(random_x)
            panel_y.append(random_y)
            axis.scatter(
                random_x,
                random_y,
                marker="o" if display_layer == min(layer_map.values()) else "^",
                s=35,
                facecolor="#9CA3AF",
                edgecolor="white",
                linewidth=0.65,
                zorder=3,
            )
        for _, row in group.iterrows():
            axis.scatter(
                float(row["collateral_rate_percent"]),
                float(row["target_rate_percent"]),
                marker="o" if int(row["depth_order"]) == 0 else "^",
                s=38,
                facecolor=METHOD_COLORS[str(row["method"])],
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
        axis.set_title(panel_titles[model_label], loc="left")
        axis.set_xlabel("Mean collateral incidence (%)")
        axis.grid(color="#E5E7EB", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        x_span = max(max(panel_x) - min(panel_x), 1.0)
        y_span = max(max(panel_y) - min(panel_y), 2.0)
        axis.set_xlim(max(0.0, min(panel_x) - 0.16 * x_span), max(panel_x) + 0.16 * x_span)
        axis.set_ylim(max(0.0, min(panel_y) - 0.14 * y_span), max(panel_y) + 0.14 * y_span)
    axes[0].set_ylabel("Target-any incidence (%)")

    method_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="none",
                         markerfacecolor=METHOD_COLORS[method], markeredgecolor="white",
                         markersize=5.9, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    depth_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="none", color="#374151",
                         markersize=5.4, label="Shallower block"),
        mpl.lines.Line2D([], [], marker="^", linestyle="none", color="#374151",
                         markersize=5.4, label="Deeper block"),
    ]
    fig.legend(
        handles=method_handles + depth_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.04),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.78, wspace=0.18)
    output = FIGURE_DIR / "path_outcome_profiles"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_cross_model_prediction_heatmap() -> None:
    outcome_order = LATER_TARGETS
    column_specs = [
        ("B", "random_forest"),
        ("B+M", "random_forest"),
        ("B+M+L", "random_forest"),
        ("B+M+L+R", "logistic"),
        ("B+M+L+R", "random_forest"),
    ]
    column_labels = [
        r"$B$",
        r"$B{+}M$",
        r"$B{+}M{+}L$",
        "Final (linear)",
        "Final (RF)",
    ]
    row_labels = ["Supp.", "Enh.", "Target", "N-dmg.", "C-dmg.", "Cl.-S", "Cl.-E", "Clean"]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.38), sharey=True)
    color_norm = mpl.colors.Normalize(vmin=0.50, vmax=0.90)
    color_map = mpl.colormaps["Blues"]
    source_rows = []
    for axis, (model_key, model_label, _, _) in zip(axes, MODEL_SPECS):
        frame = pd.read_csv(prediction_path(model_key))
        frame = frame[
            frame["cohort"].eq("all")
            & frame["split"].eq("record")
            & frame["target"].isin(outcome_order)
        ]
        values = np.column_stack(
            [
                frame[
                    frame["feature_set"].eq(feature_set)
                    & frame["model"].eq(predictor)
                ]
                .set_index("target")
                .loc[outcome_order, "auroc_mean"]
                .to_numpy(dtype=float)
                for feature_set, predictor in column_specs
            ]
        )
        for row_index, target in enumerate(outcome_order):
            for column_index, (feature_set, predictor) in enumerate(column_specs):
                value = values[row_index, column_index]
                axis.add_patch(
                    mpl.patches.Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        facecolor=color_map(color_norm(value)),
                        edgecolor="#F3F4F6",
                        linewidth=0.45,
                    )
                )
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=5.7,
                    color="white" if value >= 0.76 else "#1F2937",
                )
                source_rows.append(
                    {
                        "model": model_label,
                        "target": target,
                        "feature_set": feature_set,
                        "predictor": predictor,
                        "auroc": value,
                    }
                )
        short_title = model_label.replace("-Base-2512", "").replace("-Base", "")
        axis.set_xlim(-0.5, len(column_labels) - 0.5)
        axis.set_ylim(len(outcome_order) - 0.5, -0.5)
        axis.set_aspect("auto")
        axis.set_title(short_title, fontweight="bold", fontsize=8.3)
        axis.set_xticks(range(len(column_labels)), column_labels, rotation=27, ha="right")
        axis.tick_params(length=0, axis="x", labelsize=5.9)
        axis.tick_params(length=0, axis="y")
        for spine in axis.spines.values():
            spine.set_color("#D1D5DB")
            spine.set_linewidth(0.6)
    axes[0].set_yticks(
        range(len(outcome_order)), row_labels
    )
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.24, top=0.88, wspace=0.07)
    colorbar_axis = fig.add_axes([0.93, 0.24, 0.012, 0.58])
    color_edges = np.linspace(0.50, 0.90, 41)
    for lower, upper in zip(color_edges[:-1], color_edges[1:]):
        colorbar_axis.add_patch(
            mpl.patches.Rectangle(
                (0.0, lower),
                1.0,
                upper - lower,
                facecolor=color_map(color_norm((lower + upper) / 2.0)),
                edgecolor="none",
            )
        )
    colorbar_axis.set_xlim(0.0, 1.0)
    colorbar_axis.set_ylim(0.50, 0.90)
    colorbar_axis.set_xticks([])
    colorbar_axis.set_yticks([0.50, 0.60, 0.70, 0.80, 0.90])
    colorbar_axis.yaxis.tick_right()
    colorbar_axis.yaxis.set_label_position("right")
    colorbar_axis.set_ylabel("AUROC")
    output = FIGURE_DIR / "cross_model_prediction_heatmap"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(source_rows).to_csv(
        DATA_DIR / "cross_model_prediction_heatmap.csv", index=False
    )


def write_cross_model_prose() -> None:
    snippets = {
        "cross_model_abstract_generated.tex": (
            "Across three residual-norm-matched base models, learned directions retain\n"
            "selective-path gains and low-dose responses yield $0.801$--$0.828$\n"
            "record-held-out macro AUROC.\n"
        ),
        "cross_model_discussion_generated.tex": (
            "\\paragraph{Budget matching exposes model-dependent leverage without changing the diagnostic hierarchy.}\n"
            "Residual-norm matching aligns the intervention magnitude relative to the residual\n"
            "stream; it does not force equal response rates. Qwen3.5 exhibits substantially\n"
            "larger learned-over-random path gains than the reference model, while Ministral\n"
            "falls between them. Nevertheless, every model reproduces the two findings that\n"
            "matter for PML: learned directions create more clean paths than random controls,\n"
            "and a disjoint low-dose response is far more predictive than static\n"
            "localization alone.\n"
        ),
        "cross_model_limitations_generated.tex": (
            "The cross-model confirmations use 500 frozen records per model and\n"
            "model-specific random-null thresholds. They establish replication of\n"
            "within-model contrasts and predictor ordering, but they are not powered for\n"
            "fine-grained comparisons of absolute incidence across architectures. The\n"
            "Qwen3-1.7B 500-record block is a subset extraction from the primary outcomes,\n"
            "so its small deviations from the 3,000-record rates reflect subset composition,\n"
            "threshold recalibration, and sampling variability rather than a new steering\n"
            "scale.\n"
        ),
        "cross_model_conclusion_generated.tex": (
            "Width-corrected residual-norm confirmations on Qwen3.5-2B-Base and\n"
            "Ministral-3-3B-Base preserve these conclusions across architectures.\n"
        ),
        "cross_model_results_generated.tex": "",
    }
    section_dir = PAPER_ROOT / "sections"
    for name, text in snippets.items():
        (section_dir / name).write_text(text, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_predictor_table()
    write_worked_record_tables()
    write_appendix_cross_model_outcomes()
    make_figure2()
    make_cross_model_prediction_heatmap()
    write_cross_model_prose()
    print("Outcome/AUROC manuscript assets written")


if __name__ == "__main__":
    main()
