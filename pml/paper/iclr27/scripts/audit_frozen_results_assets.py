#!/usr/bin/env python3
"""Audit paper-facing PML tables and references against frozen result snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PAPER_ROOT / "tables"
SECTION_DIR = PAPER_ROOT / "sections"
DATA_DIR = PAPER_ROOT / "data"
RESULT_ROOT = PAPER_ROOT.parents[1] / "results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b"
SELECTOR_PAIRED = (
    RESULT_ROOT
    / "strength_predictor_multifidelity_dense/strength_policy_paired_comparisons.csv"
)

OUTCOME_LABELS = {
    "linear": "Linear",
    "logistic": "Logistic",
    "matched_norm_random": "Matched random",
    "mean_difference": "Mean difference",
    "random": "Random",
    "rfm_agop_top1": "RFM/AGOP top-1",
}
OUTCOME_COLUMNS = [
    "later_suppression_path_rate",
    "later_enhancement_path_rate",
    "later_target_any_path_rate",
    "later_neighbor_damage_any_path_rate",
    "later_capability_damage_any_path_rate",
    "later_clean_suppression_path_rate",
    "later_clean_enhancement_path_rate",
    "later_clean_any_path_rate",
]
FEATURE_SETS = ["B+M", "B+M+L", "M+R", "B+M+R", "B+M+L+R"]
ENDPOINT_LABELS = {
    "random": "Random",
    "mean_difference": "Mean diff.",
    "logistic": "Logistic",
    "rfm_agop_top1": "RFM/AGOP",
}
PAIRED_LABELS = {
    "mean_difference": "Mean difference",
    "linear": "Linear",
    "logistic": "Logistic",
    "rfm_agop_top1": "RFM/AGOP top-1",
}
PAIRED_METRICS = [
    "later_target_any_path",
    "later_clean_any_path",
    "later_neighbor_damage_any_path",
    "later_capability_damage_any_path",
]
MODEL_DEPTHS = {
    "qwen3_1_7b": 28,
    "qwen3_5_2b_base": 24,
    "ministral_3_3b_base_2512": 26,
}
MODEL_LABELS = {
    "qwen3_1_7b": "Qwen3-1.7B-Base",
    "qwen3_5_2b_base": "Qwen3.5-2B-Base",
    "ministral_3_3b_base_2512": "Ministral-3-3B-Base",
}
MODEL_METHODS = {
    "qwen3_1_7b": ["random", "mean_difference", "logistic", "linear", "rfm_agop_top1"],
    "qwen3_5_2b_base": ["random", "mean_difference", "logistic", "rfm_agop_top1"],
    "ministral_3_3b_base_2512": ["random", "mean_difference", "logistic", "rfm_agop_top1"],
}


def forward_layer(layer: int, depth: int = 28) -> int:
    return layer if layer >= 0 else depth + layer


def tex_dependency_closure(root: Path) -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in seen:
            return
        assert path.exists(), f"Missing LaTeX source: {path}"
        seen.add(path)
        ordered.append(path)
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\\input\{([^}]+)\}", text):
            child = PAPER_ROOT / target
            if not child.suffix:
                child = child.with_suffix(".tex")
            visit(child)

    visit(root)
    return ordered


def table_rows(path: Path) -> dict[tuple[str, int] | str, list[float]]:
    rows: dict[tuple[str, int] | str, list[float]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in raw_line or "\\" not in raw_line:
            continue
        cells = [cell.strip() for cell in raw_line.split("&")]
        name = cells[0]
        if name in {"Direction", "Outcome", "Split", "Selector"}:
            continue
        cleaned = [re.sub(r"[^0-9.\-]", "", cell) for cell in cells[1:]]
        if path.name == "appendix_full_outcomes.tex":
            layer = int(float(cleaned[0]))
            rows[(name, layer)] = [float(value) for value in cleaned[1:]]
        elif path.name in {"appendix_prediction_auroc.tex", "appendix_prediction_ap.tex"}:
            rows[name] = [float(value) for value in cleaned]
    return rows


def audit_outcomes() -> None:
    source = pd.read_csv(DATA_DIR / "frozen_outcome_summary.csv")
    rendered = table_rows(TABLE_DIR / "appendix_full_outcomes.tex")
    for _, row in source.iterrows():
        key = (
            OUTCOME_LABELS[str(row["method"])],
            forward_layer(int(row["layer"])),
        )
        expected = [round(100 * float(row[column]), 1) for column in OUTCOME_COLUMNS]
        actual = rendered.get(key)
        assert actual == expected, f"Outcome mismatch for {key}: {actual} != {expected}"

    main_text = (TABLE_DIR / "main_results.tex").read_text(encoding="utf-8")
    assert "Random-calibrated path incidence" in main_text
    assert "$\\Delta$T" in main_text and "$\\Delta$C" in main_text
    for block in [
        "A. Qwen3-1.7B-Base",
        "B. Qwen3-1.7B-Base",
        "C. Qwen3.5-2B-Base",
        "D. Ministral-3-3B-Base",
    ]:
        assert block in main_text, f"Missing complete main outcome block: {block}"
    assert "70.4" not in main_text and "+62.8" not in main_text
    primary_text = main_text.split("A. Qwen3-1.7B-Base", 1)[1].split(
        "B. Qwen3-1.7B-Base", 1
    )[0]

    label_by_method = {
        "random": "Random",
        "mean_difference": "Mean diff.",
        "linear": "Linear",
        "logistic": "Logistic",
        "rfm_agop_top1": "RFM/AGOP",
    }
    for _, row in source.iterrows():
        method = str(row["method"])
        if method not in label_by_method:
            continue
        layer = forward_layer(int(row["layer"]))
        prefix = f"{layer} & {label_by_method[method]} & "
        matching_lines = [line for line in primary_text.splitlines() if line.startswith(prefix)]
        assert len(matching_lines) == 1, f"Missing main outcome row: {prefix}"
        rendered_values = [
            float(value)
            for value in re.findall(
                r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", matching_lines[0]
            )
        ][1:9]
        expected = [round(100 * float(row[column]), 1) for column in OUTCOME_COLUMNS]
        assert rendered_values == expected, (
            f"Main outcome mismatch for {method}, layer {layer}: "
            f"{rendered_values} != {expected}"
        )


def audit_dataset_composition() -> None:
    source = pd.read_csv(DATA_DIR / "dataset_composition.csv")
    table_text = (TABLE_DIR / "appendix_dataset_composition.tex").read_text(
        encoding="utf-8"
    )
    rendered: dict[str, int] = {}
    for raw_line in table_text.splitlines():
        if "&" not in raw_line or "\\" not in raw_line:
            continue
        cells = [cell.strip().rstrip("\\").strip() for cell in raw_line.split("&")]
        if len(cells) != 2 or cells[0] == "Source dataset":
            continue
        numeric = re.sub(r"[^0-9]", "", cells[1])
        if numeric:
            rendered[cells[0]] = int(numeric)
    expected = {
        str(row["source_label"]): int(row["records"])
        for _, row in source.iterrows()
    }
    assert rendered.get("Total") == sum(expected.values()) == 3000
    rendered.pop("Total", None)
    assert rendered == expected, f"Dataset composition mismatch: {rendered} != {expected}"


def audit_scale_manifest() -> None:
    scale = json.loads((DATA_DIR / "experiment_scale.json").read_text(encoding="utf-8"))
    probes = sum(int(value) for value in scale["probes_per_record"].values())
    paths = int(scale["unique_records"]) * int(scale["methods"]) * len(scale["layers"])
    assert int(scale["method_layer_paths"]) == paths == 36_000
    assert int(scale["probe_trajectories"]) == paths * probes == 360_000
    path_strength = paths * len(scale["strengths"])
    assert int(scale["path_strength_evaluations"]) == path_strength == 252_000
    probe_strength = path_strength * probes
    assert int(scale["probe_strength_observations"]) == probe_strength == 2_520_000
    assert int(scale["probe_level_evaluations"]) == probe_strength
    assert int(scale["continuation_likelihood_evaluations"]) == 2 * probe_strength


def audit_prediction() -> None:
    source = pd.read_csv(DATA_DIR / "prediction_ablation_all_rf_split_average.csv")
    for metric, filename in [
        ("auroc", "appendix_prediction_auroc.tex"),
        ("ap", "appendix_prediction_ap.tex"),
    ]:
        rendered = table_rows(TABLE_DIR / filename)
        for _, row in source.iterrows():
            expected = [round(float(row[f"{feature_set}_{metric}"]), 3) for feature_set in FEATURE_SETS]
            actual = rendered.get(str(row["target_label"]))
            assert actual == expected, (
                f"Prediction mismatch for {row['target_label']} {metric}: {actual} != {expected}"
            )

    main_text = (TABLE_DIR / "prediction_ablation.tex").read_text(encoding="utf-8")
    main_rows: dict[str, list[float]] = {}
    for raw_line in main_text.splitlines():
        if "&" not in raw_line or "\\" not in raw_line:
            continue
        cells = [cell.strip() for cell in raw_line.split("&")]
        if cells[0] == "Added evidence":
            continue
        values = []
        for cell in cells[1:3]:
            match = re.search(r"[-+]?\d*\.\d+", cell)
            if match:
                values.append(float(match.group(0)))
        if len(values) == 2:
            main_rows[cells[0]] = values

    strict = pd.read_csv(RESULT_ROOT / "prediction_analysis_strict_later/prediction_summary.csv")
    response = pd.read_csv(
        RESULT_ROOT / "prediction_analysis_strict_later_r_ablation/prediction_summary.csv"
    )
    later_targets = set(response["target"])
    keys = ["cohort", "split", "target", "model"]

    def matched_delta(
        base: pd.DataFrame,
        base_feature: str,
        new: pd.DataFrame,
        new_feature: str,
    ) -> list[float]:
        left = base.loc[
            base["feature_set"].eq(base_feature) & base["target"].isin(later_targets)
        ]
        right = new.loc[
            new["feature_set"].eq(new_feature) & new["target"].isin(later_targets)
        ]
        matched = left.merge(right, on=keys, suffixes=("_base", "_new"))
        assert len(matched) == 144
        return [
            round(float((matched["auroc_mean_new"] - matched["auroc_mean_base"]).mean()), 3),
            round(
                float(
                    (
                        matched["average_precision_mean_new"]
                        - matched["average_precision_mean_base"]
                    ).mean()
                ),
                3,
            ),
        ]

    expected_rows = {
        "Static $L$ over $B+M$": matched_delta(strict, "B+M", strict, "B+M+L"),
        "Weak $R$ over $B+M$": matched_delta(strict, "B+M", response, "B+M+R"),
        "$L$ after $R$": matched_delta(response, "B+M+R", response, "B+M+L+R"),
    }
    assert main_rows == expected_rows, f"Main prediction summary mismatch: {main_rows} != {expected_rows}"


def audit_geometry_ablation() -> None:
    source = pd.read_csv(DATA_DIR / "geometry_ablation_rfm.csv")
    table_text = (TABLE_DIR / "appendix_geometry_ablation.tex").read_text(encoding="utf-8")
    rendered: dict[str, list[float]] = {}
    for raw_line in table_text.splitlines():
        if "&" not in raw_line or raw_line.startswith("Outcome"):
            continue
        cells = [cell.strip() for cell in raw_line.split("&")]
        if len(cells) != 5:
            continue
        values: list[float] = []
        for cell in cells[1:]:
            match = re.search(r"[-+]?\d*\.\d+", cell)
            if not match:
                break
            values.append(float(match.group(0)))
        if len(values) == 4:
            rendered[cells[0]] = values

    for _, row in source.iterrows():
        expected = [
            round(float(row["B+M+L_auroc"]), 3),
            round(float(row["B+M+L+G_auroc"]), 3),
            round(float(row["delta_auroc"]), 3),
            round(float(row["delta_ap"]), 3),
        ]
        actual = rendered.get(str(row["target_label"]))
        assert actual == expected, (
            f"Geometry ablation mismatch for {row['target_label']}: {actual} != {expected}"
        )


def audit_strength_selector() -> None:
    source = pd.read_csv(DATA_DIR / "strength_selector_summary.csv")
    policy_source = pd.read_csv(DATA_DIR / "strength_policy_evaluation.csv")
    table_path = TABLE_DIR / "appendix_strength_selector.tex"
    table_text = table_path.read_text(encoding="utf-8")
    rendered: dict[tuple[str, str], list[float]] = {}
    oracle_lines: dict[str, str] = {}
    current_direction = ""
    for raw_line in table_text.splitlines():
        if "{Suppression}" in raw_line:
            current_direction = "suppression"
            continue
        if "{Enhancement}" in raw_line:
            current_direction = "enhancement"
            continue
        if raw_line.startswith("Dense oracle"):
            oracle_lines[current_direction] = raw_line
            continue
        if "&" not in raw_line or "\\" not in raw_line:
            continue
        cells = [cell.strip() for cell in raw_line.split("&")]
        if cells[0] in {"", "Selector"} or len(cells) != 5 or not current_direction:
            continue
        cleaned = [re.sub(r"[^0-9.\-]", "", cell) for cell in cells[1:]]
        if all(cleaned):
            rendered[(current_direction, cells[0])] = [float(value) for value in cleaned]

    specifications = {
        "Dense $M+A+R$": ("dense_only", "MAR", "hist_gbdt"),
        "MF $B+M+A$": ("multifidelity", "BMA", "hist_gbdt"),
        "MF $M+A+R$": ("multifidelity", "MAR", "hist_gbdt"),
        "MF $B+M+A+R$": ("multifidelity", "BMAR", "hist_gbdt"),
        "MF $B+M+L+A+R$": ("multifidelity", "BMLAR", "hist_gbdt"),
    }
    metrics = ["clean_rate", "utility_mean", "neighbor_damage_rate", "abstention_rate"]
    for label, (regime, feature_set, model) in specifications.items():
        for direction in ["suppression", "enhancement"]:
            row = source.loc[
                source["regime"].eq(regime)
                & source["feature_set"].eq(feature_set)
                & source["model"].eq(model)
                & source["direction"].eq(direction)
            ].iloc[0]
            expected = [round(float(row[metric]), 3) for metric in metrics]
            actual = rendered.get((direction, label))
            assert actual == expected, (
                f"Strength table mismatch for {direction}, {label}: {actual} != {expected}"
            )

    for table_label, policy_label in {
        "No intervention": "No intervention",
        "Fixed $|\\alpha|=1$": "Train-tuned fixed alpha",
    }.items():
        for direction in ["suppression", "enhancement"]:
            row = policy_source.loc[
                policy_source["policy"].eq(policy_label)
                & policy_source["direction"].eq(direction)
            ].iloc[0]
            expected = [round(float(row[metric]), 3) for metric in metrics]
            actual = rendered.get((direction, table_label))
            assert actual == expected, (
                f"Strength policy mismatch for {direction}, {table_label}: {actual} != {expected}"
            )

    for direction in ["suppression", "enhancement"]:
        row = source.loc[
            source["direction"].eq(direction) & source["oracle_clean_rate"].notna()
        ].iloc[0]
        actual = [float(value) for value in re.findall(r"-?\d*\.\d+", oracle_lines[direction])[:2]]
        expected = [
            round(float(row["oracle_clean_rate"]), 3),
            round(float(row["oracle_utility_mean"]), 3),
        ]
        assert actual == expected, f"Oracle mismatch for {direction}: {actual} != {expected}"

    efficiency = (TABLE_DIR / "appendix_strength_efficiency.tex").read_text(encoding="utf-8")
    multi = policy_source[policy_source["policy"] == "Multi-fidelity M+A+R selector"]
    for direction in ["suppression", "enhancement"]:
        row = multi[multi["direction"] == direction].iloc[0]
        assert f"{float(row['expected_evaluations']):.3f}" in efficiency
        assert f"{100 * float(row['evaluation_reduction_vs_dense']):.1f}\\%" in efficiency

    main_table = (TABLE_DIR / "main_strength_selector.tex").read_text(encoding="utf-8")
    paired_source = pd.read_csv(SELECTOR_PAIRED)
    for direction, label in [("suppression", "Suppression"), ("enhancement", "Enhancement")]:
        line = next(raw for raw in main_table.splitlines() if label in raw and "&" in raw)
        actual = [
            float(value)
            for value in re.findall(r"-?(?:\d+(?:\.\d+)?|\.\d+)", line)
        ]
        fixed = policy_source.loc[
            policy_source["policy"].eq("Train-tuned fixed alpha")
            & policy_source["direction"].eq(direction)
        ].iloc[0]
        selected = policy_source.loc[
            policy_source["policy"].eq("Multi-fidelity M+A+R selector")
            & policy_source["direction"].eq(direction)
        ].iloc[0]
        versus_zero = paired_source.loc[
            paired_source["policy"].eq("Multi-fidelity M+A+R selector")
            & paired_source["baseline"].eq("No intervention")
            & paired_source["direction"].eq(direction)
            & paired_source["metric"].eq("utility")
        ].iloc[0]
        gain = paired_source.loc[
            paired_source["policy"].eq("Multi-fidelity M+A+R selector")
            & paired_source["baseline"].eq("Train-tuned fixed alpha")
            & paired_source["direction"].eq(direction)
            & paired_source["metric"].eq("utility")
        ].iloc[0]
        expected = [
            round(float(versus_zero["difference"]), 3),
            round(float(versus_zero["ci_low"]), 3),
            round(float(versus_zero["ci_high"]), 3),
            round(float(gain["difference"]), 3),
            round(float(gain["ci_low"]), 3),
            round(float(gain["ci_high"]), 3),
            round(float(fixed["neighbor_damage_rate"]), 3),
            round(float(selected["neighbor_damage_rate"]), 3),
            round(float(selected["abstention_rate"]), 3),
        ]
        assert actual == expected, f"Main selector mismatch for {direction}: {actual} != {expected}"

    dataset_manifest = json.loads(
        (DATA_DIR / "strength_dataset_manifest.json").read_text(encoding="utf-8")
    )
    run_config = json.loads(
        (DATA_DIR / "strength_run_config.json").read_text(encoding="utf-8")
    )
    assert dataset_manifest["lambda_neighbor"] == 1.0
    assert dataset_manifest["lambda_capability"] == 1.0
    assert run_config["clean_bonus"] == 0.1
    assert run_config["alpha_penalty"] == 0.01
    assert run_config["abstain_score"] == 0.0


def audit_endpoint_validation() -> None:
    source = pd.read_csv(DATA_DIR / "endpoint_validation_summary.csv")
    table_text = (TABLE_DIR / "appendix_endpoint_validation.tex").read_text(encoding="utf-8")
    rendered: dict[tuple[str, str], tuple[int, float, int, int, float, float]] = {}
    row_pattern = re.compile(
        r"^(?P<label>[^&]+?)\s*&\s*(?P<paths>\d+)\s*&\s*"
        r"(?P<base>[0-9.]+)\\%\s*&\s*(?P<gain>\d+)\s*/\s*(?P<damage>\d+)\s*&\s*"
        r"(?P<changed>[0-9.]+)\\%\s*&\s*(?P<clause>[0-9.]+)\\%\s*\\\\"
    )
    current_model: str | None = None
    for raw_line in table_text.splitlines():
        model_match = re.search(r"\\multicolumn\{6\}\{c\}\{([^}]+)\}", raw_line)
        if model_match:
            current_model = model_match.group(1).strip()
            continue
        match = row_pattern.match(raw_line.strip())
        if match:
            assert current_model is not None, f"Endpoint row precedes model header: {raw_line}"
            rendered[(current_model, match.group("label").strip())] = (
                int(match.group("paths")),
                float(match.group("base")),
                int(match.group("gain")),
                int(match.group("damage")),
                float(match.group("changed")),
                float(match.group("clause")),
            )

    for _, row in source.iterrows():
        label = ENDPOINT_LABELS[str(row["method"])]
        expected_prefix = (
            int(row["n_paths"]),
            round(100 * float(row["baseline_target_correct_rate"]), 1),
            int(row["unique_target_gain_count"]),
            int(row["unique_target_damage_count"]),
            round(100 * float(row["target_text_changed_rate"]), 1),
        )
        actual = rendered.get(("Qwen3-1.7B", label))
        assert actual is not None and actual[:5] == expected_prefix, (
            f"Endpoint mismatch for Qwen3-1.7B {label}: {actual} != {expected_prefix}"
        )
    assert len(rendered) == 8, f"Expected eight appendix endpoint rows, found {len(rendered)}"
    assert all(key[0] == "Qwen3.5-2B" for key in rendered if key[0] != "Qwen3-1.7B")
    assert "unmatched raw-alpha stress protocol" in table_text


def audit_paired_bootstrap() -> None:
    source = pd.read_csv(DATA_DIR / "paired_bootstrap_vs_random.csv")
    table_text = (TABLE_DIR / "appendix_paired_bootstrap.tex").read_text(encoding="utf-8")
    rendered: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    current_layer: int | None = None
    for raw_line in table_text.splitlines():
        layer_header = re.search(r"\\multicolumn\{5\}\{c\}\{Block (\d+)\}", raw_line)
        if layer_header:
            current_layer = int(layer_header.group(1))
            continue
        if "&" not in raw_line or "[" not in raw_line:
            continue
        cells = [cell.strip() for cell in raw_line.split("&")]
        if cells[0] == "Direction":
            continue
        if current_layer is None:
            continue
        values: list[tuple[float, float, float]] = []
        for cell in cells[1:5]:
            numbers = [
                float(value)
                for value in re.findall(r"[-+]?(?:\d+\.\d+|\.\d+)", cell)
            ]
            assert len(numbers) == 3, f"Cannot parse paired bootstrap cell: {cell}"
            values.append((numbers[0], numbers[1], numbers[2]))
        rendered[(cells[0], current_layer)] = values

    for (method, layer), group in source.groupby(["method", "layer"], sort=False):
        expected = []
        for metric in PAIRED_METRICS:
            row = group.loc[group["metric"].eq(metric)].iloc[0]
            expected.append(
                (
                    round(100 * float(row["difference"]), 1),
                    round(100 * float(row["ci_lower"]), 1),
                    round(100 * float(row["ci_upper"]), 1),
                )
            )
        actual = rendered.get(
            (PAIRED_LABELS[str(method)], forward_layer(int(layer)))
        )
        assert actual == expected, (
            f"Paired bootstrap mismatch for {method}, layer {layer}: {actual} != {expected}"
        )


def audit_selector_bootstrap() -> None:
    source = pd.read_csv(SELECTOR_PAIRED)
    table_text = (TABLE_DIR / "appendix_selector_bootstrap.tex").read_text(encoding="utf-8")
    rendered: dict[str, list[tuple[float, float, float]]] = {
        "suppression": [],
        "enhancement": [],
    }
    for raw_line in table_text.splitlines():
        if "&" not in raw_line or "[" not in raw_line:
            continue
        cells = [cell.strip() for cell in raw_line.split("&")]
        if cells[0] == "Metric" or len(cells) != 3:
            continue
        for direction, cell in zip(["suppression", "enhancement"], cells[1:]):
            numbers = [
                float(value)
                for value in re.findall(r"[-+]?(?:\d+\.\d+|\.\d+)", cell)
            ]
            assert len(numbers) == 3, f"Cannot parse selector bootstrap cell: {cell}"
            rendered[direction].append((numbers[0], numbers[1], numbers[2]))

    comparisons = [
        ("No intervention", "utility", 3, 1.0),
        ("Train-tuned fixed alpha", "utility", 3, 1.0),
        ("Train-tuned fixed alpha", "target_success", 1, 100.0),
        ("Train-tuned fixed alpha", "is_clean", 1, 100.0),
        ("Train-tuned fixed alpha", "neighbor_damage", 1, 100.0),
        ("Train-tuned fixed alpha", "capability_damage", 1, 100.0),
    ]
    source = source.loc[source["policy"].eq("Multi-fidelity M+A+R selector")]
    for direction, group in source.groupby("direction", sort=False):
        expected = []
        for baseline, metric, digits, scale in comparisons:
            row = group.loc[
                group["baseline"].eq(baseline) & group["metric"].eq(metric)
            ].iloc[0]
            expected.append(
                (
                    round(scale * float(row["difference"]), digits),
                    round(scale * float(row["ci_low"]), digits),
                    round(scale * float(row["ci_high"]), digits),
                )
            )
        actual = rendered.get(str(direction))
        assert actual == expected, (
            f"Selector bootstrap mismatch for {direction}: {actual} != {expected}"
        )


def audit_cross_model_replication() -> None:
    source_path = DATA_DIR / "cross_model_replication_summary.csv"
    if not source_path.exists():
        return
    source = pd.read_csv(source_path)
    assert len(source) == 24, f"Expected 24 RMS paths, found {len(source)}"
    assert set(source["model_key"]) == {
        "qwen3_1_7b",
        "qwen3_5_2b_base",
        "ministral_3_3b_base_2512",
    }
    assert set(source["method"]) == {
        "random",
        "mean_difference",
        "logistic",
        "rfm_agop_top1",
    }
    assert set(source["protocol"]) == {
        "residual_rms_reference_matching_dimension_corrected_v2"
    }
    assert source.groupby("model_key")["layer"].nunique().eq(2).all()
    assert source.groupby(["model_key", "layer"])["method"].nunique().eq(4).all()

    table_text = (TABLE_DIR / "cross_model_replication.tex").read_text(encoding="utf-8")
    appendix_text = (TABLE_DIR / "appendix_cross_model_outcomes.tex").read_text(
        encoding="utf-8"
    )
    assert "Macro AUROC across the eight outcomes" in table_text
    assert r"\label{tab:cross-model-prediction}" in table_text
    assert "$\\Delta L$" in table_text and "$\\Delta R$" in table_text
    for required in ["Qwen3-1.7B primary", "Qwen3-1.7B subset", "Qwen3.5-2B", "Ministral-3B"]:
        assert required in table_text, f"Missing predictor row: {required}"
    for required in ["Qwen3-1.7B-Base", "Qwen3.5-2B-Base", "Ministral-3-3B-Base"]:
        assert required in appendix_text, f"Missing appendix outcome block: {required}"
    interpretation_path = SECTION_DIR / "cross_model_results_generated.tex"
    fragment_paths = {
        "results": interpretation_path,
        "abstract": SECTION_DIR / "cross_model_abstract_generated.tex",
        "discussion": SECTION_DIR / "cross_model_discussion_generated.tex",
        "limitations": SECTION_DIR / "cross_model_limitations_generated.tex",
        "conclusion": SECTION_DIR / "cross_model_conclusion_generated.tex",
    }
    for name, path in fragment_paths.items():
        assert path.exists(), f"Missing generated cross-model {name} fragment"
    if not bool(source["complete"].all()):
        for name, path in fragment_paths.items():
            assert not path.read_text(encoding="utf-8").strip(), (
                f"Incomplete RMS results leaked into generated {name} prose"
            )
    else:
        for name in ["abstract", "discussion", "limitations", "conclusion"]:
            assert fragment_paths[name].read_text(encoding="utf-8").strip(), (
                f"Completed RMS results missing from generated {name} prose"
            )
    report = (DATA_DIR / "CROSS_MODEL_REPLICATION_REPORT.md").read_text(encoding="utf-8")
    assert "Partial paths are execution status only" in report


def audit_latex_graph() -> None:
    """Validate ICLR structure and dependencies without locking prose wording."""
    roots = [PAPER_ROOT / "main.tex", PAPER_ROOT / "supplement.tex"]
    closures = {root: tex_dependency_closure(root) for root in roots}
    for root, paths in closures.items():
        component = "\n".join(p.read_text(encoding="utf-8") for p in paths)
        assert "TBD" not in component
        active = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in component.splitlines())
        assert not re.search(r"\\iclrfinalcopy\b", active), "Review PDF must remain anonymous"
        assert r"\affiliations" not in active
        assert "aaai2027" not in active
        assert r"\usepackage{iclr2027_conference,times}" in active
        assert r"\bibliographystyle{iclr2027_conference}" in active
        labels = re.findall(r"\\label\{([^}]+)\}", active)
        assert len(labels) == len(set(labels)), f"Duplicate labels in {root.name}"
        refs = {key for key in re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", active)
                if not re.fullmatch(r"#[1-9]", key)}
        assert not refs - set(labels), f"Unresolved labels: {refs - set(labels)}"
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert text.count("{") == text.count("}"), f"Unbalanced braces: {path}"
            assert sorted(re.findall(r"\\begin\{([^}]+)\}", text)) == sorted(
                re.findall(r"\\end\{([^}]+)\}", text)), f"Unbalanced environments: {path}"
            if r"\begin{tabular" in text and path.parent == TABLE_DIR:
                assert text.index(r"\caption") < text.index(r"\begin{tabular"), (
                    f"Table caption must precede data: {path.name}"
                )
    main = roots[0].read_text()
    assert main.index(r"\input{sections/conclusion}") < main.index(r"\bibliography{references}")
    for section in ["ai_usage_statement", "reproducibility_statement"]:
        assert rf"\input{{sections/{section}}}" in main
    experiments = (SECTION_DIR / "experiments.tex").read_text()
    assert r"\setcounter{table}" not in experiments
    layer = (TABLE_DIR / "layer_selection.tex").read_text()
    assert r"\begin{figure}" in layer and r"\label{fig:layer-selection}" in layer
    appendix = (SECTION_DIR / "appendix.tex").read_text()
    for table in ["appendix_cpu_prediction_controls", "appendix_strength_selector",
                  "appendix_selector_bootstrap"]:
        assert rf"\input{{tables/{table}}}" in appendix
    assert "0.1774" not in appendix and "0.1821" not in appendix
    assert "36,000 logged rows" in appendix
    assert "logged-cohort estimate" in " ".join(appendix.split())
    for filename in ["iclr2027_conference.sty", "iclr2027_conference.bst"]:
        assert (PAPER_ROOT / filename).read_bytes() == (
            PAPER_ROOT / "iclr2027" / filename).read_bytes(), f"Modified official style: {filename}"


def audit_source_dependencies() -> None:
    manuscript_files = sorted(
        {
            path
            for root in [PAPER_ROOT / "main.tex", PAPER_ROOT / "supplement.tex"]
            for path in tex_dependency_closure(root)
        }
    )
    manuscript = "\n".join(path.read_text(encoding="utf-8") for path in manuscript_files)
    for target in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript):
        path = PAPER_ROOT / target
        assert path.exists(), f"Missing figure dependency: {path}"
    for group in re.findall(r"\\bibliography\{([^}]+)\}", manuscript):
        for target in group.split(","):
            path = (PAPER_ROOT / target.strip()).with_suffix(".bib")
            assert path.exists(), f"Missing bibliography dependency: {path}"
    for name in ["iclr2027_conference.sty", "iclr2027_conference.bst",
                 "math_commands.tex", "natbib.sty", "fancyhdr.sty"]:
        assert (PAPER_ROOT / name).exists(), f"Missing ICLR dependency: {name}"


def audit_main_figure_annotation_policy() -> None:
    manifest_path = DATA_DIR / "frozen_confirmatory_figure_manifest.json"
    assert manifest_path.exists(), f"Missing main-figure manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    direct_labels = manifest["direct_labels"]
    expected = {
        ("random", -21),
        ("mean_difference", -21),
        ("linear", -21),
        ("logistic", -21),
        ("rfm_agop_top1", -21),
        ("random", -17),
        ("mean_difference", -17),
    }
    actual = {(str(row["method"]), int(row["layer"])) for row in direct_labels}
    assert int(manifest["direct_label_count"]) == 7
    assert len(direct_labels) == 7
    assert actual == expected, f"Main-figure direct labels changed: {sorted(actual)}"


def audit_reproducibility_assets() -> None:
    environment = DATA_DIR / "computational_environment.json"
    environment_tex = SECTION_DIR / "computational_environment_generated.tex"
    for path in [environment, environment_tex]:
        assert path.exists(), f"Missing reproducibility asset: {path}"
    payload = json.loads(environment.read_text(encoding="utf-8"))
    rendered = environment_tex.read_text(encoding="utf-8")
    for key in [
        "operating_system",
        "cpu",
        "gpu",
        "nvidia_driver",
        "python",
        "pytorch",
        "transformers",
        "scikit_learn",
    ]:
        assert str(payload[key]) in rendered, f"Environment field missing from prose: {key}"


def audit_claim_evidence_ledger() -> None:
    ledger_path = DATA_DIR / "CLAIM_EVIDENCE_LEDGER.md"
    assert ledger_path.exists(), f"Missing claim-evidence ledger: {ledger_path}"
    ledger = ledger_path.read_text(encoding="utf-8")
    assert "Width-corrected residual-norm confirmation across three models." in ledger
    assert "24-path scientific gate" in ledger
    assert "Free-generation endpoint behavior is boundary evidence only." in ledger
    assert "Supplement only" in ledger
    assert "Partial `raw_scores.jsonl` files" in ledger


def audit_citations() -> None:
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(SECTION_DIR.glob("*.tex"))
    )
    citation_groups = re.findall(r"\\cite[pt]?\{([^}]+)\}", manuscript)
    cited = {key.strip() for group in citation_groups for key in group.split(",")}
    bib = (PAPER_ROOT / "references.bib").read_text(encoding="utf-8")
    key_list = re.findall(r"@[A-Za-z]+\{([^,]+),", bib)
    available = set(key_list)
    assert len(key_list) == len(available), "Duplicate BibTeX keys detected"
    missing = cited - available
    assert not missing, f"Missing BibTeX keys: {sorted(missing)}"
    uncited = available - cited
    assert not uncited, f"Uncited BibTeX entries: {sorted(uncited)}"

    starts = list(re.finditer(r"@[A-Za-z]+\{([^,]+),", bib))
    verification = json.loads((DATA_DIR / "citation_verification.json").read_text())
    verified = verification["entries"]
    downloads = json.loads((DATA_DIR / "citation_bibtex/download_manifest.json").read_text())
    downloaded = {row["key"]: row for row in downloads["records"]}
    assert len(downloaded) == len(downloads["records"]), "Duplicate citation download records"
    assert set(downloaded) == available, "Every citation must have a source-download record"
    assert set(verified) == available, (
        "Bibliography changed: verify added/removed entries against primary sources "
        "and update citation_verification.json"
    )
    source_fields = {"booktitle", "journal", "publisher", "howpublished", "school"}
    for index, match in enumerate(starts):
        key = match.group(1)
        record = downloaded[key]
        assert record["status"] == "downloaded", f"Missing original BibTeX download: {key}"
        source_path = DATA_DIR / verified[key]["source_bibtex_file"]
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert source_hash == record["sha256"] == verified[key]["source_bibtex_sha256"], (
            f"Archived source BibTeX changed: {key}"
        )
        assert record["source_kind"] == verified[key]["source_kind"], (
            f"Source provenance mismatch: {key}"
        )
        end = starts[index + 1].start() if index + 1 < len(starts) else len(bib)
        block = bib[match.start():end]
        fingerprint = hashlib.sha256(" ".join(block.split()).encode()).hexdigest()
        assert fingerprint == verified[match.group(1)]["bibliography_entry_sha256"], (
            f"Bibliography entry {match.group(1)} differs from its source-verified snapshot; "
            "recheck the source before updating the verification record"
        )
        body = bib[match.end() : end]
        fields = set(re.findall(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=", body, re.MULTILINE))
        required = {"title", "author", "year"}
        assert required <= fields, (
            f"BibTeX entry {match.group(1)} is missing fields: "
            f"{sorted(required - fields)}"
        )
        assert source_fields & fields, (
            f"BibTeX entry {match.group(1)} lacks venue/source metadata"
        )
    required_dataset_keys = {
        "wang2024mmlupro",
        "gema2024mmluredux",
        "clark2018arc",
        "mihaylov-etal-2018-suit",
        "welbl-etal-2017-crowdsourcing",
        "white2025livebench",
        "zellers-etal-2019-hellaswag",
        "khot2020qasc",
    }
    assert required_dataset_keys <= cited, (
        "Frozen benchmark sources are missing manuscript citations: "
        f"{sorted(required_dataset_keys - cited)}"
    )
    provenance_keys = set(
        pd.read_csv(DATA_DIR / "dataset_composition.csv")["citation_key"].astype(str)
    )
    assert provenance_keys == required_dataset_keys
    assert provenance_keys <= cited


def main() -> None:
    audit_dataset_composition()
    audit_scale_manifest()
    audit_outcomes()
    audit_prediction()
    audit_geometry_ablation()
    audit_strength_selector()
    audit_endpoint_validation()
    audit_paired_bootstrap()
    audit_selector_bootstrap()
    audit_cross_model_replication()
    audit_latex_graph()
    audit_source_dependencies()
    audit_main_figure_annotation_policy()
    audit_reproducibility_assets()
    audit_claim_evidence_ledger()
    audit_citations()
    print("Frozen paper asset audit passed")


if __name__ == "__main__":
    main()
