from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset import (
    atomic_write_csv,
    atomic_write_json,
)


DEFAULT_ROOT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/replication_500_residual_rms_v2"
)
DEFAULT_MODELS = [
    "qwen3_1_7b",
    "qwen3_5_2b_base",
    "ministral_3_3b_base_2512",
]
MODEL_LABELS = {
    "qwen3_1_7b": "Qwen3-1.7B-Base",
    "qwen3_5_2b_base": "Qwen3.5-2B-Base",
    "ministral_3_3b_base_2512": "Ministral-3-3B-Base",
}
DEFAULT_TARGETS = [
    "later_suppression_path",
    "later_enhancement_path",
    "later_target_any_path",
    "later_neighbor_damage_any_path",
    "later_capability_damage_any_path",
    "later_clean_suppression_path",
    "later_clean_enhancement_path",
    "later_clean_any_path",
]
DAMAGE_TARGETS = {
    "later_neighbor_damage_any_path",
    "later_capability_damage_any_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute record-paired learned-minus-random bootstrap confidence intervals "
            "for the three-model Residual-RMS outcome replication."
        )
    )
    parser.add_argument("--replication-root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_ROOT / "cross_model_outcome_paired_bootstrap")
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--learned-methods",
        nargs="+",
        default=["mean_difference", "logistic", "rfm_agop_top1"],
    )
    parser.add_argument("--random-method", default="random")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_path(model_root: Path) -> Path:
    manifest_path = (
        model_root
        / "outcome_analysis_later_strength"
        / "strict_later_outcome_summary_manifest.json"
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = Path(manifest["dataset_csv"])
        if path.exists():
            return path
    fallback = model_root / "prediction_dataset_later_strength" / "pml_path_prediction_dataset.csv"
    if not fallback.exists():
        raise FileNotFoundError(f"Missing strict-later prediction dataset below {model_root}")
    return fallback


def validate_model_frame(
    frame: pd.DataFrame,
    model_name: str,
    methods: List[str],
    random_method: str,
    targets: List[str],
) -> None:
    required = {"record_id", "method", "layer", *targets}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{model_name}: missing columns {missing}")
    expected_methods = {random_method, *methods}
    observed_methods = set(frame["method"].astype(str))
    missing_methods = sorted(expected_methods - observed_methods)
    if missing_methods:
        raise ValueError(f"{model_name}: missing methods {missing_methods}")
    selected = frame[frame["method"].isin(expected_methods)].copy()
    key_columns = ["record_id", "method", "layer"]
    if selected.duplicated(key_columns).any():
        examples = selected[selected.duplicated(key_columns, keep=False)][key_columns].head(10)
        raise ValueError(
            f"{model_name}: duplicate record/method/layer rows: {examples.to_dict('records')}"
        )
    record_counts = selected.groupby(["method", "layer"])["record_id"].nunique()
    if record_counts.nunique() != 1:
        raise ValueError(f"{model_name}: unbalanced method-layer record counts {record_counts.to_dict()}")


def build_comparisons(
    frame: pd.DataFrame,
    model_name: str,
    learned_methods: List[str],
    random_method: str,
    targets: List[str],
) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    rows: List[Dict[str, Any]] = []
    difference_columns: List[np.ndarray] = []
    comparison_keys: List[str] = []
    layers = sorted(int(value) for value in frame["layer"].unique())
    canonical_records = sorted(frame["record_id"].astype(str).unique())
    for layer in layers:
        reference = frame[
            frame["method"].eq(random_method) & frame["layer"].astype(int).eq(layer)
        ].copy()
        reference["record_id"] = reference["record_id"].astype(str)
        reference = reference.set_index("record_id").loc[canonical_records]
        for method in learned_methods:
            learned = frame[
                frame["method"].eq(method) & frame["layer"].astype(int).eq(layer)
            ].copy()
            learned["record_id"] = learned["record_id"].astype(str)
            learned = learned.set_index("record_id")
            if set(learned.index) != set(canonical_records):
                raise ValueError(
                    f"{model_name}: record mismatch for method={method}, layer={layer}"
                )
            learned = learned.loc[canonical_records]
            for target in targets:
                learned_values = learned[target].astype(float).to_numpy()
                random_values = reference[target].astype(float).to_numpy()
                differences = learned_values - random_values
                key = f"{method}|{layer}|{target}"
                comparison_keys.append(key)
                difference_columns.append(differences)
                rows.append(
                    {
                        "model": model_name,
                        "model_label": MODEL_LABELS.get(model_name, model_name),
                        "method": method,
                        "layer": layer,
                        "target": target,
                        "n_records": len(canonical_records),
                        "learned_rate": float(learned_values.mean()),
                        "random_rate": float(random_values.mean()),
                        "difference": float(differences.mean()),
                        "difference_points": float(100.0 * differences.mean()),
                        "learned_only_count": int(np.sum((learned_values == 1) & (random_values == 0))),
                        "random_only_count": int(np.sum((learned_values == 0) & (random_values == 1))),
                        "higher_is_better": target not in DAMAGE_TARGETS,
                    }
                )
    matrix = np.column_stack(difference_columns)
    return pd.DataFrame(rows), matrix, comparison_keys


def bootstrap_matrix(
    differences: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_records = differences.shape[0]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_records,
        np.full(n_records, 1.0 / n_records),
        size=reps,
    )
    bootstrap_means = weights @ differences / n_records
    low = np.quantile(bootstrap_means, 0.025, axis=0)
    high = np.quantile(bootstrap_means, 0.975, axis=0)
    return low, high


def attach_intervals(
    comparisons: pd.DataFrame,
    matrix: np.ndarray,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    low, high = bootstrap_matrix(matrix, reps, seed)
    output = comparisons.copy()
    output["ci_low"] = low
    output["ci_high"] = high
    output["ci_low_points"] = 100.0 * low
    output["ci_high_points"] = 100.0 * high
    output["favorable_difference"] = np.where(
        output["higher_is_better"], output["difference"], -output["difference"]
    )
    output["favorable_ci_low"] = np.where(
        output["higher_is_better"], output["ci_low"], -output["ci_high"]
    )
    output["favorable_ci_high"] = np.where(
        output["higher_is_better"], output["ci_high"], -output["ci_low"]
    )
    output["significant_favorable"] = output["favorable_ci_low"] > 0.0
    output["significant_unfavorable"] = output["favorable_ci_high"] < 0.0
    return output


def write_report(path: Path, results: pd.DataFrame, bootstrap_reps: int) -> None:
    focus_targets = [
        "later_target_any_path",
        "later_clean_any_path",
        "later_neighbor_damage_any_path",
        "later_capability_damage_any_path",
    ]
    focus = results[results["target"].isin(focus_targets)].copy()
    lines = [
        "# Three-Model Paired Outcome Bootstrap",
        "",
        f"- paired bootstrap replicates: `{bootstrap_reps}`",
        "- resampling unit: `record_id`",
        "- comparison: learned direction minus random direction at the same model and layer",
        "- target/clean: positive differences are favorable",
        "- neighbor/capability damage: negative differences are favorable",
        "",
        "| Model | Method | Layer | Outcome | Learned | Random | Delta (points) | 95% CI (points) |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.method} | {row.layer} | {row.target} | "
            f"{row.learned_rate:.3f} | {row.random_rate:.3f} | "
            f"{row.difference_points:+.1f} | "
            f"[{row.ci_low_points:+.1f}, {row.ci_high_points:+.1f}] |"
        )
    lines.extend(
        [
            "",
            "The complete CSV includes suppression, enhancement, clean-direction outcomes, paired discordant counts, and favorable-effect flags.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    replication_root = Path(args.replication_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "cross_model_outcome_paired_bootstrap.csv"
    report_path = output_dir / "CROSS_MODEL_OUTCOME_PAIRED_BOOTSTRAP_REPORT.md"
    manifest_path = output_dir / "cross_model_outcome_paired_bootstrap_manifest.json"
    if all(path.exists() for path in [result_path, report_path, manifest_path]) and not args.overwrite:
        print(manifest_path.read_text(encoding="utf-8"))
        return

    all_results: List[pd.DataFrame] = []
    model_datasets: Dict[str, str] = {}
    record_sets: Dict[str, set[str]] = {}
    for model_index, model_name in enumerate(args.models):
        model_root = replication_root / model_name
        source_path = dataset_path(model_root)
        frame = pd.read_csv(source_path)
        frame["record_id"] = frame["record_id"].astype(str)
        frame["layer"] = frame["layer"].astype(int)
        validate_model_frame(
            frame,
            model_name,
            args.learned_methods,
            args.random_method,
            args.targets,
        )
        selected_methods = {args.random_method, *args.learned_methods}
        frame = frame[frame["method"].isin(selected_methods)].copy()
        comparisons, matrix, _ = build_comparisons(
            frame,
            model_name,
            args.learned_methods,
            args.random_method,
            args.targets,
        )
        all_results.append(
            attach_intervals(
                comparisons,
                matrix,
                args.bootstrap_reps,
                args.seed + 10000 * model_index,
            )
        )
        model_datasets[model_name] = str(source_path.resolve())
        record_sets[model_name] = set(frame["record_id"])

    reference_model = args.models[0]
    cross_model_overlap = {
        model_name: len(record_sets[reference_model] & record_sets[model_name])
        for model_name in args.models
    }
    if any(count != len(record_sets[reference_model]) for count in cross_model_overlap.values()):
        raise ValueError(f"Cross-model record cohorts do not match: {cross_model_overlap}")

    results = pd.concat(all_results, ignore_index=True).sort_values(
        ["model", "layer", "method", "target"]
    )
    atomic_write_csv(result_path, results)
    manifest = {
        "replication_root": str(replication_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "models": args.models,
        "model_datasets": model_datasets,
        "learned_methods": args.learned_methods,
        "random_method": args.random_method,
        "targets": args.targets,
        "bootstrap_unit": "record_id",
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "n_result_rows": len(results),
        "records_per_model": {
            model_name: len(record_sets[model_name]) for model_name in args.models
        },
        "cross_model_overlap_with_reference": cross_model_overlap,
    }
    atomic_write_json(manifest_path, manifest)
    write_report(report_path, results, args.bootstrap_reps)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
