from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset import (
    atomic_write_csv,
    atomic_write_json,
)


DEFAULT_ROOT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b"
)
DEFAULT_PILOT_ROOT = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage2_bidirectional_pilot_500/qwen3_1_7b"
)
TARGETS = [
    "later_suppression_path",
    "later_enhancement_path",
    "later_target_any_path",
    "later_neighbor_damage_any_path",
    "later_capability_damage_any_path",
    "later_clean_suppression_path",
    "later_clean_enhancement_path",
    "later_clean_any_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict-later dataset excluding records used for layer selection."
    )
    parser.add_argument(
        "--source-dataset-dir", default=str(DEFAULT_ROOT / "prediction_dataset_strict_later")
    )
    parser.add_argument("--pilot-root", default=str(DEFAULT_PILOT_ROOT))
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_ROOT / "prediction_dataset_strict_later_excluding_pilot500")
    )
    parser.add_argument("--random-method", default="random")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pilot_record_ids(pilot_root: Path) -> tuple[set[str], List[str]]:
    paths = sorted(pilot_root.glob("*/layer_*/path_rows.csv"))
    if not paths:
        raise FileNotFoundError(f"No pilot path_rows.csv files found below {pilot_root}")
    record_ids: set[str] = set()
    for path in paths:
        frame = pd.read_csv(path, usecols=["record_id"])
        record_ids.update(frame["record_id"].astype(str))
    return record_ids, [str(path.resolve()) for path in paths]


def paired_bootstrap_ci(
    differences: np.ndarray, reps: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(differences)
    if n == 0:
        return float("nan"), float("nan")
    batch_size = min(250, reps)
    means: List[np.ndarray] = []
    completed = 0
    while completed < reps:
        current = min(batch_size, reps - completed)
        indices = rng.integers(0, n, size=(current, n))
        means.append(differences[indices].mean(axis=1))
        completed += current
    samples = np.concatenate(means)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_outcome_comparisons(
    frame: pd.DataFrame,
    random_method: str,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    methods = [
        method
        for method in sorted(frame["method"].astype(str).unique())
        if method not in {random_method, "matched_norm_random"}
    ]
    for layer in sorted(frame["layer"].astype(int).unique()):
        layer_frame = frame[frame["layer"].astype(int).eq(layer)]
        random_rows = layer_frame[layer_frame["method"].eq(random_method)].set_index("record_id")
        if random_rows.empty:
            raise ValueError(f"Missing random reference rows at layer {layer}")
        for method_index, method in enumerate(methods):
            learned_rows = layer_frame[layer_frame["method"].eq(method)].set_index("record_id")
            shared = learned_rows.index.intersection(random_rows.index)
            if len(shared) != frame["record_id"].nunique():
                raise ValueError(
                    f"Incomplete paired records for method={method}, layer={layer}: {len(shared)}"
                )
            for target_index, target in enumerate(TARGETS):
                differences = (
                    learned_rows.loc[shared, target].astype(float).to_numpy()
                    - random_rows.loc[shared, target].astype(float).to_numpy()
                )
                low, high = paired_bootstrap_ci(
                    differences,
                    reps,
                    seed + 1000 * method_index + 100 * target_index + abs(layer),
                )
                rows.append(
                    {
                        "method": method,
                        "layer": layer,
                        "target": target,
                        "n_records": len(shared),
                        "learned_rate": float(learned_rows.loc[shared, target].mean()),
                        "random_rate": float(random_rows.loc[shared, target].mean()),
                        "difference": float(differences.mean()),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    subset = frame[list(columns)]
    lines = [
        "| " + " | ".join(subset.columns) + " |",
        "| " + " | ".join(["---"] * len(subset.columns)) + " |",
    ]
    for row in subset.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "pml_path_prediction_dataset.csv"
    manifest_path = output_dir / "prediction_dataset_manifest.json"
    comparison_path = output_dir / "paired_outcome_vs_random.csv"
    report_path = output_dir / "PILOT500_EXCLUSION_REPORT.md"
    expected = [dataset_path, manifest_path, comparison_path, report_path]
    if all(path.exists() for path in expected) and not args.overwrite:
        print(manifest_path.read_text(encoding="utf-8"))
        return

    source_path = source_dir / "pml_path_prediction_dataset.csv"
    source_manifest_path = source_dir / "prediction_dataset_manifest.json"
    source = pd.read_csv(source_path)
    source["record_id"] = source["record_id"].astype(str)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    excluded_ids, pilot_files = pilot_record_ids(Path(args.pilot_root))
    source_ids = set(source["record_id"])
    overlap = excluded_ids & source_ids
    if len(excluded_ids) != 500 or len(overlap) != 500:
        raise ValueError(
            f"Expected 500 pilot records fully contained in source; "
            f"pilot={len(excluded_ids)}, overlap={len(overlap)}"
        )
    filtered = source[~source["record_id"].isin(excluded_ids)].copy().reset_index(drop=True)
    if filtered["record_id"].nunique() != source["record_id"].nunique() - 500:
        raise ValueError("Unexpected record count after excluding the layer-selection subset")

    comparisons = paired_outcome_comparisons(
        filtered, args.random_method, args.bootstrap_reps, args.seed
    )
    manifest = {
        **source_manifest,
        "source_dataset_path": str(source_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "exclusion_protocol": "Remove every record appearing in the 500-record layer-selection pilot.",
        "pilot_root": str(Path(args.pilot_root).resolve()),
        "pilot_path_files": pilot_files,
        "n_excluded_records": len(excluded_ids),
        "n_rows": int(len(filtered)),
        "n_records": int(filtered["record_id"].nunique()),
        "paired_bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    atomic_write_csv(dataset_path, filtered)
    atomic_write_json(manifest_path, manifest)
    atomic_write_csv(comparison_path, comparisons)

    focus = comparisons[
        comparisons["target"].isin(["later_target_any_path", "later_clean_any_path"])
    ].sort_values(["layer", "target", "difference"], ascending=[True, True, False])
    report = [
        "# Layer-Pilot 500-Record Exclusion Sensitivity",
        "",
        f"- Source records: `{source['record_id'].nunique()}`",
        f"- Excluded layer-selection records: `{len(excluded_ids)}`",
        f"- Remaining records: `{filtered['record_id'].nunique()}`",
        f"- Remaining path rows: `{len(filtered)}`",
        f"- Paired bootstrap replicates: `{args.bootstrap_reps}`",
        "",
        "## Learned-minus-random later-path differences",
        "",
        markdown_table(
            focus,
            [
                "method",
                "layer",
                "target",
                "n_records",
                "learned_rate",
                "random_rate",
                "difference",
                "ci_low",
                "ci_high",
            ],
        ),
        "",
        "The full target-by-method table is stored in `paired_outcome_vs_random.csv`.",
        "Prediction sensitivity is run separately on this filtered dataset.",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
