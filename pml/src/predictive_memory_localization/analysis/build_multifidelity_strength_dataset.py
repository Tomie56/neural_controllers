from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine dense selector training curves with non-overlapping sparse auxiliary trajectories."
    )
    parser.add_argument("--dense-train-dataset-dir", required=True)
    parser.add_argument("--dense-validation-dataset-dir", required=True)
    parser.add_argument("--sparse-aux-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparse-methods", nargs="*", default=["mean_difference", "logistic", "random"])
    parser.add_argument("--sparse-layers", nargs="*", type=int, default=[-21, -17])
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_alpha_dataset(directory: Path) -> pd.DataFrame:
    path = directory / "alpha_level_selector_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["record_id"] = frame["record_id"].astype(str)
    return frame


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_path = output_dir / "alpha_level_selector_dataset.csv"
    manifest_path = output_dir / "multifidelity_dataset_manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Outputs exist in {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    dense_train = read_alpha_dataset(Path(args.dense_train_dataset_dir))
    dense_validation = read_alpha_dataset(Path(args.dense_validation_dataset_dir))
    sparse_aux = read_alpha_dataset(Path(args.sparse_aux_dataset_dir))

    dense_train_ids = set(dense_train["record_id"])
    validation_ids = set(dense_validation["record_id"])
    if dense_train_ids & validation_ids:
        raise ValueError("Dense train and dense validation record IDs overlap.")

    sparse_aux = sparse_aux[
        sparse_aux["method"].isin(args.sparse_methods)
        & sparse_aux["layer"].astype(int).isin(args.sparse_layers)
    ].copy()
    sparse_candidate_ids = set(sparse_aux["record_id"])
    excluded_ids = dense_train_ids | validation_ids
    sparse_aux = sparse_aux[~sparse_aux["record_id"].isin(excluded_ids)].copy()
    sparse_used_ids = set(sparse_aux["record_id"])
    if sparse_used_ids & validation_ids:
        raise AssertionError("Validation leakage detected in sparse auxiliary records.")

    dense_train["trajectory_source"] = "dense_train_500"
    dense_train["source_weight"] = float(args.dense_weight)
    sparse_aux["trajectory_source"] = "sparse_aux_main_3000_exclusive"
    sparse_aux["source_weight"] = float(args.sparse_weight)
    combined = pd.concat([dense_train, sparse_aux], ignore_index=True, sort=False)
    combined.to_csv(output_path, index=False)

    manifest = {
        "dense_train_dataset_dir": args.dense_train_dataset_dir,
        "dense_validation_dataset_dir": args.dense_validation_dataset_dir,
        "sparse_aux_dataset_dir": args.sparse_aux_dataset_dir,
        "output_dir": str(output_dir),
        "dense_train_unique_records": len(dense_train_ids),
        "dense_validation_unique_records": len(validation_ids),
        "sparse_candidate_unique_records": len(sparse_candidate_ids),
        "sparse_aux_unique_records_used": len(sparse_used_ids),
        "combined_train_unique_records": int(combined["record_id"].nunique()),
        "study_universe_unique_records": len(dense_train_ids | validation_ids | sparse_used_ids),
        "dense_train_rows": len(dense_train),
        "sparse_aux_rows": len(sparse_aux),
        "combined_train_rows": len(combined),
        "dense_validation_rows": len(dense_validation),
        "sparse_methods": args.sparse_methods,
        "sparse_layers": args.sparse_layers,
        "dense_weight": args.dense_weight,
        "sparse_weight": args.sparse_weight,
        "validation_overlap_count": len(set(combined["record_id"]) & validation_ids),
        "reporting_convention": (
            "500 dense training records + non-overlapping sparse auxiliary records + "
            "100 dense held-out validation records; never describe sparse records as dense scans"
        ),
        "alpha_level_selector_dataset": str(output_path),
    }
    if manifest["validation_overlap_count"] != 0:
        raise AssertionError(f"Validation overlap: {manifest['validation_overlap_count']}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
