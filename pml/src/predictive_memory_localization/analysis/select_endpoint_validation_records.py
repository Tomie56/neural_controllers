from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_PATH_DATASET = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage_activation_pml/main_3000/qwen3_1_7b/prediction_dataset/"
    "pml_path_prediction_dataset.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select balanced endpoint-validation records from the current PML path dataset."
    )
    parser.add_argument("--path-outcome-csv", default=DEFAULT_PATH_DATASET)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--per-bucket", type=int, default=25)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument(
        "--prefer-methods",
        nargs="*",
        default=["mean_difference", "logistic", "rfm_agop_top1"],
    )
    parser.add_argument("--prefer-layers", nargs="*", type=int, default=[-21, -17])
    return parser.parse_args()


def bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    series = df[column]
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).fillna(False)


def sample_bucket(
    df: pd.DataFrame,
    mask: pd.Series,
    bucket: str,
    per_bucket: int,
    seed: int,
    excluded_record_ids: set[str],
) -> pd.DataFrame:
    bucket_df = df[mask & ~df["record_id"].astype(str).isin(excluded_record_ids)].copy()
    if bucket_df.empty:
        return bucket_df
    sort_columns = [
        column
        for column in ["clean_any_path", "target_any_path", "damage_any_path"]
        if column in bucket_df.columns
    ]
    if sort_columns:
        bucket_df = bucket_df.sort_values(sort_columns, ascending=False)
    bucket_df = bucket_df.drop_duplicates("record_id", keep="first")
    if len(bucket_df) > per_bucket:
        bucket_df = bucket_df.sample(n=per_bucket, random_state=seed)
    bucket_df["selection_bucket"] = bucket
    return bucket_df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.path_outcome_csv)
    required = {
        "record_id",
        "target_any_path",
        "clean_any_path",
        "damage_any_path",
        "no_effect_path",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing current path columns in {args.path_outcome_csv}: {missing}")

    preferred = df.copy()
    if "method" in preferred.columns and args.prefer_methods:
        filtered = preferred[preferred["method"].isin(args.prefer_methods)]
        if not filtered.empty:
            preferred = filtered
    if "layer" in preferred.columns and args.prefer_layers:
        filtered = preferred[preferred["layer"].astype(int).isin(args.prefer_layers)]
        if not filtered.empty:
            preferred = filtered

    target = bool_series(preferred, "target_any_path")
    clean = bool_series(preferred, "clean_any_path")
    damage = bool_series(preferred, "damage_any_path")
    no_effect = bool_series(preferred, "no_effect_path")
    masks: Dict[str, pd.Series] = {
        "clean_target_path": clean,
        "target_with_damage": target & damage,
        "damage_without_target": damage & ~target,
        "no_effect_path": no_effect,
    }

    selected_parts: List[pd.DataFrame] = []
    selected_ids: set[str] = set()
    available_counts: Dict[str, int] = {}
    selected_counts: Dict[str, int] = {}
    for bucket_index, (bucket, mask) in enumerate(masks.items()):
        available_counts[bucket] = int(preferred[mask]["record_id"].astype(str).nunique())
        part = sample_bucket(
            preferred,
            mask,
            bucket,
            args.per_bucket,
            args.seed + bucket_index,
            selected_ids,
        )
        if not part.empty:
            selected_parts.append(part)
            selected_ids.update(part["record_id"].astype(str))
        selected_counts[bucket] = int(part["record_id"].nunique()) if not part.empty else 0

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if selected.empty:
        raise ValueError("No endpoint validation records selected.")

    if len(selected) < args.max_records:
        filler = preferred[~preferred["record_id"].astype(str).isin(selected_ids)].copy()
        if not filler.empty:
            filler = filler.drop_duplicates("record_id", keep="first")
            filler = filler.sample(
                n=min(args.max_records - len(selected), len(filler)),
                random_state=args.seed,
            )
            filler["selection_bucket"] = "coverage_filler"
            selected = pd.concat([selected, filler], ignore_index=True)

    selected = selected.drop_duplicates("record_id", keep="first").head(args.max_records)
    selected["has_neighbor_damage_path"] = bool_series(selected, "neighbor_damage_any_path")
    selected["has_capability_damage_path"] = bool_series(selected, "capability_damage_any_path")
    record_ids = selected["record_id"].astype(str).tolist()

    record_ids_path = out_dir / "endpoint_validation_record_ids.txt"
    selection_path = out_dir / "endpoint_validation_record_selection.csv"
    manifest_path = out_dir / "endpoint_validation_record_selection_manifest.json"
    record_ids_path.write_text("\n".join(record_ids) + "\n", encoding="utf-8")
    selected.to_csv(selection_path, index=False)
    manifest = {
        "path_outcome_csv": str(Path(args.path_outcome_csv)),
        "n_selected_records": len(record_ids),
        "max_records": args.max_records,
        "per_bucket": args.per_bucket,
        "seed": args.seed,
        "prefer_methods": args.prefer_methods,
        "prefer_layers": args.prefer_layers,
        "available_unique_records_by_bucket": available_counts,
        "selected_unique_records_by_bucket": selected_counts,
        "final_bucket_counts": selected["selection_bucket"].value_counts().to_dict(),
        "neighbor_damage_records": int(selected["has_neighbor_damage_path"].sum()),
        "capability_damage_records": int(selected["has_capability_damage_path"].sum()),
        "record_ids_file": str(record_ids_path),
        "selection_csv": str(selection_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
