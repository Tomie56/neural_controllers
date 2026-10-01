from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select unique record-method-layer paths for endpoint validation."
    )
    parser.add_argument("--path-outcome-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-paths", type=int, default=100)
    parser.add_argument("--per-bucket", type=int, default=25)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=113)
    return parser.parse_args()


def as_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column]
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def layer_name(layer: int) -> str:
    return f"neg{abs(layer)}" if layer < 0 else str(layer)


def sample_paths(
    frame: pd.DataFrame,
    mask: pd.Series,
    bucket: str,
    count: int,
    seed: int,
    used_records: set[str],
) -> pd.DataFrame:
    candidates = frame[mask & ~frame["record_id"].isin(used_records)].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    grouped = {
        key: group.to_dict("records")
        for key, group in candidates.groupby(["method", "layer"], sort=True)
    }
    selected: List[Dict[str, object]] = []
    selected_records: set[str] = set()
    group_keys = list(grouped)
    while len(selected) < count:
        added = False
        for key in group_keys:
            rows = grouped[key]
            while rows:
                row = rows.pop()
                record_id = str(row["record_id"])
                if record_id in used_records or record_id in selected_records:
                    continue
                row["selection_bucket"] = bucket
                selected.append(row)
                selected_records.add(record_id)
                added = True
                break
            if len(selected) >= count:
                break
        if not added:
            break
    return pd.DataFrame(selected)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.path_outcome_csv)
    required = {
        "record_id",
        "method",
        "layer",
        "target_any_path",
        "clean_any_path",
        "damage_any_path",
        "no_effect_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing endpoint path-selection columns: {missing}")

    frame["record_id"] = frame["record_id"].astype(str)
    frame["layer"] = frame["layer"].astype(int)
    frame = frame[
        frame["method"].isin(args.methods) & frame["layer"].isin(args.layers)
    ].copy()
    if frame.empty:
        raise ValueError("No path rows match the requested methods and layers")
    for column in ["target_any_path", "clean_any_path", "damage_any_path", "no_effect_path"]:
        frame[column] = as_bool(frame, column)

    target = frame["target_any_path"]
    clean = frame["clean_any_path"]
    damage = frame["damage_any_path"]
    no_effect = frame["no_effect_path"]
    buckets: Dict[str, pd.Series] = {
        "clean_target_path": clean,
        "target_with_damage": target & damage,
        "damage_without_target": damage & ~target,
        "no_effect_path": no_effect,
    }

    selected_parts: List[pd.DataFrame] = []
    used_records: set[str] = set()
    available_counts: Dict[str, int] = {}
    selected_counts: Dict[str, int] = {}
    for bucket_index, (bucket, mask) in enumerate(buckets.items()):
        available_counts[bucket] = int(frame[mask]["record_id"].nunique())
        part = sample_paths(
            frame,
            mask,
            bucket,
            args.per_bucket,
            args.seed + bucket_index,
            used_records,
        )
        selected_parts.append(part)
        used_records.update(part["record_id"].tolist())
        selected_counts[bucket] = int(len(part))

    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) < args.max_paths:
        filler = frame[~frame["record_id"].isin(used_records)].drop_duplicates(
            "record_id", keep="first"
        )
        if not filler.empty:
            filler = filler.sample(
                n=min(args.max_paths - len(selected), len(filler)), random_state=args.seed
            )
            filler["selection_bucket"] = "coverage_filler"
            selected = pd.concat([selected, filler], ignore_index=True)
    selected = selected.head(args.max_paths).copy()
    if selected["record_id"].duplicated().any():
        raise ValueError("Path-conditioned endpoint selection must use unique records")

    selection_path = output_dir / "endpoint_validation_record_selection.csv"
    selected.to_csv(selection_path, index=False)
    record_ids_path = output_dir / "endpoint_validation_record_ids.txt"
    record_ids_path.write_text("\n".join(selected["record_id"]) + "\n", encoding="utf-8")

    path_root = output_dir / "path_record_ids"
    for (method, layer), group in selected.groupby(["method", "layer"]):
        path = path_root / str(method) / f"layer_{layer_name(int(layer))}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(group["record_id"]) + "\n", encoding="utf-8")

    manifest = {
        "path_outcome_csv": str(Path(args.path_outcome_csv).resolve()),
        "selection_granularity": "record_method_layer_path",
        "max_paths": args.max_paths,
        "n_selected_paths": int(len(selected)),
        "n_unique_records": int(selected["record_id"].nunique()),
        "methods": sorted(selected["method"].unique().tolist()),
        "layers": sorted(int(layer) for layer in selected["layer"].unique()),
        "available_paths_by_bucket": available_counts,
        "selected_paths_by_bucket": selected_counts,
        "final_bucket_counts": selected["selection_bucket"].value_counts().to_dict(),
        "selected_paths_by_method_layer": {
            f"{method}|{int(layer)}": int(len(group))
            for (method, layer), group in selected.groupby(["method", "layer"])
        },
        "selection_csv": str(selection_path),
        "record_ids_file": str(record_ids_path),
        "path_record_ids_root": str(path_root),
        "seed": args.seed,
    }
    (output_dir / "endpoint_validation_record_selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
