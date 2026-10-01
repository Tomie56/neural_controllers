from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from predictive_memory_localization.common import load_jsonl


DEFAULT_DATA_DIR = "/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen"
DEFAULT_OUTPUT_DIR = "/data/neural_controllers/pml/data/pml_replication_500_stratified_seed113"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select deterministic stratified PML record IDs for compact model replication."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-records", type=int, default=500)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def record_stratum(record: Dict[str, Any]) -> tuple[str, str, str]:
    metadata = record.get("metadata") or {}
    source = record.get("source") or {}
    dataset = str(metadata.get("seed_dataset") or source.get("dataset") or "unknown_dataset")
    domain = str(metadata.get("seed_domain") or source.get("domain") or "unknown_domain")
    freshness = str(
        metadata.get("seed_freshness_group")
        or source.get("freshness_group")
        or "unknown_freshness"
    )
    return dataset, domain, freshness


def allocate_counts(group_sizes: Dict[tuple[str, str, str], int], total: int) -> Dict[tuple[str, str, str], int]:
    population = sum(group_sizes.values())
    if total <= 0 or total > population:
        raise ValueError(f"n_records must be in [1, {population}], got {total}")
    exact = {key: size * total / population for key, size in group_sizes.items()}
    allocated = {key: min(size, int(math.floor(exact[key]))) for key, size in group_sizes.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        group_sizes,
        key=lambda key: (exact[key] - allocated[key], group_sizes[key], key),
        reverse=True,
    )
    for key in order:
        if remaining == 0:
            break
        if allocated[key] < group_sizes[key]:
            allocated[key] += 1
            remaining -= 1
    if remaining != 0:
        raise RuntimeError(f"Failed to allocate {remaining} records")
    return allocated


def counts_by(records: Sequence[Dict[str, Any]], key_index: int) -> Dict[str, int]:
    return dict(sorted(Counter(record_stratum(record)[key_index] for record in records).items()))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ids_path = output_dir / "record_ids.txt"
    manifest_path = output_dir / "selection_manifest.json"
    expected = {
        "data_dir": str(data_dir),
        "n_records": args.n_records,
        "seed": args.seed,
    }
    if ids_path.exists() and manifest_path.exists() and not args.overwrite:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"previous": previous.get(key), "current": value}
            for key, value in expected.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Replication selection mismatch. Use a new output directory or --overwrite. "
                f"Mismatches: {mismatches}"
            )
        print(json.dumps(previous, ensure_ascii=False, indent=2))
        return

    records = list(load_jsonl(data_dir / "records.jsonl"))
    groups: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record_stratum(record)].append(record)
    allocation = allocate_counts({key: len(value) for key, value in groups.items()}, args.n_records)
    rng = np.random.default_rng(args.seed)
    selected: List[Dict[str, Any]] = []
    stratum_rows: List[Dict[str, Any]] = []
    for key in sorted(groups):
        candidates = groups[key]
        count = allocation[key]
        indices = rng.choice(len(candidates), size=count, replace=False)
        chosen = [candidates[int(index)] for index in sorted(indices)]
        selected.extend(chosen)
        stratum_rows.append(
            {
                "dataset": key[0],
                "domain": key[1],
                "freshness_group": key[2],
                "population": len(candidates),
                "selected": count,
            }
        )
    selected = sorted(selected, key=lambda record: str(record["record_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(
        "".join(f"{record['record_id']}\n" for record in selected), encoding="utf-8"
    )
    manifest = {
        **expected,
        "record_ids_path": str(ids_path),
        "population_records": len(records),
        "n_strata": len(groups),
        "dataset_counts": counts_by(selected, 0),
        "domain_counts": counts_by(selected, 1),
        "freshness_counts": counts_by(selected, 2),
        "strata": stratum_rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
