from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl, write_jsonl


DEFAULT_SRC_PREFIX = "/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000_v1"
DEFAULT_OUTPUT_DIR = "/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a frozen PML bidirectional dataset directory with uniformly rebalanced "
            "capability eval assignments."
        )
    )
    parser.add_argument("--src-prefix", default=DEFAULT_SRC_PREFIX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-capability-per-record", type=int, default=4)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_paths(prefix: str) -> Dict[str, Path]:
    p = Path(prefix)
    return {
        "records": Path(f"{p}.records.jsonl"),
        "eval_bank": Path(f"{p}.eval_bank.jsonl"),
        "assignments": Path(f"{p}.eval_assignments.jsonl"),
        "validation": Path(f"{p}.validation.json"),
        "sample_report": Path(f"{p}.sample_report.jsonl"),
    }


def output_paths(output_dir: str) -> Dict[str, Path]:
    d = Path(output_dir)
    return {
        "records": d / "records.jsonl",
        "eval_bank": d / "eval_bank.jsonl",
        "assignments": d / "eval_assignments.jsonl",
        "validation": d / "validation.json",
        "manifest": d / "manifest.json",
        "readme": d / "README.md",
        "sample_report": d / "sample_report.jsonl",
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return list(load_jsonl(path))


def rotate_rebalanced_capabilities(
    assignments: List[Dict[str, Any]],
    capability_ids: List[str],
    n_per_record: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if n_per_record <= 0:
        raise ValueError("--n-capability-per-record must be positive")
    if len(capability_ids) < n_per_record:
        raise ValueError(f"Need at least {n_per_record} capability ids, got {len(capability_ids)}")

    rng = random.Random(seed)
    pool = capability_ids[:]
    rng.shuffle(pool)

    out = []
    cursor = 0
    for row in assignments:
        new_row = dict(row)
        chosen = []
        while len(chosen) < n_per_record:
            candidate = pool[cursor % len(pool)]
            cursor += 1
            if candidate in chosen:
                continue
            chosen.append(candidate)
        new_row["capability_eval_ids"] = chosen
        policy = dict(new_row.get("assignment_policy", {}))
        policy.update(
            {
                "capability": "global_uniform_round_robin_rebalanced",
                "n_capability": n_per_record,
                "rebalance_seed": seed,
                "capability_pool_size": len(capability_ids),
            }
        )
        new_row["assignment_policy"] = policy
        out.append(new_row)
    return out


def validate(
    records: List[Dict[str, Any]],
    eval_bank: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    n_capability_per_record: int,
) -> Dict[str, Any]:
    record_ids = {row["record_id"] for row in records}
    eval_by_id = {row["eval_id"]: row for row in eval_bank}
    eval_ids = set(eval_by_id)
    capability_ids = {row["eval_id"] for row in eval_bank if row.get("eval_type") == "capability"}

    missing_refs = []
    assignment_len_counts = {key: collections.Counter() for key in ["target_eval_ids", "neighbor_eval_ids", "capability_eval_ids"]}
    cap_usage = collections.Counter()
    target_usage = collections.Counter()
    neighbor_usage = collections.Counter()
    for row in assignments:
        if row["record_id"] not in record_ids:
            missing_refs.append({"kind": "record_id", "value": row["record_id"]})
        for key in assignment_len_counts:
            ids = row.get(key, [])
            assignment_len_counts[key][len(ids)] += 1
            for eval_id in ids:
                if eval_id not in eval_ids:
                    missing_refs.append({"kind": key, "record_id": row["record_id"], "value": eval_id})
        cap_usage.update(row.get("capability_eval_ids", []))
        target_usage.update(row.get("target_eval_ids", []))
        neighbor_usage.update(row.get("neighbor_eval_ids", []))

    cap_counts = list(cap_usage.values())
    unused_capability = sorted(capability_ids - set(cap_usage))
    overused_examples = [
        {
            "eval_id": eval_id,
            "count": count,
            "prompt": eval_by_id[eval_id]["prompt"],
            "correct": eval_by_id[eval_id]["correct"],
            "contrast": eval_by_id[eval_id]["contrast"],
        }
        for eval_id, count in cap_usage.most_common(20)
    ]

    by_dataset = collections.Counter(row.get("source", {}).get("dataset") for row in records)
    by_domain = collections.Counter(row.get("source", {}).get("domain") for row in records)
    by_freshness = collections.Counter(row.get("metadata", {}).get("seed_freshness_group") for row in records)
    by_release_year = collections.Counter(str(row.get("metadata", {}).get("seed_release_year")) for row in records)
    by_eval_type = collections.Counter(row["eval_type"] for row in eval_bank)

    return {
        "schema": "pml_bidirectional_v1_frozen_rebalanced_capability",
        "n_records": len(records),
        "n_eval_bank": len(eval_bank),
        "n_assignments": len(assignments),
        "eval_type_counts": dict(by_eval_type),
        "dataset_counts": dict(by_dataset),
        "domain_counts": dict(by_domain),
        "freshness_group_counts": dict(by_freshness),
        "release_year_counts": dict(by_release_year),
        "n_duplicate_record_ids": len(records) - len(record_ids),
        "n_duplicate_eval_ids": len(eval_bank) - len(eval_ids),
        "n_missing_refs": len(missing_refs),
        "missing_refs_examples": missing_refs[:20],
        "assignment_len_counts": {
            key: {str(k): v for k, v in counter.items()} for key, counter in assignment_len_counts.items()
        },
        "target_eval_usage": {
            "n_used": len(target_usage),
            "min": min(target_usage.values()) if target_usage else None,
            "max": max(target_usage.values()) if target_usage else None,
        },
        "neighbor_eval_usage": {
            "n_used": len(neighbor_usage),
            "min": min(neighbor_usage.values()) if neighbor_usage else None,
            "max": max(neighbor_usage.values()) if neighbor_usage else None,
        },
        "capability_eval_usage": {
            "n_pool": len(capability_ids),
            "n_used": len(cap_usage),
            "n_unused": len(unused_capability),
            "total_assignments": sum(cap_counts),
            "mean": (sum(cap_counts) / len(cap_counts)) if cap_counts else None,
            "min": min(cap_counts) if cap_counts else None,
            "max": max(cap_counts) if cap_counts else None,
            "top20": overused_examples,
            "unused_examples": unused_capability[:20],
        },
        "is_freeze_ready": (
            len(records) == len(assignments)
            and len(records) == 3000
            and len(records) == len(record_ids)
            and len(eval_bank) == len(eval_ids)
            and not missing_refs
            and assignment_len_counts["target_eval_ids"] == collections.Counter({3: len(records)})
            and assignment_len_counts["neighbor_eval_ids"] == collections.Counter({3: len(records)})
            and assignment_len_counts["capability_eval_ids"] == collections.Counter({n_capability_per_record: len(records)})
            and len(unused_capability) == 0
            and (max(cap_counts) - min(cap_counts) <= 1 if cap_counts else False)
        ),
    }


def write_readme(path: Path, manifest: Dict[str, Any], validation: Dict[str, Any]) -> None:
    text = f"""# Frozen PML Fresh Multidomain 3000 v1

This directory is the frozen dataset for the next PML bidirectional steering experiments.

Do not edit files in this directory in-place. If the dataset changes, create a new frozen directory with a new version suffix.

## Files

- `records.jsonl`: 3000 intervention records. These contain training positive/negative statements and no eval fields.
- `eval_bank.jsonl`: global contrastive eval bank with `prompt`, `correct`, and `contrast`.
- `eval_assignments.jsonl`: record-to-eval mapping. Target and neighbor assignments are record-specific; capability assignments are globally rebalanced.
- `validation.json`: schema, reference, distribution, and capability usage checks.
- `manifest.json`: source file paths and generation settings.
- `sample_report.jsonl`: human QA sample by dataset.

## Capability Rebalancing

Capability probes are assigned by uniform round-robin over the global capability pool.

- records: {validation["n_records"]}
- capability pool size: {validation["capability_eval_usage"]["n_pool"]}
- capability assignments: {validation["capability_eval_usage"]["total_assignments"]}
- mean usage: {validation["capability_eval_usage"]["mean"]}
- min usage: {validation["capability_eval_usage"]["min"]}
- max usage: {validation["capability_eval_usage"]["max"]}

## Readiness

`is_freeze_ready`: `{validation["is_freeze_ready"]}`

## Source

Source prefix: `{manifest["src_prefix"]}`

Created by:

```bash
{manifest["command_hint"]}
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    src = source_paths(args.src_prefix)
    out = output_paths(args.output_dir)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {out_dir}. Use --overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_rows(src["records"])
    eval_bank = load_rows(src["eval_bank"])
    assignments = load_rows(src["assignments"])
    capability_ids = sorted(row["eval_id"] for row in eval_bank if row.get("eval_type") == "capability")
    rebalanced = rotate_rebalanced_capabilities(
        assignments,
        capability_ids,
        args.n_capability_per_record,
        args.seed,
    )

    write_jsonl(out["records"], records)
    write_jsonl(out["eval_bank"], eval_bank)
    write_jsonl(out["assignments"], rebalanced)
    if src["sample_report"].exists():
        shutil.copy2(src["sample_report"], out["sample_report"])

    validation = validate(records, eval_bank, rebalanced, args.n_capability_per_record)
    manifest = {
        "dataset_name": "pml_fresh_multidomain_3000_v1_frozen",
        "src_prefix": args.src_prefix,
        "output_dir": str(out_dir),
        "n_capability_per_record": args.n_capability_per_record,
        "rebalance_seed": args.seed,
        "source_paths": {key: str(value) for key, value in src.items()},
        "output_paths": {key: str(value) for key, value in out.items()},
        "command_hint": (
            "PYTHONPATH=/data/neural_controllers/pml/src "
            "/data/miniconda3/envs/rfm/bin/python -m "
            "predictive_memory_localization.data.rebalance_capability_assignments "
            f"--src-prefix {args.src_prefix} --output-dir {out_dir} "
            f"--n-capability-per-record {args.n_capability_per_record} --seed {args.seed} --overwrite"
        ),
    }
    out["validation"].write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    out["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out["readme"], manifest, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
