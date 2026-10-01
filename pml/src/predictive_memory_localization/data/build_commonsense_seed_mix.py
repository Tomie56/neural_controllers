from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import append_jsonl
from predictive_memory_localization.data.generate_suppression_data_openrouter import (
    load_seed_dataset,
    normalize_seed_item,
)


DEFAULT_SOURCES = [
    {
        "dataset_name": "allenai/openbookqa",
        "dataset_config": "additional",
        "split": "train",
        "dataset_path": "/data/neural_controllers/openbookqa",
        "target_records": 1200,
    },
    {
        "dataset_name": "tau/commonsense_qa",
        "dataset_config": "",
        "split": "train",
        "dataset_path": "",
        "target_records": 900,
    },
    {
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Challenge",
        "split": "train",
        "dataset_path": "",
        "target_records": 600,
    },
    {
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Easy",
        "split": "train",
        "dataset_path": "",
        "target_records": 300,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized multi-source commonsense QA seed JSONL."
    )
    parser.add_argument(
        "--output-jsonl",
        default="/data/neural_controllers/pml/data/commonsense_seed_mix_3000.jsonl",
    )
    parser.add_argument("--target-records", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-spec-json",
        default=None,
        help="Optional JSON file containing a list of source specs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_specs(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_SOURCES
    return json.loads(Path(path).read_text(encoding="utf-8"))


def namespace_source_id(dataset_name: str, dataset_config: str, source_id: str, fallback_idx: int) -> str:
    clean_name = dataset_name.replace("/", "__")
    clean_config = dataset_config.replace("/", "__") if dataset_config else "default"
    sid = source_id or str(fallback_idx)
    return f"{clean_name}::{clean_config}::{sid}"


def load_source_rows(spec: Dict[str, Any]):
    ns = argparse.Namespace(
        dataset_name=spec["dataset_name"],
        dataset_config=spec.get("dataset_config", ""),
        split=spec.get("split", "train"),
        dataset_path=spec.get("dataset_path") or None,
        local_config=spec.get("local_config"),
        seed_jsonl=None,
    )
    return load_seed_dataset(ns)


def iter_normalized_source(spec: Dict[str, Any], seed: int) -> Iterable[Dict[str, Any]]:
    dataset = load_source_rows(spec)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for fallback_idx, idx in enumerate(indices):
        row = dict(dataset[idx])
        seed_item = normalize_seed_item(row, spec["dataset_name"])
        if not seed_item["question"] or not seed_item["correct_answer"] or not seed_item["wrong_answers"]:
            continue
        seed_item["source_id"] = namespace_source_id(
            seed_item["dataset_name"],
            spec.get("dataset_config", ""),
            seed_item["source_id"],
            fallback_idx,
        )
        seed_item["seed_source"] = {
            "dataset_name": spec["dataset_name"],
            "dataset_config": spec.get("dataset_config", ""),
            "split": spec.get("split", "train"),
        }
        yield seed_item


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        out_path.unlink(missing_ok=True)
    elif out_path.exists():
        existing = sum(1 for line in out_path.open(encoding="utf-8") if line.strip())
        if existing >= args.target_records:
            manifest = {
                "output_jsonl": str(out_path),
                "target_records": args.target_records,
                "written": existing,
                "source_counts": {},
                "sources": source_specs(args.source_spec_json),
                "status": "existing_file_reused",
            }
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return
        raise SystemExit(
            f"Existing seed file has only {existing}/{args.target_records} rows. "
            "Rerun with --overwrite to rebuild it."
        )

    specs = source_specs(args.source_spec_json)
    written = 0
    seen = set()
    source_counts: Dict[str, int] = {}

    for source_idx, spec in enumerate(specs):
        if written >= args.target_records:
            break
        target = int(spec.get("target_records", args.target_records))
        count = 0
        try:
            iterator = iter_normalized_source(spec, args.seed + source_idx)
            for seed_item in iterator:
                if written >= args.target_records or count >= target:
                    break
                key = (seed_item["dataset_name"], seed_item["source_id"])
                if key in seen:
                    continue
                append_jsonl(out_path, seed_item)
                seen.add(key)
                count += 1
                written += 1
        except Exception as exc:
            print(f"WARNING: skipped source {spec['dataset_name']} due to {exc}", file=sys.stderr)
        count_key = f"{spec['dataset_name']}:{spec.get('dataset_config', '')}"
        source_counts[count_key] = count

    manifest = {
        "output_jsonl": str(out_path),
        "target_records": args.target_records,
        "written": written,
        "source_counts": source_counts,
        "sources": specs,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if written < args.target_records:
        raise SystemExit(f"Only wrote {written}/{args.target_records} seed rows")


if __name__ == "__main__":
    main()
