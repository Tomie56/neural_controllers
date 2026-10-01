from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import append_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach unrelated capability probes to suppression records. "
            "The probes are sampled from other records' target eval items, so no new API data is required."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_candidate_probes(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id", ""))
        source = record.get("source", {})
        dataset = source.get("dataset") if isinstance(source, dict) else None
        for idx, item in enumerate(record.get("eval", [])):
            if not item.get("prompt") or not item.get("target"):
                continue
            probes.append(
                {
                    "prompt": item["prompt"],
                    "target": item["target"],
                    "source_record_id": record_id,
                    "source_eval_index": idx,
                    "source_dataset": dataset,
                    "probe_type": "unrelated_eval",
                }
            )
    return probes


def choose_probes(
    record: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    *,
    num_probes: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    record_id = str(record.get("id", ""))
    source = record.get("source", {})
    dataset = source.get("dataset") if isinstance(source, dict) else None
    filtered = [
        probe
        for probe in candidates
        if probe.get("source_record_id") != record_id
        and (dataset is None or probe.get("source_dataset") != dataset or len(candidates) < num_probes * 4)
    ]
    if len(filtered) < num_probes:
        filtered = [probe for probe in candidates if probe.get("source_record_id") != record_id]
    if len(filtered) <= num_probes:
        return [dict(probe) for probe in filtered]
    return [dict(probe) for probe in rng.sample(filtered, num_probes)]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(load_jsonl(input_path))
    candidates = collect_candidate_probes(records)
    rng = random.Random(args.seed)

    for record in records:
        out = dict(record)
        out["capability"] = choose_probes(record, candidates, num_probes=args.num_probes, rng=rng)
        append_jsonl(output_path, out)

    report = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "n_records": len(records),
        "n_candidate_probes": len(candidates),
        "num_probes_per_record": args.num_probes,
        "seed": args.seed,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote records with capability probes to {output_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
