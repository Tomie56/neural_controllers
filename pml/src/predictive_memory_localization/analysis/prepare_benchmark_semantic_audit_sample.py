from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(
    "/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "benchmark_semantic_audit_20260722"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible dataset-stratified semantic-audit sample."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--per-dataset", type=int, default=12)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_eval(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "eval_id": str(item["eval_id"]),
        "prompt": str(item["prompt"]),
        "correct": str(item["correct"]),
        "contrast": str(item["contrast"]),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "semantic_audit_sample.jsonl"
    index_path = output_dir / "semantic_audit_sample_index.csv"
    worksheet_path = output_dir / "SEMANTIC_AUDIT_WORKSHEET.md"
    manifest_path = output_dir / "semantic_audit_sample_manifest.json"
    expected = [sample_path, index_path, worksheet_path, manifest_path]
    if all(path.exists() for path in expected) and not args.overwrite:
        print(manifest_path.read_text(encoding="utf-8"))
        return

    records = load_jsonl(data_dir / "records.jsonl")
    assignments = {
        str(row["record_id"]): row for row in load_jsonl(data_dir / "eval_assignments.jsonl")
    }
    eval_bank = {str(row["eval_id"]): row for row in load_jsonl(data_dir / "eval_bank.jsonl")}
    frame = pd.DataFrame(
        {
            "record_id": [str(row["record_id"]) for row in records],
            "dataset": [str(row["source"]["dataset"]) for row in records],
            "record_index": list(range(len(records))),
        }
    )
    rng = np.random.default_rng(args.seed)
    selected_indices: List[int] = []
    dataset_counts: Dict[str, int] = {}
    for dataset, group in frame.groupby("dataset", sort=True):
        if len(group) < args.per_dataset:
            raise ValueError(
                f"Dataset {dataset} has only {len(group)} records, below requested {args.per_dataset}"
            )
        choices = rng.choice(group["record_index"].to_numpy(), size=args.per_dataset, replace=False)
        selected_indices.extend(int(value) for value in choices)
        dataset_counts[dataset] = args.per_dataset

    samples: List[Dict[str, Any]] = []
    for audit_index, record_index in enumerate(selected_indices, start=1):
        record = records[record_index]
        record_id = str(record["record_id"])
        assignment = assignments[record_id]
        sample = {
            "audit_index": audit_index,
            "dataset": str(record["source"]["dataset"]),
            "record_id": record_id,
            "domain": str(record["source"]["domain"]),
            "source_question": str(record["source"]["question"]),
            "source_correct_answer": str(record["source"]["correct_answer"]),
            "concept": str(record["concept"]),
            "subject": str(record["subject"]),
            "seed_question": str(record["seed_question"]),
            "seed_answer": str(record["seed_answer"]),
            "fact_or_concept": str(record["fact_or_concept"]),
            "train_positive": [str(value) for value in record["train_positive"]],
            "train_negative": [str(value) for value in record["train_negative"]],
            "target_eval": [
                compact_eval(eval_bank[eval_id]) for eval_id in assignment["target_eval_ids"]
            ],
            "neighbor_eval": [
                compact_eval(eval_bank[eval_id]) for eval_id in assignment["neighbor_eval_ids"]
            ],
            "capability_eval": [
                compact_eval(eval_bank[eval_id]) for eval_id in assignment["capability_eval_ids"]
            ],
        }
        samples.append(sample)

    with sample_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    pd.DataFrame(
        [
            {
                "audit_index": row["audit_index"],
                "dataset": row["dataset"],
                "record_id": row["record_id"],
                "domain": row["domain"],
            }
            for row in samples
        ]
    ).to_csv(index_path, index=False)

    lines = [
        "# Benchmark Semantic Audit Worksheet",
        "",
        f"- sample size: `{len(samples)}`",
        f"- records per dataset: `{args.per_dataset}`",
        f"- sampling seed: `{args.seed}`",
        "- reviewer fields: pass / minor / fail",
        "",
    ]
    for sample in sorted(samples, key=lambda row: (row["dataset"], row["audit_index"])):
        lines.extend(
            [
                f"## {sample['audit_index']:03d} — {sample['dataset']} — `{sample['record_id']}`",
                "",
                f"- Source: {sample['source_question']} → **{sample['source_correct_answer']}**",
                f"- Fact: {sample['fact_or_concept']}",
                f"- Positive example: {sample['train_positive'][0]}",
                f"- Negative example: {sample['train_negative'][0]}",
                "- Target: "
                + " ; ".join(
                    f"{row['prompt']} → {row['correct']} / {row['contrast']}"
                    for row in sample["target_eval"]
                ),
                "- Neighbor: "
                + " ; ".join(
                    f"{row['prompt']} → {row['correct']} / {row['contrast']}"
                    for row in sample["neighbor_eval"]
                ),
                "- Capability: "
                + " ; ".join(
                    f"{row['prompt']} → {row['correct']} / {row['contrast']}"
                    for row in sample["capability_eval"]
                ),
                "- Desired/contrast validity: TBD",
                "- Target consistency: TBD",
                "- Neighbor locality: TBD",
                "- Capability independence: TBD",
                "- Construction polarity/naturalness: TBD",
                "- Overall: TBD",
                "- Notes:",
                "",
            ]
        )
    worksheet_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "sample_path": str(sample_path.resolve()),
        "index_path": str(index_path.resolve()),
        "worksheet_path": str(worksheet_path.resolve()),
        "sampling_protocol": "Uniform sample without replacement within each source dataset.",
        "per_dataset": args.per_dataset,
        "seed": args.seed,
        "n_records": len(samples),
        "dataset_counts": dataset_counts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
