from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import continuation_logprob, load_jsonl, load_model_and_tokenizer


DEFAULT_DATA_DIR = "/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen"
DEFAULT_MODEL_PATH = "/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base"
DEFAULT_OUTPUT_DIR = (
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "stage0_margin_sanity/qwen3_1_7b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0: alpha=0 correct-vs-contrast margin sanity for frozen PML three-table data."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-local-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default="qwen3_1_7b")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument("--sample-per-type", type=int, default=200)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def quantiles(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    vals = sorted(values)
    return {
        "mean": mean(vals),
        "median": vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]),
        "min": vals[0],
        "max": vals[-1],
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_tables(data_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    records = list(load_jsonl(data_dir / "records.jsonl"))
    eval_bank = {row["eval_id"]: row for row in load_jsonl(data_dir / "eval_bank.jsonl")}
    assignments = list(load_jsonl(data_dir / "eval_assignments.jsonl"))
    return records, eval_bank, assignments


def sample_eval_ids(assignments: List[Dict[str, Any]], eval_bank: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    rng = random.Random(args.seed)
    by_type: Dict[str, List[str]] = collections.defaultdict(list)
    for eval_id, row in eval_bank.items():
        by_type[row["eval_type"]].append(eval_id)

    sampled: List[str] = []
    if args.sample_per_type > 0:
        for eval_type in sorted(by_type):
            ids = by_type[eval_type][:]
            rng.shuffle(ids)
            sampled.extend(ids[: min(args.sample_per_type, len(ids))])
    else:
        ids = list(eval_bank)
        rng.shuffle(ids)
        sampled = ids[: args.sample_size]
    return sampled[: args.sample_size] if args.sample_size > 0 else sampled


def add_source_context(rows: List[Dict[str, Any]], records: List[Dict[str, Any]], assignments: List[Dict[str, Any]]) -> None:
    record_by_id = {row["record_id"]: row for row in records}
    eval_to_records: Dict[str, List[str]] = collections.defaultdict(list)
    for assignment in assignments:
        for key in ["target_eval_ids", "neighbor_eval_ids", "capability_eval_ids"]:
            for eval_id in assignment.get(key, []):
                eval_to_records[eval_id].append(assignment["record_id"])

    for row in rows:
        linked_records = eval_to_records.get(row["eval_id"], [])
        row["n_assigned_records"] = len(linked_records)
        if linked_records:
            record = record_by_id.get(linked_records[0], {})
            source = record.get("source", {})
            metadata = record.get("metadata", {})
            row["record_id"] = linked_records[0]
            row["dataset"] = source.get("dataset")
            row["domain"] = source.get("domain")
            row["freshness_group"] = metadata.get("seed_freshness_group")
            row["release_year"] = metadata.get("seed_release_year")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def group_key(row: Dict[str, Any], key: str) -> str:
        return str(row.get(key) or "unknown")

    summary: Dict[str, Any] = {
        "n_rows": len(rows),
        "overall": summarize_group(rows),
        "by_eval_type": {},
        "by_dataset": {},
        "by_domain": {},
        "by_freshness_group": {},
    }
    for key, out_key in [
        ("eval_type", "by_eval_type"),
        ("dataset", "by_dataset"),
        ("domain", "by_domain"),
        ("freshness_group", "by_freshness_group"),
    ]:
        groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            groups[group_key(row, key)].append(row)
        summary[out_key] = {name: summarize_group(group_rows) for name, group_rows in sorted(groups.items())}
    return summary


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    margins = [float(row["margin"]) for row in rows]
    return {
        "n": len(rows),
        "positive_margin_rate": mean(m > 0 for m in margins),
        "margin_quantiles": quantiles(margins),
        "correct_logprob_mean": mean(float(row["correct_logprob"]) for row in rows),
        "contrast_logprob_mean": mean(float(row["contrast_logprob"]) for row in rows),
    }


def write_report(path: Path, args: argparse.Namespace, summary: Dict[str, Any], bad_cases: List[Dict[str, Any]]) -> None:
    lines = [
        "# Stage 0 Margin Sanity",
        "",
        f"Model: `{args.model_local_path}`",
        f"Data: `{args.data_dir}`",
        f"Sample size: `{summary['n_rows']}`",
        "",
        "## Overall",
        "",
        f"- positive margin rate: `{summary['overall']['positive_margin_rate']}`",
        f"- mean margin: `{summary['overall']['margin_quantiles']['mean']}`",
        f"- median margin: `{summary['overall']['margin_quantiles']['median']}`",
        "",
        "## By Eval Type",
        "",
        "| eval_type | n | positive_margin_rate | mean_margin | median_margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for eval_type, row in summary["by_eval_type"].items():
        q = row["margin_quantiles"]
        lines.append(
            f"| `{eval_type}` | {row['n']} | {row['positive_margin_rate']:.4f} | {q['mean']:.4f} | {q['median']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Bad Cases",
            "",
            f"Rows with margin <= 0: `{len(bad_cases)}`",
            "",
            "See `margin_sanity_bad_cases.jsonl` for examples.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_names = [
        "run_config.json",
        "model_config.json",
        "data_manifest.json",
        "margin_sanity_rows.csv",
        "margin_sanity_by_eval_type.csv",
        "margin_sanity_by_dataset.csv",
        "margin_sanity_bad_cases.jsonl",
        "margin_sanity_summary.json",
        "STAGE0_MARGIN_SANITY.md",
    ]
    if any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite.")
    if args.overwrite:
        for name in output_names:
            (out_dir / name).unlink(missing_ok=True)

    config = vars(args)
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "model_local_path": args.model_local_path,
                "torch_dtype": args.torch_dtype,
                "device_map": args.device_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "data_manifest.json").write_text(
        json.dumps({"data_dir": args.data_dir}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records, eval_bank, assignments = load_tables(Path(args.data_dir))
    sampled_ids = sample_eval_ids(assignments, eval_bank, args)

    model, tokenizer = load_model_and_tokenizer(
        args.model_local_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    rows: List[Dict[str, Any]] = []
    for eval_id in tqdm(sampled_ids, desc="margin sanity"):
        item = eval_bank[eval_id]
        correct = continuation_logprob(model, tokenizer, item["prompt"], item["correct"])
        contrast = continuation_logprob(model, tokenizer, item["prompt"], item["contrast"])
        rows.append(
            {
                "eval_id": eval_id,
                "eval_type": item["eval_type"],
                "prompt": item["prompt"],
                "correct": item["correct"],
                "contrast": item["contrast"],
                "correct_logprob": correct["total_logprob"],
                "contrast_logprob": contrast["total_logprob"],
                "correct_mean_logprob": correct["mean_logprob"],
                "contrast_mean_logprob": contrast["mean_logprob"],
                "correct_num_tokens": correct["num_tokens"],
                "contrast_num_tokens": contrast["num_tokens"],
                "margin": correct["total_logprob"] - contrast["total_logprob"],
                "mean_margin": correct["mean_logprob"] - contrast["mean_logprob"],
            }
        )
    add_source_context(rows, records, assignments)

    summary = summarize(rows)
    bad_cases = [row for row in rows if float(row["margin"]) <= 0]
    bad_cases = sorted(bad_cases, key=lambda row: float(row["margin"]))

    write_csv(out_dir / "margin_sanity_rows.csv", rows)
    write_csv(out_dir / "margin_sanity_by_eval_type.csv", [
        {"eval_type": key, **value, **{f"margin_{k}": v for k, v in value["margin_quantiles"].items()}}
        for key, value in summary["by_eval_type"].items()
    ])
    write_csv(out_dir / "margin_sanity_by_dataset.csv", [
        {"dataset": key, **value, **{f"margin_{k}": v for k, v in value["margin_quantiles"].items()}}
        for key, value in summary["by_dataset"].items()
    ])
    with (out_dir / "margin_sanity_bad_cases.jsonl").open("w", encoding="utf-8") as f:
        for row in bad_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "margin_sanity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir / "STAGE0_MARGIN_SANITY.md", args, summary, bad_cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
