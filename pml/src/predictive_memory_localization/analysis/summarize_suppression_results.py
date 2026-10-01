from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize activation suppression result JSONL files."
    )
    parser.add_argument(
        "--results-jsonl",
        nargs="+",
        required=True,
        help="One or more suppression_results.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for summary.json, per_record.csv, and per_eval.csv.",
    )
    parser.add_argument(
        "--success-delta",
        type=float,
        default=-0.05,
        help="Target suppression success threshold on total logprob delta.",
    )
    parser.add_argument(
        "--neighbor-damage-delta",
        type=float,
        default=-0.05,
        help="Neighbor damage threshold on total logprob delta.",
    )
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def flatten_results(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for record in load_jsonl(path):
            method = record.get("control_method", "unknown")
            record_id = record.get("id", "")
            concept = record.get("concept", record_id)
            features = record.get("feature_summary", {})
            for eval_item in record.get("eval", []):
                kind = eval_item.get("kind", "")
                base = eval_item.get("base", {})
                for controlled in eval_item.get("controlled", []):
                    rows.append(
                        {
                            "source_file": path,
                            "id": record_id,
                            "concept": concept,
                            "control_method": method,
                            "kind": kind,
                            "eval_index": eval_item.get("index", 0),
                            "coef": float(controlled.get("coef")),
                            "base_total_logprob": float(base.get("total_logprob", 0.0)),
                            "base_mean_logprob": float(base.get("mean_logprob", 0.0)),
                            "delta_total_logprob": float(controlled.get("delta_total_logprob", 0.0)),
                            "delta_mean_logprob": float(controlled.get("delta_mean_logprob", 0.0)),
                            "num_tokens": int(controlled.get("num_tokens", base.get("num_tokens", 0))),
                            **{f"feature_{k}": v for k, v in features.items() if not isinstance(v, list)},
                        }
                    )
    return rows


def grouped_summary(rows: List[Dict[str, Any]], success_delta: float, damage_delta: float) -> Dict[str, Any]:
    groups: Dict[tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["control_method"], row["kind"], row["coef"])].append(row)

    by_method_kind_coef: List[Dict[str, Any]] = []
    for (method, kind, coef), group_rows in sorted(groups.items()):
        deltas = [row["delta_total_logprob"] for row in group_rows]
        out = {
            "control_method": method,
            "kind": kind,
            "coef": coef,
            "n_eval": len(group_rows),
            **{f"delta_total_logprob_{k}": v for k, v in quantiles(deltas).items()},
        }
        if kind == "target":
            out["target_success_rate"] = mean(d <= success_delta for d in deltas)
        if kind == "neighbor":
            out["neighbor_damage_rate"] = mean(d <= damage_delta for d in deltas)
            out["neighbor_preservation_rate"] = mean(d > damage_delta for d in deltas)
        if kind == "capability":
            out["capability_damage_rate"] = mean(d <= damage_delta for d in deltas)
            out["capability_preservation_rate"] = mean(d > damage_delta for d in deltas)
        by_method_kind_coef.append(out)

    record_groups: Dict[tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        record_groups[(row["control_method"], row["id"], row["coef"])].append(row)

    per_record: List[Dict[str, Any]] = []
    for (method, record_id, coef), group_rows in sorted(record_groups.items()):
        target_deltas = [r["delta_total_logprob"] for r in group_rows if r["kind"] == "target"]
        neighbor_deltas = [r["delta_total_logprob"] for r in group_rows if r["kind"] == "neighbor"]
        capability_deltas = [r["delta_total_logprob"] for r in group_rows if r["kind"] == "capability"]
        first = group_rows[0]
        per_record.append(
            {
                "control_method": method,
                "id": record_id,
                "concept": first.get("concept", record_id),
                "coef": coef,
                "target_mean_delta": mean(target_deltas),
                "target_median_delta": float(median(target_deltas)) if target_deltas else 0.0,
                "target_success": mean(target_deltas) <= success_delta if target_deltas else False,
                "neighbor_mean_delta": mean(neighbor_deltas),
                "neighbor_median_delta": float(median(neighbor_deltas)) if neighbor_deltas else 0.0,
                "neighbor_damaged": mean(neighbor_deltas) <= damage_delta if neighbor_deltas else False,
                "capability_mean_delta": mean(capability_deltas),
                "capability_median_delta": float(median(capability_deltas)) if capability_deltas else 0.0,
                "capability_damaged": mean(capability_deltas) <= damage_delta if capability_deltas else False,
                "n_target_eval": len(target_deltas),
                "n_neighbor_eval": len(neighbor_deltas),
                "n_capability_eval": len(capability_deltas),
                **{k: v for k, v in first.items() if k.startswith("feature_")},
            }
        )

    return {
        "n_eval_rows": len(rows),
        "n_records": len({row["id"] for row in rows}),
        "control_methods": sorted({row["control_method"] for row in rows}),
        "coefs": sorted({row["coef"] for row in rows}),
        "success_delta": success_delta,
        "neighbor_damage_delta": damage_delta,
        "by_method_kind_coef": by_method_kind_coef,
        "per_record": per_record,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = flatten_results(args.results_jsonl)
    summary = grouped_summary(rows, args.success_delta, args.neighbor_damage_delta)

    (output_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "per_record"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "per_eval.csv", rows)
    write_csv(output_dir / "per_record.csv", summary["per_record"])

    print(f"Wrote summary to {output_dir / 'summary.json'}")
    print(f"Wrote per-eval rows to {output_dir / 'per_eval.csv'}")
    print(f"Wrote per-record rows to {output_dir / 'per_record.csv'}")


if __name__ == "__main__":
    main()
