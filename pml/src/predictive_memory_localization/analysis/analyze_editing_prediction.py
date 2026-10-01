from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl
from predictive_memory_localization.analysis.analyze_suppression_prediction import (
    average_precision,
    auroc,
    pearson,
    spearman,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether localization features predict ROME/MEMIT edit outcomes."
    )
    parser.add_argument("--editing-results-jsonl", required=True)
    parser.add_argument(
        "--suppression-per-record-csv",
        required=True,
        help="per_record.csv from suppression summary, containing feature_* columns.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def flatten_metrics(value: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten_metrics(item, f"{prefix}{key}."))
    elif isinstance(value, list):
        if value and all(isinstance(x, (int, float, bool)) for x in value):
            nums = [float(x) for x in value]
            out[prefix.rstrip(".") + ".mean"] = sum(nums) / len(nums)
        elif len(value) == 1:
            out.update(flatten_metrics(value[0], prefix))
    else:
        out[prefix.rstrip(".")] = value
    return out


def metric_value(flat: Dict[str, Any], candidates: List[str]) -> Optional[float]:
    for candidate in candidates:
        if candidate in flat:
            value = to_float(flat[candidate])
            if value is not None:
                return value
    for key, raw in flat.items():
        lower = key.lower()
        if any(candidate.lower() in lower for candidate in candidates):
            value = to_float(raw)
            if value is not None:
                return value
    return None


def load_feature_rows(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            # Keep one feature row per fact; prefer coef -0.5 logistic if present,
            # otherwise first row. Features are currently identical across coefs for a method.
            key = row.get("id", "")
            if not key:
                continue
            current = rows.get(key)
            preferred = row.get("control_method") == "logistic" and row.get("coef") == "-0.5"
            if current is None or preferred:
                rows[key] = row
    return rows


def load_editing_rows(path: str, feature_rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in load_jsonl(path):
        flat = flatten_metrics(result.get("metrics", {}))
        efficacy = metric_value(
            flat,
            [
                "post_rewrite_beats_old",
                "rewrite_acc",
                "post.rewrite_acc",
                "efficacy",
                "edit_success",
            ],
        )
        rewrite_margin_delta = metric_value(flat, ["rewrite_margin_delta"])
        paraphrase = metric_value(
            flat,
            [
                "post_rephrase_beats_old_rate",
                "rephrase_acc",
                "paraphrase",
                "generalization",
            ],
        )
        rephrase_margin_delta = metric_value(flat, ["mean_rephrase_margin_delta"])
        locality = metric_value(
            flat,
            [
                "mean_locality_preserved",
                "locality",
                "locality_acc",
                "neighborhood_acc",
            ],
        )
        locality_logprob_delta = metric_value(flat, ["mean_locality_logprob_delta"])
        row = {
            "id": result.get("id", ""),
            "method": result.get("method", ""),
            "efficacy": efficacy,
            "rewrite_margin_delta": rewrite_margin_delta,
            "paraphrase": paraphrase,
            "rephrase_margin_delta": rephrase_margin_delta,
            "locality": locality,
            "locality_logprob_delta": locality_logprob_delta,
        }
        features = feature_rows.get(row["id"], {})
        row.update({k: v for k, v in features.items() if k.startswith("feature_")})
        rows.append(row)
    return rows


def feature_columns(rows: List[Dict[str, Any]]) -> List[str]:
    cols = sorted({k for row in rows for k in row if k.startswith("feature_") and k != "feature_top_saliency_layer"})
    out = []
    for col in cols:
        vals = [to_float(row.get(col)) for row in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 3 and len(set(vals)) > 1:
            out.append(col)
    return out


def report_for_outcome(rows: List[Dict[str, Any]], outcome: str) -> List[Dict[str, Any]]:
    valid = [(idx, to_float(row.get(outcome))) for idx, row in enumerate(rows)]
    valid = [(idx, value) for idx, value in valid if value is not None]
    if len(valid) < 3:
        return []
    y = np.array([value for _, value in valid], dtype=float)
    labels = (y >= 0.5).astype(int)
    reports = []
    for feature in feature_columns(rows):
        pairs = []
        for idx, value in valid:
            fval = to_float(rows[idx].get(feature))
            if fval is not None:
                pairs.append((fval, value))
        if len(pairs) < 3:
            continue
        x = np.array([p[0] for p in pairs], dtype=float)
        yy = np.array([p[1] for p in pairs], dtype=float)
        ll = (yy >= 0.5).astype(int)
        reports.append(
            {
                "outcome": outcome,
                "feature": feature,
                "n": len(pairs),
                "pearson": pearson(x, yy),
                "spearman": spearman(x, yy),
                "auroc_threshold_0.5": auroc(x, ll),
                "average_precision_threshold_0.5": average_precision(x, ll),
            }
        )
    return sorted(reports, key=lambda row: abs(row["spearman"]), reverse=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = load_feature_rows(args.suppression_per_record_csv)
    rows = load_editing_rows(args.editing_results_jsonl, feature_rows)
    write_csv(out_dir / "editing_joined_rows.csv", rows)

    reports: List[Dict[str, Any]] = []
    for outcome in [
        "efficacy",
        "rewrite_margin_delta",
        "paraphrase",
        "rephrase_margin_delta",
        "locality",
        "locality_logprob_delta",
    ]:
        reports.extend(report_for_outcome(rows, outcome))
    write_csv(out_dir / "editing_feature_report.csv", reports)

    summary = {
        "editing_results_jsonl": args.editing_results_jsonl,
        "suppression_per_record_csv": args.suppression_per_record_csv,
        "n_rows": len(rows),
        "n_with_features": sum(1 for row in rows if any(k.startswith("feature_") for k in row)),
        "outcomes_available": {
            outcome: sum(1 for row in rows if to_float(row.get(outcome)) is not None)
            for outcome in [
                "efficacy",
                "rewrite_margin_delta",
                "paraphrase",
                "rephrase_margin_delta",
                "locality",
                "locality_logprob_delta",
            ]
        },
    }
    (out_dir / "editing_prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
