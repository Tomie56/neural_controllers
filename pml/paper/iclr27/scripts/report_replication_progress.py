#!/usr/bin/env python3
"""Report compact-replication execution progress without interpreting results."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parents[2]
RESULT_ROOT = (
    REPO_ROOT
    / "pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500"
)

MODEL_SPECS = [
    {
        "key": "qwen3_5_2b_base",
        "label": "Qwen3.5-2B",
        "methods": ["random", "mean_difference", "logistic", "rfm_agop_top1"],
        "layers": [-18, -15],
        "records": 500,
    },
    {
        "key": "ministral_3_3b_base_2512",
        "label": "Ministral-3-3B",
        "methods": ["random", "mean_difference", "logistic", "rfm_agop_top1"],
        "layers": [-20, -16],
        "records": 500,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    minutes = int(round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def model_progress(spec: dict[str, Any], now: float) -> dict[str, Any]:
    root = RESULT_ROOT / spec["key"]
    env = parse_env(root / "replication_config.env")
    methods = env.get("METHODS", " ".join(spec["methods"])).split()
    layers = [int(value) for value in env.get("LAYERS", " ".join(map(str, spec["layers"]))).split()]
    records = int(env.get("MAX_RECORDS", spec["records"]))
    configs = []
    for method in methods:
        for layer in layers:
            out_dir = root / "outcomes" / method / f"layer_neg{abs(layer)}"
            summary = json_payload(out_dir / "method_summary.json")
            raw_path = out_dir / "raw_scores.jsonl"
            failures = line_count(out_dir / "failures.jsonl")
            completed = int(summary.get("n_records", 0)) if summary else line_count(raw_path)
            completed = min(completed, records)
            status = "complete" if completed >= records and failures == 0 else (
                "running" if completed > 0 else "pending"
            )
            started_at = None
            config_path = out_dir / "run_config.json"
            if config_path.exists():
                started_at = config_path.stat().st_mtime
            elif raw_path.exists():
                started_at = raw_path.stat().st_mtime
            seconds_per_record = None
            eta_seconds = None
            if status == "running" and started_at is not None and completed > 0:
                elapsed = max(now - started_at, 0.0)
                seconds_per_record = elapsed / completed
                eta_seconds = seconds_per_record * (records - completed)
            configs.append(
                {
                    "method": method,
                    "layer": layer,
                    "completed_records": completed,
                    "expected_records": records,
                    "status": status,
                    "failures": failures,
                    "seconds_per_record_naive": seconds_per_record,
                    "eta_seconds_naive": eta_seconds,
                    "raw_path": str(raw_path),
                }
            )

    total_units = records * len(methods) * len(layers)
    completed_units = sum(item["completed_records"] for item in configs)
    active = [item for item in configs if item["status"] == "running"]
    strict_summary = root / "prediction_analysis_strict_later/outcome_summary.csv"
    return {
        "model_key": spec["key"],
        "model_label": spec["label"],
        "root": str(root),
        "records": records,
        "methods": methods,
        "layers": layers,
        "completed_record_method_layer_units": completed_units,
        "total_record_method_layer_units": total_units,
        "progress_fraction": completed_units / total_units if total_units else 0.0,
        "complete_configs": sum(item["status"] == "complete" for item in configs),
        "total_configs": len(configs),
        "active_configs": active,
        "strict_summary_complete": strict_summary.exists() and strict_summary.stat().st_size > 0,
        "configs": configs,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Compact Replication Execution Progress",
        "",
        f"Snapshot time: `{payload['snapshot_time']}`.",
        "",
        "This file reports execution progress only. Partial raw-score rows are not",
        "scientific results and are never used by the manuscript updater.",
        "",
        "| Model | Record-method-layer units | Configs | Active config | Naive current ETA | Strict summary |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for model in payload["models"]:
        active = model["active_configs"]
        if active:
            item = active[0]
            active_label = (
                f"{item['method']} / {item['layer']}: "
                f"{item['completed_records']}/{item['expected_records']}"
            )
            eta = format_duration(item["eta_seconds_naive"])
        else:
            active_label = "--"
            eta = "--"
        lines.append(
            f"| {model['model_label']} | "
            f"{model['completed_record_method_layer_units']:,}/"
            f"{model['total_record_method_layer_units']:,} "
            f"({100 * model['progress_fraction']:.1f}%) | "
            f"{model['complete_configs']}/{model['total_configs']} | "
            f"{active_label} | {eta} | "
            f"{'complete' if model['strict_summary_complete'] else 'pending'} |"
        )
    lines.extend(
        [
            "",
            "## Configuration Detail",
            "",
            "| Model | Method | Layer | Records | Status | Failures |",
            "| --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for model in payload["models"]:
        for item in model["configs"]:
            lines.append(
                f"| {model['model_label']} | {item['method']} | {item['layer']} | "
                f"{item['completed_records']}/{item['expected_records']} | "
                f"{item['status']} | {item['failures']} |"
            )
    lines.extend(
        [
            "",
            "The ETA is a naive extrapolation from the current configuration's wall",
            "time and should not be used as a paper result or a guarantee for later",
            "methods, which may have different direction-fitting costs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    now = time.time()
    payload = {
        "snapshot_epoch": now,
        "snapshot_time": datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
        "result_root": str(RESULT_ROOT),
        "models": [model_progress(spec, now) for spec in MODEL_SPECS],
    }
    markdown = render_markdown(payload)
    if args.write:
        data_dir = PAPER_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "REPLICATION_PROGRESS.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (data_dir / "REPLICATION_PROGRESS.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
