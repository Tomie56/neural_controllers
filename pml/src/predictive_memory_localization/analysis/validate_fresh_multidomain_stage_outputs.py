from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_ROOT = "/data/neural_controllers/pml/results/fresh_multidomain_3000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate fresh multidomain Stage 0-2 output completeness.")
    parser.add_argument("--results-root", default=DEFAULT_ROOT)
    parser.add_argument("--model-slug", default="qwen3_1_7b")
    parser.add_argument("--stage1-records", type=int, default=50)
    parser.add_argument("--stage2-records", type=int, default=500)
    parser.add_argument("--methods", nargs="*", default=["mean_difference", "logistic", "random"])
    parser.add_argument("--stage1-layers", type=int, nargs="*", default=[-1, -9, -17])
    parser.add_argument("--stage2-layers", type=int, nargs="*", default=[-1, -5, -9, -13, -17, -21])
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_method_dir(path: Path, expected_records: int) -> Dict[str, Any]:
    summary = load_json(path / "method_summary.json")
    failures = 0
    if (path / "failures.jsonl").exists():
        failures = sum(1 for line in (path / "failures.jsonl").open(encoding="utf-8") if line.strip())
    return {
        "path": str(path),
        "exists": path.exists(),
        "summary_exists": summary is not None,
        "raw_scores_exists": (path / "raw_scores.jsonl").exists(),
        "per_eval_exists": (path / "per_eval.csv").exists(),
        "per_record_exists": (path / "per_record.csv").exists(),
        "path_rows_exists": (path / "path_rows.csv").exists(),
        "n_records": summary.get("n_records") if summary else None,
        "expected_records": expected_records,
        "n_failures": summary.get("n_failures", failures) if summary else failures,
        "ok": bool(
            path.exists()
            and summary
            and (path / "raw_scores.jsonl").exists()
            and (path / "per_eval.csv").exists()
            and (path / "per_record.csv").exists()
            and (path / "path_rows.csv").exists()
            and summary.get("n_records") == expected_records
            and int(summary.get("n_failures", failures) or 0) == 0
        ),
    }


def layer_dir_name(layer: int) -> str:
    return f"layer_{str(layer).replace('-', 'neg')}"


def validate_stage_paths(stage_root: Path, methods: List[str], layers: List[int], expected_records: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for method in methods:
        out[method] = {}
        for layer in layers:
            out[method][str(layer)] = validate_method_dir(stage_root / method / layer_dir_name(layer), expected_records)
    return out


def all_stage_paths_ok(stage: Dict[str, Any]) -> bool:
    for layers in stage.values():
        if not all(row["ok"] for row in layers.values()):
            return False
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    model = args.model_slug
    out: Dict[str, Any] = {
        "results_root": str(root),
        "model_slug": model,
        "stage0": {},
        "stage1": {},
        "stage2": {},
    }

    stage0 = root / "stage0_margin_sanity" / model
    stage0_summary = load_json(stage0 / "margin_sanity_summary.json")
    out["stage0"] = {
        "path": str(stage0),
        "summary_exists": stage0_summary is not None,
        "rows_exists": (stage0 / "margin_sanity_rows.csv").exists(),
        "bad_cases_exists": (stage0 / "margin_sanity_bad_cases.jsonl").exists(),
        "n_rows": stage0_summary.get("n_rows") if stage0_summary else None,
        "ok": bool(stage0_summary and (stage0 / "margin_sanity_rows.csv").exists()),
    }

    stage1_root = root / "stage1_evaluator_smoke_50" / model
    out["stage1"] = validate_stage_paths(stage1_root, args.methods, args.stage1_layers, args.stage1_records)

    stage2_root = root / "stage2_bidirectional_pilot_500" / model
    out["stage2"] = validate_stage_paths(stage2_root, args.methods, args.stage2_layers, args.stage2_records)

    out["ok"] = (
        out["stage0"]["ok"]
        and all_stage_paths_ok(out["stage1"])
        and all_stage_paths_ok(out["stage2"])
    )

    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    output_json = Path(args.output_json) if args.output_json else root / "stage0_2_validation.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
