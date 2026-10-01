from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check status of the path-representative capability-onset probe.")
    parser.add_argument(
        "--input-jsonl",
        default="/data/neural_controllers/pml/data/suppression_commonsenseqa_path_representative_50_with_capability.jsonl",
    )
    parser.add_argument(
        "--result-root",
        default="/data/neural_controllers/pml/results/path_representative_50_capability_onset_probe",
    )
    parser.add_argument(
        "--diagnostics-csv",
        default="/data/neural_controllers/pml/results/path_representative_subset/representative_50_diagnostics.csv",
    )
    parser.add_argument("--methods", nargs="+", default=["mean_difference", "logistic"])
    parser.add_argument(
        "--output-json",
        default="/data/neural_controllers/pml/results/path_representative_50_capability_onset_probe/status.json",
    )
    parser.add_argument(
        "--output-md",
        default="/data/neural_controllers/pml/results/path_representative_50_capability_onset_probe/STATUS.md",
    )
    return parser.parse_args()


def torch_cuda_status() -> Dict[str, Any]:
    code = (
        "import json\n"
        "import numpy\n"
        "import torch\n"
        "out={'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count(), 'devices': []}\n"
        "if out['cuda_available']:\n"
        "    out['devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
        "print(json.dumps(out))\n"
    )
    env = dict(os.environ)
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False, env=env)
    result = {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    try:
        result.update(json.loads(proc.stdout.strip().splitlines()[-1]))
    except Exception:
        result.update({"cuda_available": False, "device_count": 0, "devices": []})
    return result


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in load_jsonl(path))


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def file_info(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
    }


def method_status(result_root: Path, method: str) -> Dict[str, Any]:
    method_dir = result_root / method
    result_path = method_dir / "suppression_results.jsonl"
    config_path = method_dir / "config.json"
    n_result_rows = count_jsonl(result_path)
    return {
        "method": method,
        "method_dir": str(method_dir),
        "config": file_info(config_path),
        "results": file_info(result_path),
        "n_result_rows": n_result_rows,
        "state": "complete" if n_result_rows > 0 else ("config_only" if config_path.exists() else "missing"),
    }


def classify(report: Dict[str, Any]) -> str:
    if not report["input"]["exists"]:
        return "missing_input"
    if not report["diagnostics"]["exists"]:
        return "missing_diagnostics"
    method_states = [m["state"] for m in report["methods"]]
    if all(state == "complete" for state in method_states):
        if report["analysis"]["representative_summary"]["exists"]:
            return "complete_with_analysis"
        return "scoring_complete_analysis_missing"
    if any(state == "complete" for state in method_states):
        return "partial_scoring"
    if any(state == "config_only" for state in method_states):
        if not report["cuda"]["cuda_available"]:
            return "blocked_cuda_config_only"
        return "config_only_ready_to_rerun"
    if not report["cuda"]["cuda_available"]:
        return "blocked_cuda"
    return "ready_to_run"


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_jsonl)
    result_root = Path(args.result_root)
    diagnostics_path = Path(args.diagnostics_csv)
    summary_dir = result_root / "summary"
    path_audit_dir = result_root / "path_geometry_audit"
    representative_dir = path_audit_dir / "representative_outcomes"

    report: Dict[str, Any] = {
        "input": {
            **file_info(input_path),
            "n_records": count_jsonl(input_path),
        },
        "diagnostics": {
            **file_info(diagnostics_path),
            "n_lines": line_count(diagnostics_path),
        },
        "result_root": str(result_root),
        "cuda": torch_cuda_status(),
        "methods": [method_status(result_root, method) for method in args.methods],
        "analysis": {
            "summary_json": file_info(summary_dir / "summary.json"),
            "per_record_csv": file_info(summary_dir / "per_record.csv"),
            "path_rows_csv": file_info(path_audit_dir / "strength_path_rows.csv"),
            "path_summary_json": file_info(path_audit_dir / "strength_curve_summary.json"),
            "representative_summary": file_info(representative_dir / "REPRESENTATIVE_OUTCOME_SUMMARY.md"),
            "selection_reason_summary": file_info(representative_dir / "selection_reason_summary.csv"),
        },
    }
    report["state"] = classify(report)
    return report


def build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Path-Representative Capability Probe Status",
        "",
        f"State: `{report['state']}`",
        "",
        "## CUDA",
        "",
        f"- torch: `{report['cuda'].get('torch_version', '')}`",
        f"- cuda_available: `{report['cuda'].get('cuda_available')}`",
        f"- device_count: `{report['cuda'].get('device_count')}`",
        f"- devices: `{', '.join(report['cuda'].get('devices') or [])}`",
        "",
        "## Input",
        "",
        f"- input exists: `{report['input']['exists']}`",
        f"- input records: `{report['input']['n_records']}`",
        f"- diagnostics exists: `{report['diagnostics']['exists']}`",
        f"- diagnostics lines: `{report['diagnostics']['n_lines']}`",
        "",
        "## Methods",
        "",
        "| Method | State | Result rows | Config | Results |",
        "|---|---|---:|---:|---:|",
    ]
    for item in report["methods"]:
        lines.append(
            f"| {item['method']} | {item['state']} | {item['n_result_rows']} | "
            f"{item['config']['exists']} | {item['results']['exists']} |"
        )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "| Artifact | Exists | Size |",
            "|---|---:|---:|",
        ]
    )
    for key, info in report["analysis"].items():
        lines.append(f"| {key} | {info['exists']} | {info['size']} |")
    lines.extend(
        [
            "",
            "## Next Command",
            "",
        ]
    )
    if report["state"] in {"ready_to_run", "config_only_ready_to_rerun"}:
        lines.extend(["```bash", "bash pml/scripts/run_path_representative_capability_probe.sh", "```"])
    elif report["state"] == "scoring_complete_analysis_missing":
        lines.extend(["```bash", "bash pml/scripts/analyze_path_representative_capability_probe.sh", "```"])
    elif report["state"].startswith("blocked_cuda"):
        lines.append("Run the probe in an environment where `torch.cuda.is_available()` is true.")
    elif report["state"] == "partial_scoring":
        lines.append("Inspect method result files before deciding whether to resume or rerun with `OVERWRITE=0/1`.")
    else:
        lines.append("Inspect missing input/result artifacts above.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"state": report["state"], "output_json": str(output_json), "output_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
