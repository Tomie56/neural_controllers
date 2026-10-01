from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ROOT = "/data/neural_controllers/pml/results/fresh_multidomain_3000"
DEFAULT_OUTPUT = "/data/neural_controllers/pml/results/fresh_multidomain_3000/FRESH_MULTIDOMAIN_STAGE0_2_REPORT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a staged report for fresh multidomain Stage 0-2 runs.")
    parser.add_argument("--results-root", default=DEFAULT_ROOT)
    parser.add_argument("--model-slug", default="qwen3_1_7b")
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def method_dirs(stage_dir: Path) -> List[Path]:
    if not stage_dir.exists():
        return []
    direct = sorted([p for p in stage_dir.iterdir() if p.is_dir() and (p / "method_summary.json").exists()])
    if direct:
        return direct
    nested = []
    for method_dir in sorted([p for p in stage_dir.iterdir() if p.is_dir()]):
        nested.extend(sorted([p for p in method_dir.iterdir() if p.is_dir() and (p / "method_summary.json").exists()]))
    return nested


def display_path_name(path: Path) -> str:
    if path.parent and path.parent.name not in {"qwen3_1_7b", "qwen3_4b", "qwen3_7b"}:
        return f"{path.parent.name}/{path.name}"
    return path.name


def table_for_stage_methods(stage_dir: Path) -> List[str]:
    lines = [
        "| Path | Records | Supp any | Enh any | Clean supp | Clean enh | Bidir clean | Neg damage | Pos damage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_dir in method_dirs(stage_dir):
        summary = load_json(method_dir / "method_summary.json")
        if not summary:
            continue
        lines.append(
            "| `{path}` | {n} | {supp} | {enh} | {cs} | {ce} | {bc} | {nd} | {pd} |".format(
                path=display_path_name(method_dir),
                n=summary.get("n_records"),
                supp=fmt(summary.get("suppression_any_rate")),
                enh=fmt(summary.get("enhancement_any_rate")),
                cs=fmt(summary.get("clean_suppression_rate")),
                ce=fmt(summary.get("clean_enhancement_rate")),
                bc=fmt(summary.get("bidirectional_clean_rate")),
                nd=fmt(summary.get("negative_damage_any_rate")),
                pd=fmt(summary.get("positive_damage_any_rate")),
            )
        )
    return lines


def stage_available(path: Path) -> str:
    return "complete" if path.exists() else "missing"


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    model = args.model_slug
    stage0 = root / "stage0_margin_sanity" / model
    stage1 = root / "stage1_evaluator_smoke_50" / model
    stage2 = root / "stage2_bidirectional_pilot_500" / model

    lines = [
        "# Fresh Multidomain 3000 Stage 0-2 Report",
        "",
        f"Model: `{model}`",
        f"Results root: `{root}`",
        "",
        "## Status",
        "",
        "| Stage | Path | Status |",
        "|---|---|---|",
        f"| Stage 0 margin sanity | `{stage0}` | {stage_available(stage0 / 'margin_sanity_summary.json')} |",
        f"| Stage 1 evaluator smoke | `{stage1}` | {stage_available(stage1)} |",
        f"| Stage 2 bidirectional pilot | `{stage2}` | {stage_available(stage2)} |",
        "",
        "## Stage 0: Margin Sanity",
        "",
    ]

    stage0_summary = load_json(stage0 / "margin_sanity_summary.json")
    if stage0_summary:
        overall = stage0_summary.get("overall", {})
        lines.extend(
            [
                f"- rows: `{stage0_summary.get('n_rows')}`",
                f"- overall positive margin rate: `{fmt(overall.get('positive_margin_rate'))}`",
                f"- overall mean margin: `{fmt((overall.get('margin_quantiles') or {}).get('mean'))}`",
                "",
                "| Eval type | n | Positive margin rate | Mean margin | Median margin |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for eval_type, row in (stage0_summary.get("by_eval_type") or {}).items():
            q = row.get("margin_quantiles") or {}
            lines.append(
                f"| `{eval_type}` | {row.get('n')} | {fmt(row.get('positive_margin_rate'))} | {fmt(q.get('mean'))} | {fmt(q.get('median'))} |"
            )
        lines.append("")
    else:
        lines.extend(["Stage 0 summary not found.", ""])

    lines.extend(["## Stage 1: Evaluator Smoke 50", ""])
    stage1_table = table_for_stage_methods(stage1)
    if len(stage1_table) > 2:
        lines.extend(stage1_table)
    else:
        lines.append("Stage 1 method summaries not found.")
    lines.append("")

    lines.extend(["## Stage 2: Bidirectional Pilot 500", ""])
    stage2_table = table_for_stage_methods(stage2)
    if len(stage2_table) > 2:
        lines.extend(stage2_table)
    else:
        lines.append("Stage 2 method summaries not found.")
    lines.append("")

    lines.extend(
        [
            "## Next Decision",
            "",
            "- If Stage 0 positive margin rates are acceptable and Stage 1 has no runner failures, Stage 2 can be interpreted.",
            "- If Stage 2 curves are stable, proceed to Stage 3a full 3000 on Qwen3-1.7B.",
            "- Do not start 4B/7B until the 1.7B Stage 2 pilot is clean.",
            "",
        ]
    )

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
