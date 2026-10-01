from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DEFAULT_STAGE2E = "/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2e_gradient_response/qwen3_1_7b"
DEFAULT_STAGE2G = "/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2g_adaptive_alpha_scan/qwen3_1_7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decide whether gradient-response/adaptive-alpha validation is ready to scale to 3000 records."
    )
    parser.add_argument("--stage2e-root", default=DEFAULT_STAGE2E)
    parser.add_argument("--stage2g-root", default=DEFAULT_STAGE2G)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-gradient-sign-accuracy", type=float, default=0.70)
    parser.add_argument("--min-gradient-effective-agreement", type=float, default=0.70)
    parser.add_argument("--min-adaptive-target-accuracy", type=float, default=0.80)
    parser.add_argument("--min-adaptive-clean-accuracy", type=float, default=0.75)
    parser.add_argument("--min-adaptive-damage-accuracy", type=float, default=0.75)
    parser.add_argument("--max-onset-error", type=float, default=0.15)
    return parser.parse_args()


def read_csv_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def find_row(df: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    rows = df[df[column].astype(str) == value]
    if rows.empty:
        return None
    return rows.iloc[0]


def metric_float(row: pd.Series | None, key: str) -> float | None:
    if row is None or key not in row:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    return float(value)


def status_from_checks(checks: List[Dict[str, Any]]) -> str:
    if any(check["status"] == "FAIL" for check in checks):
        return "FAIL"
    if any(check["status"] == "WARN" for check in checks):
        return "WARN"
    return "PASS"


def make_check(name: str, value: float | None, threshold: float, direction: str = ">=") -> Dict[str, Any]:
    if value is None:
        return {"name": name, "value": None, "threshold": threshold, "direction": direction, "status": "FAIL"}
    if direction == ">=":
        ok = value >= threshold
    elif direction == "<=":
        ok = value <= threshold
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "direction": direction,
        "status": "PASS" if ok else "FAIL",
    }


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_empty_"
    cols = ["name", "value", "direction", "threshold", "status"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    stage2e_root = Path(args.stage2e_root)
    stage2g_root = Path(args.stage2g_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gradient_metrics = read_csv_or_none(stage2e_root / "validation_vs_stage2_dense" / "gradient_vs_dense_metrics.csv")
    adaptive_labels = read_csv_or_none(stage2g_root / "validation_vs_stage2_dense" / "reduced_vs_dense_label_metrics.csv")
    adaptive_onsets = read_csv_or_none(stage2g_root / "validation_vs_stage2_dense" / "reduced_vs_dense_onset_metrics.csv")

    checks: List[Dict[str, Any]] = []
    if gradient_metrics is None:
        checks.append({"name": "gradient metrics file exists", "value": None, "threshold": 1.0, "direction": ">=", "status": "FAIL"})
    else:
        all_row = find_row(gradient_metrics, "method", "__all__")
        checks.append(
            make_check(
                "gradient sign_accuracy_nonzero",
                metric_float(all_row, "sign_accuracy_nonzero"),
                args.min_gradient_sign_accuracy,
            )
        )
        checks.append(
            make_check(
                "gradient effective_agreement",
                metric_float(all_row, "effective_agreement"),
                args.min_gradient_effective_agreement,
            )
        )

    if adaptive_labels is None:
        checks.append({"name": "adaptive label metrics file exists", "value": None, "threshold": 1.0, "direction": ">=", "status": "FAIL"})
    else:
        for label, threshold in [
            ("target_any", args.min_adaptive_target_accuracy),
            ("damage_any", args.min_adaptive_damage_accuracy),
            ("clean_suppression_window_exists", args.min_adaptive_clean_accuracy),
            ("clean_enhancement_window_exists", args.min_adaptive_clean_accuracy),
        ]:
            row = find_row(adaptive_labels, "label", label)
            checks.append(make_check(f"adaptive {label} accuracy", metric_float(row, "accuracy"), threshold))

    if adaptive_onsets is None:
        checks.append({"name": "adaptive onset metrics file exists", "value": None, "threshold": args.max_onset_error, "direction": "<=", "status": "WARN"})
    else:
        for onset in ["suppression", "enhancement", "negative_damage", "positive_damage"]:
            row = find_row(adaptive_onsets, "onset", onset)
            value = metric_float(row, "mean_abs_alpha_error")
            check = make_check(f"adaptive {onset} onset error", value, args.max_onset_error, "<=")
            # Missing onset pairs can happen in a tiny pilot; this is not a hard fail if labels are good.
            if check["status"] == "FAIL" and value is None:
                check["status"] = "WARN"
            checks.append(check)

    overall = status_from_checks(checks)
    if overall == "PASS":
        recommendation = (
            "Proceed to 3000-record gradient response and adaptive alpha scan. "
            "Keep a small dense-grid audit for representative method/layer groups."
        )
    elif overall == "WARN":
        recommendation = (
            "Run a 500-record validation before 3000 records. "
            "Use adaptive scan only for labels that passed; keep dense audit for uncertain onsets."
        )
    else:
        recommendation = (
            "Do not scale to 3000 yet. Inspect failed checks, then try larger pilot, second-order/few-point calibration, "
            "or less aggressive alpha reduction."
        )

    manifest = {
        "stage2e_root": str(stage2e_root),
        "stage2g_root": str(stage2g_root),
        "overall_status": overall,
        "recommendation": recommendation,
        "checks": checks,
    }
    (out_dir / "go_3000_decision.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# GO / NO-GO Decision for 3000-Record Gradient Response",
        "",
        f"- Stage2e root: `{stage2e_root}`",
        f"- Stage2g root: `{stage2g_root}`",
        f"- Overall status: **{overall}**",
        "",
        "## Checks",
        "",
        markdown_table(checks),
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## How to Interpret",
        "",
        "- `PASS`: the reduced/adaptive strategy is credible enough for 3000 records, with a small dense audit.",
        "- `WARN`: run a larger validation or keep dense scan for uncertain outcomes.",
        "- `FAIL`: the structural approximation is not yet reliable enough to replace dense alpha sweeps.",
        "",
    ]
    (out_dir / "GO_3000_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "overall_status": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
