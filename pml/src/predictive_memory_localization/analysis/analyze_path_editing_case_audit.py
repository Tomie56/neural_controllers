from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a qualitative/quantitative case audit for held-out path-to-editing transfer. "
            "This uses existing joined rows and does not run model inference."
        )
    )
    parser.add_argument(
        "--joined-csv",
        default=(
            "pml/results/commonsenseqa_balanced_followup_144/"
            "rome/path_editing_transfer/path_editing_joined_rows.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "pml/results/commonsenseqa_balanced_followup_144/"
            "rome/path_editing_transfer/case_audit"
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def to_bool(value: Any) -> Optional[bool]:
    if value in {None, ""}:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def bool01(value: Any) -> int:
    return 1 if to_bool(value) else 0


def rate(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [to_bool(row.get(key)) for row in rows]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    return mean(1.0 if value else 0.0 for value in vals)


def quantile_threshold(rows: List[Dict[str, Any]], key: str, q: float) -> float:
    vals = sorted(to_float(row.get(key)) for row in rows)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def build_risk_scores(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    topk_neighbor_auc_hi = quantile_threshold(rows, "path_agop_topk_project_neighbor_damage_auc", 0.75)
    topk_pareto_hi = quantile_threshold(rows, "path_agop_topk_project_pareto_score", 0.75)
    top1_pareto_hi = quantile_threshold(rows, "path_agop_top1_pareto_score", 0.75)
    top1_target_auc_hi = quantile_threshold(rows, "path_agop_top1_target_auc", 0.75)
    out: List[Dict[str, Any]] = []
    for row in rows:
        topk_path = row.get("path_agop_topk_project_path_type", "")
        top1_path = row.get("path_agop_top1_path_type", "")
        topk_damage_auc = to_float(row.get("path_agop_topk_project_neighbor_damage_auc"))
        topk_pareto = to_float(row.get("path_agop_topk_project_pareto_score"))
        top1_pareto = to_float(row.get("path_agop_top1_pareto_score"))
        top1_target_auc = to_float(row.get("path_agop_top1_target_auc"))
        topk_damage_minus_target = to_float(row.get("path_agop_topk_project_damage_minus_target_onset"))
        top1_damage_minus_target = to_float(row.get("path_agop_top1_damage_minus_target_onset"))

        risk = 0.0
        reasons = []
        if topk_path == "collapse":
            risk += 3.0
            reasons.append("topk_collapse")
        if topk_path == "damage-first":
            risk += 2.0
            reasons.append("topk_damage_first")
        if bool01(row.get("path_agop_topk_project_collapse_flag")):
            risk += 2.0
            reasons.append("topk_collapse_flag")
        if topk_damage_auc >= topk_neighbor_auc_hi:
            risk += 1.0
            reasons.append("topk_neighbor_auc_high")
        if topk_pareto >= topk_pareto_hi:
            risk += 1.0
            reasons.append("topk_pareto_high")
        if topk_damage_minus_target < 0:
            risk += 1.0
            reasons.append("topk_damage_before_target")
        if top1_path == "damage-first":
            risk += 1.0
            reasons.append("top1_damage_first")

        clean = 0.0
        clean_reasons = []
        if top1_path == "clean-window":
            clean += 2.0
            clean_reasons.append("top1_clean_window")
        if bool01(row.get("path_agop_top1_clean_window_exists")):
            clean += 1.0
            clean_reasons.append("top1_clean_exists")
        if top1_pareto >= top1_pareto_hi:
            clean += 1.0
            clean_reasons.append("top1_pareto_high")
        if top1_target_auc >= top1_target_auc_hi:
            clean += 1.0
            clean_reasons.append("top1_target_auc_high")
        if top1_damage_minus_target > 0:
            clean += 1.0
            clean_reasons.append("top1_damage_after_target")
        if topk_path == "clean-window":
            clean += 1.0
            clean_reasons.append("topk_clean_window")

        item = dict(row)
        item.update(
            {
                "path_fragile_risk_score": risk,
                "path_clean_reliability_score": clean,
                "path_fragile_risk_reasons": ";".join(reasons),
                "path_clean_reliability_reasons": ";".join(clean_reasons),
            }
        )
        out.append(item)
    return out


def compact_case(row: Dict[str, Any], rank: int, category: str) -> Dict[str, Any]:
    keep = {
        "rank": rank,
        "category": category,
        "id": row.get("id"),
        "source_dataset": row.get("source_dataset"),
        "subject": row.get("subject"),
        "target_new": row.get("target_new"),
        "ground_truth": row.get("ground_truth"),
        "prompt": row.get("prompt"),
        "standard_success_bool": row.get("standard_success_bool"),
        "robust_success_bool": row.get("robust_success_bool"),
        "fragile_success_bool": row.get("fragile_success_bool"),
        "locality_damage_bool": row.get("locality_damage_bool"),
        "pre_margin": row.get("pre_margin"),
        "post_margin": row.get("post_margin"),
        "rewrite_margin_delta": row.get("rewrite_margin_delta"),
        "rephrase_margin_delta": row.get("rephrase_margin_delta"),
        "locality_logprob_delta": row.get("locality_logprob_delta"),
        "path_fragile_risk_score": row.get("path_fragile_risk_score"),
        "path_fragile_risk_reasons": row.get("path_fragile_risk_reasons"),
        "path_clean_reliability_score": row.get("path_clean_reliability_score"),
        "path_clean_reliability_reasons": row.get("path_clean_reliability_reasons"),
        "path_agop_top1_path_type": row.get("path_agop_top1_path_type"),
        "path_agop_top1_target_onset_strength": row.get("path_agop_top1_target_onset_strength"),
        "path_agop_top1_damage_onset_strength": row.get("path_agop_top1_damage_onset_strength"),
        "path_agop_top1_clean_window_width": row.get("path_agop_top1_clean_window_width"),
        "path_agop_topk_project_path_type": row.get("path_agop_topk_project_path_type"),
        "path_agop_topk_project_target_onset_strength": row.get("path_agop_topk_project_target_onset_strength"),
        "path_agop_topk_project_damage_onset_strength": row.get("path_agop_topk_project_damage_onset_strength"),
        "path_agop_topk_project_clean_window_width": row.get("path_agop_topk_project_clean_window_width"),
        "path_agop_topk_project_collapse_flag": row.get("path_agop_topk_project_collapse_flag"),
    }
    return keep


def select_cases(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    categories = [
        (
            "high_path_risk_fragile",
            lambda r: to_bool(r.get("fragile_success_bool")) is True,
            lambda r: (-to_float(r.get("path_fragile_risk_score")), -to_float(r.get("superficial_risk_score")), r.get("id", "")),
        ),
        (
            "high_path_risk_not_fragile",
            lambda r: to_bool(r.get("fragile_success_bool")) is False and to_float(r.get("path_fragile_risk_score")) > 0,
            lambda r: (-to_float(r.get("path_fragile_risk_score")), -to_float(r.get("superficial_risk_score")), r.get("id", "")),
        ),
        (
            "clean_path_robust",
            lambda r: to_bool(r.get("robust_success_bool")) is True and to_float(r.get("path_clean_reliability_score")) > 0,
            lambda r: (-to_float(r.get("path_clean_reliability_score")), -to_float(r.get("rewrite_margin_delta")), r.get("id", "")),
        ),
        (
            "clean_path_but_fragile",
            lambda r: to_bool(r.get("fragile_success_bool")) is True and to_float(r.get("path_clean_reliability_score")) > 0,
            lambda r: (-to_float(r.get("path_clean_reliability_score")), -to_float(r.get("path_fragile_risk_score")), r.get("id", "")),
        ),
        (
            "diagnostic_counterexample_high_risk_robust",
            lambda r: to_bool(r.get("robust_success_bool")) is True and to_float(r.get("path_fragile_risk_score")) >= 3,
            lambda r: (-to_float(r.get("path_fragile_risk_score")), r.get("id", "")),
        ),
    ]
    per_category = max(3, top_k // len(categories))
    for category, pred, sort_key in categories:
        subset = sorted([row for row in rows if pred(row)], key=sort_key)[:per_category]
        for rank, row in enumerate(subset, start=1):
            cases.append(compact_case(row, rank, category))
    return cases


def aggregate_by_risk(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    buckets = [
        ("risk_0", lambda r: to_float(r.get("path_fragile_risk_score")) == 0),
        ("risk_1_2", lambda r: 1 <= to_float(r.get("path_fragile_risk_score")) <= 2),
        ("risk_3_4", lambda r: 3 <= to_float(r.get("path_fragile_risk_score")) <= 4),
        ("risk_5_plus", lambda r: to_float(r.get("path_fragile_risk_score")) >= 5),
        ("clean_0", lambda r: to_float(r.get("path_clean_reliability_score")) == 0),
        ("clean_1_2", lambda r: 1 <= to_float(r.get("path_clean_reliability_score")) <= 2),
        ("clean_3_plus", lambda r: to_float(r.get("path_clean_reliability_score")) >= 3),
    ]
    for bucket, pred in buckets:
        subset = [row for row in rows if pred(row)]
        if not subset:
            continue
        out.append(
            {
                "bucket": bucket,
                "n_records": len(subset),
                "standard_success_rate": rate(subset, "standard_success_bool"),
                "robust_success_rate": rate(subset, "robust_success_bool"),
                "fragile_success_rate": rate(subset, "fragile_success_bool"),
                "locality_damage_rate": rate(subset, "locality_damage_bool"),
                "rewrite_margin_delta_mean": mean(to_float(row.get("rewrite_margin_delta")) for row in subset),
                "superficial_risk_score_mean": mean(to_float(row.get("superficial_risk_score")) for row in subset),
                "top1_path_type_counts": dict(Counter(row.get("path_agop_top1_path_type", "") for row in subset)),
                "topk_path_type_counts": dict(Counter(row.get("path_agop_topk_project_path_type", "") for row in subset)),
            }
        )
    return out


def build_markdown(summary: Dict[str, Any], bucket_rows: List[Dict[str, Any]], case_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Path-to-Editing Case Audit",
        "",
        "This audit inspects which held-out 144 examples drive the path-geometry to editing-risk signal.",
        "",
        "## Inputs",
        "",
        f"- joined rows: `{summary['joined_csv']}`",
        f"- records: `{summary['n_records']}`",
        "",
        "## Risk Buckets",
        "",
        "| Bucket | n | standard | robust | fragile | locality damage | rewrite margin mean | superficial risk mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_rows:
        lines.append(
            "| {bucket} | {n_records} | {standard:.4f} | {robust:.4f} | {fragile:.4f} | {locality:.4f} | {rewrite:.4f} | {risk:.4f} |".format(
                bucket=row["bucket"],
                n_records=row["n_records"],
                standard=row["standard_success_rate"],
                robust=row["robust_success_rate"],
                fragile=row["fragile_success_rate"],
                locality=row["locality_damage_rate"],
                rewrite=row["rewrite_margin_delta_mean"],
                risk=row["superficial_risk_score_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Representative Cases",
            "",
            "| Category | Rank | ID | top1 path | topk path | fragile risk | clean score | standard | robust | fragile | target_new | ground_truth | reasons |",
            "|---|---:|---|---|---|---:|---:|---|---|---|---|---|---|",
        ]
    )
    for row in case_rows:
        lines.append(
            "| {category} | {rank} | `{id}` | {top1} | {topk} | {risk_score} | {clean_score} | {standard} | {robust} | {fragile} | {target_new} | {ground_truth} | {reasons} |".format(
                category=row.get("category", ""),
                rank=row.get("rank", ""),
                id=row.get("id", ""),
                top1=row.get("path_agop_top1_path_type", ""),
                topk=row.get("path_agop_topk_project_path_type", ""),
                risk_score=row.get("path_fragile_risk_score", ""),
                clean_score=row.get("path_clean_reliability_score", ""),
                standard=row.get("standard_success_bool", ""),
                robust=row.get("robust_success_bool", ""),
                fragile=row.get("fragile_success_bool", ""),
                target_new=str(row.get("target_new", "")).replace("|", "/"),
                ground_truth=str(row.get("ground_truth", "")).replace("|", "/"),
                reasons=str(row.get("path_fragile_risk_reasons", "")).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The case audit is diagnostic, not a new predictive benchmark.",
            "- High path-risk examples are useful for understanding failure modes such as top-k collapse or damage-first paths.",
            "- Counterexamples are expected: representation-level path risk is not equivalent to parameter-editing failure.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = build_risk_scores(read_csv(Path(args.joined_csv)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "path_editing_case_audit_scored_rows.csv"
    cases_path = output_dir / "path_editing_case_audit_cases.csv"
    buckets_path = output_dir / "path_editing_case_audit_buckets.csv"
    case_rows = select_cases(rows, args.top_k)
    bucket_rows = aggregate_by_risk(rows)
    write_csv(scored_path, rows)
    write_csv(cases_path, case_rows)
    write_csv(buckets_path, bucket_rows)
    summary = {
        "joined_csv": args.joined_csv,
        "n_records": len(rows),
        "top_k": args.top_k,
        "risk_score_counts": dict(Counter(str(row.get("path_fragile_risk_score")) for row in rows)),
        "clean_score_counts": dict(Counter(str(row.get("path_clean_reliability_score")) for row in rows)),
        "output_files": {
            "scored_rows": str(scored_path),
            "cases": str(cases_path),
            "buckets": str(buckets_path),
        },
    }
    (output_dir / "path_editing_case_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "PATH_EDITING_CASE_AUDIT.md").write_text(
        build_markdown(summary, bucket_rows, case_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "n_records": len(rows), "n_cases": len(case_rows)}, indent=2))


if __name__ == "__main__":
    main()
