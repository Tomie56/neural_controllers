from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.analysis.analyze_editing_grouped_prediction import (
    OUTCOMES,
    average_precision,
    auroc,
    feature_sets,
    grouped_logistic_report,
    merge_rows,
    read_csv,
    source_by_id,
    split_rows,
    to_float,
    write_csv,
)
from predictive_memory_localization.common import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROME 500 error analysis and source-specific AGOP ablations."
    )
    parser.add_argument("--editing-results-jsonl", required=True)
    parser.add_argument("--baseline-joined-csv", required=True)
    parser.add_argument("--agop-joined-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def safe_len(text: Any) -> int:
    return len(str(text or ""))


def safe_words(text: Any) -> int:
    return len(re.findall(r"\w+", str(text or "")))


def ratio(numer: float, denom: float) -> Optional[float]:
    if denom == 0:
        return None
    return numer / denom


def first_metric(metrics: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    value = metrics.get(key)
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return None


def mean_metric(metrics: Dict[str, Any], key: str, field: str) -> Optional[float]:
    value = metrics.get(key)
    if not isinstance(value, list):
        return None
    vals = [to_float(item.get(field)) for item in value if isinstance(item, dict)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def result_diagnostics(row: Dict[str, Any]) -> Dict[str, Any]:
    task = row.get("task", {}) if isinstance(row.get("task"), dict) else {}
    pre = row.get("pre_metrics", {}) if isinstance(row.get("pre_metrics"), dict) else {}
    post = row.get("post_metrics", {}) if isinstance(row.get("post_metrics"), dict) else {}
    metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
    source = task.get("source", {}) if isinstance(task.get("source"), dict) else {}

    pre_rewrite = first_metric(pre, "rewrite") or {}
    pre_old = first_metric(pre, "old_answer") or {}
    post_rewrite = first_metric(post, "rewrite") or {}
    post_old = first_metric(post, "old_answer") or {}
    prompt = str(task.get("prompt", ""))
    subject = str(task.get("subject", ""))
    target_new = str(task.get("target_new", ""))
    ground_truth = str(task.get("ground_truth", ""))
    paraphrases = task.get("paraphrase_prompts", [])
    locality = task.get("locality_prompts", [])

    out: Dict[str, Any] = {
        "id": row.get("id", ""),
        "source_dataset": source.get("dataset", "unknown"),
        "prompt": prompt,
        "subject": subject,
        "target_new": target_new,
        "ground_truth": ground_truth,
        "prompt_chars": safe_len(prompt),
        "prompt_words": safe_words(prompt),
        "subject_chars": safe_len(subject),
        "subject_words": safe_words(subject),
        "target_new_chars": safe_len(target_new),
        "target_new_words": safe_words(target_new),
        "ground_truth_chars": safe_len(ground_truth),
        "ground_truth_words": safe_words(ground_truth),
        "num_paraphrase_prompts": len(paraphrases) if isinstance(paraphrases, list) else 0,
        "num_locality_prompts": len(locality) if isinstance(locality, list) else 0,
        "subject_prompt_pos": prompt.lower().find(subject.lower()) if subject else -1,
        "subject_prompt_char_ratio": "",
        "pre_rewrite_logprob": to_float(pre_rewrite.get("total_logprob")),
        "pre_old_logprob": to_float(pre_old.get("total_logprob")),
        "pre_rewrite_mean_logprob": to_float(pre_rewrite.get("mean_logprob")),
        "pre_old_mean_logprob": to_float(pre_old.get("mean_logprob")),
        "pre_rewrite_num_tokens": to_float(pre_rewrite.get("num_tokens")),
        "pre_old_num_tokens": to_float(pre_old.get("num_tokens")),
        "post_rewrite_logprob": to_float(post_rewrite.get("total_logprob")),
        "post_old_logprob": to_float(post_old.get("total_logprob")),
        "post_rewrite_mean_logprob": to_float(post_rewrite.get("mean_logprob")),
        "post_old_mean_logprob": to_float(post_old.get("mean_logprob")),
        "pre_mean_rephrase_new_logprob": mean_metric(pre, "rephrase", "total_logprob"),
        "post_mean_rephrase_new_logprob": mean_metric(post, "rephrase", "total_logprob"),
        "pre_mean_locality_logprob": mean_metric(pre, "locality", "total_logprob"),
        "post_mean_locality_logprob": mean_metric(post, "locality", "total_logprob"),
        "pre_mean_attack_new_logprob": mean_metric(pre, "attack_new", "total_logprob"),
        "pre_mean_attack_old_logprob": mean_metric(pre, "attack_old", "total_logprob"),
        "post_mean_attack_new_logprob": mean_metric(post, "attack_new", "total_logprob"),
        "post_mean_attack_old_logprob": mean_metric(post, "attack_old", "total_logprob"),
        "pre_mean_attack_new_minus_old_logprob": to_float(pre.get("mean_attack_new_minus_old_logprob")),
        "post_mean_attack_new_minus_old_logprob": to_float(post.get("mean_attack_new_minus_old_logprob")),
        "pre_attack_new_beats_old_rate": to_float(pre.get("attack_new_beats_old_rate")),
        "post_attack_new_beats_old_rate": to_float(post.get("attack_new_beats_old_rate")),
        "efficacy": float(bool(metrics.get("post_rewrite_beats_old"))),
        "rewrite_margin_delta": to_float(metrics.get("rewrite_margin_delta")),
        "rephrase_margin_delta": to_float(metrics.get("mean_rephrase_margin_delta")),
        "locality": to_float(metrics.get("mean_locality_preserved")),
        "locality_logprob_delta": to_float(metrics.get("mean_locality_logprob_delta")),
        "attack_margin_delta": to_float(metrics.get("mean_attack_margin_delta")),
        "attack_old_resurgence_rate": to_float(metrics.get("post_attack_old_resurgence_rate")),
        "attack_success_rate": to_float(metrics.get("post_attack_new_beats_old_rate")),
        "num_attack_prompts": to_float(metrics.get("num_attack_prompts")),
    }
    if out["subject_prompt_pos"] is not None and out["subject_prompt_pos"] >= 0:
        value = ratio(float(out["subject_prompt_pos"]), max(1.0, float(out["prompt_chars"])))
        out["subject_prompt_char_ratio"] = value if value is not None else ""
    return out


def load_diagnostics(path: str) -> List[Dict[str, Any]]:
    return [result_diagnostics(row) for row in load_jsonl(path)]


def numeric_columns(rows: List[Dict[str, Any]]) -> List[str]:
    cols = sorted({key for row in rows for key in row})
    out = []
    for col in cols:
        vals = [to_float(row.get(col)) for row in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 20 and len(set(vals)) > 1:
            out.append(col)
    return out


def median_or_empty(vals: Sequence[float]) -> Any:
    return float(np.median(vals)) if vals else ""


def mean_or_empty(vals: Sequence[float]) -> Any:
    return float(np.mean(vals)) if vals else ""


def quantile_group_report(
    rows: List[Dict[str, Any]],
    score_key: str,
    outcome_keys: Sequence[str],
    aux_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    scored = [(row, to_float(row.get(score_key))) for row in rows]
    scored = [(row, score) for row, score in scored if score is not None]
    if len(scored) < 20:
        return []
    scores = np.array([score for _, score in scored], dtype=float)
    low_cut = float(np.quantile(scores, 0.25))
    high_cut = float(np.quantile(scores, 0.75))
    groups = {
        "bottom_quartile": [row for row, score in scored if score <= low_cut],
        "middle_half": [row for row, score in scored if low_cut < score < high_cut],
        "top_quartile": [row for row, score in scored if score >= high_cut],
    }
    reports = []
    for group_name, group_rows in groups.items():
        if not group_rows:
            continue
        report: Dict[str, Any] = {
            "score_key": score_key,
            "group": group_name,
            "n": len(group_rows),
            "score_min": min(to_float(row.get(score_key)) for row in group_rows),
            "score_max": max(to_float(row.get(score_key)) for row in group_rows),
        }
        for key in list(outcome_keys) + list(aux_keys):
            vals = [to_float(row.get(key)) for row in group_rows]
            vals = [v for v in vals if v is not None]
            report[f"{key}_mean"] = mean_or_empty(vals)
            report[f"{key}_median"] = median_or_empty(vals)
        reports.append(report)
    return reports


def diagnostic_correlations(
    rows: List[Dict[str, Any]],
    diag_cols: Sequence[str],
    outcome_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    reports = []
    for outcome in outcome_keys:
        y_pairs = [(idx, to_float(row.get(outcome))) for idx, row in enumerate(rows)]
        y_pairs = [(idx, y) for idx, y in y_pairs if y is not None]
        if len(y_pairs) < 20:
            continue
        labels = np.array([1 if y >= 0.5 else 0 for _, y in y_pairs], dtype=int)
        if len(np.unique(labels)) < 2:
            continue
        for col in diag_cols:
            pairs = []
            for idx, y in y_pairs:
                x = to_float(rows[idx].get(col))
                if x is not None:
                    pairs.append((x, y))
            if len(pairs) < 20:
                continue
            x = np.array([p[0] for p in pairs], dtype=float)
            yy = np.array([p[1] for p in pairs], dtype=float)
            ll = np.array([1 if value >= 0.5 else 0 for value in yy], dtype=int)
            if len(np.unique(ll)) < 2:
                continue
            reports.append(
                {
                    "outcome": outcome,
                    "diagnostic": col,
                    "n": len(pairs),
                    "pearson": float(np.corrcoef(x, yy)[0, 1]) if np.std(x) > 0 and np.std(yy) > 0 else "",
                    "auroc_threshold_0_5": auroc(x, ll),
                    "average_precision_threshold_0_5": average_precision(x, ll),
                    "x_mean": float(np.mean(x)),
                    "x_median": float(np.median(x)),
                }
            )
    return sorted(
        reports,
        key=lambda row: abs(float(row["pearson"])) if row["pearson"] != "" else -1,
        reverse=True,
    )


def merge_diagnostics_with_features(
    diagnostics: List[Dict[str, Any]],
    merged_feature_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id = {row["id"]: row for row in diagnostics}
    out = []
    for feature_row in merged_feature_rows:
        row_id = str(feature_row.get("id", ""))
        diag = by_id.get(row_id)
        if not diag:
            continue
        merged = dict(feature_row)
        for key, value in diag.items():
            if key not in merged:
                merged[key] = value
            elif key in {"prompt", "subject", "target_new", "ground_truth"}:
                merged[f"diag_{key}"] = value
        out.append(merged)
    return out


def feature_ablation_sets(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    sets = feature_sets(rows)
    agop_cols = [
        col
        for col in rows[0].keys()
        if col.startswith("agop_feature_") and "agop_layer_" in col
    ]
    layer_spectrum = [
        col
        for col in agop_cols
        if any(
            token in col
            for token in [
                "condition_number",
                "effective_rank",
                "eigengap",
                "fro_norm",
                "spectral_entropy",
                "top1_ratio",
                "topk_ratio",
                "trace",
            ]
        )
    ]
    layer_topvec = [
        col
        for col in agop_cols
        if any(token in col for token in ["top_vec_abs_cos", "top_vec_cohen", "top_vec_mean", "top_vec_threshold"])
    ]
    by_layer: Dict[str, List[str]] = {}
    for layer in ["neg1", "neg5", "neg9", "neg13", "neg17", "neg21"]:
        by_layer[f"agop_{layer}_spectrum"] = [col for col in layer_spectrum if f"agop_layer_{layer}_" in col]
        by_layer[f"agop_{layer}_topvec"] = [col for col in layer_topvec if f"agop_layer_{layer}_" in col]
    projection_cols = [col for col in rows[0].keys() if col.startswith("agop_feature_proj_layer_")]
    projection_last = [col for col in projection_cols if "proj_layer_-1_" in col]
    projection_neg5 = [col for col in projection_cols if "proj_layer_-5_" in col]
    out = {
        "baseline_aggregate": sets.get("baseline_aggregate", []),
        "projection_last": projection_last,
        "projection_neg5": projection_neg5,
        "projection_all_layers": projection_cols,
        "agop_spectrum_all_layers": layer_spectrum,
        "agop_topvec_all_layers": layer_topvec,
        "agop_compact_spectrum": sets.get("agop_compact_spectrum", []),
        "agop_neg5_spectrum": sets.get("agop_neg5_spectrum", []),
        "agop_neg13_topvec": sets.get("agop_neg13_topvec", []),
        "projection_last_plus_agop_neg5": sets.get("projection_last_plus_agop_neg5", []),
    }
    out.update(by_layer)
    return {key: cols for key, cols in out.items() if cols}


def ablation_report(
    rows: List[Dict[str, Any]],
    *,
    bootstrap_iters: int,
    seed: int,
) -> List[Dict[str, Any]]:
    out = []
    for split_name, split in split_rows(rows).items():
        if split_name == "all":
            continue
        sets = feature_ablation_sets(split)
        for feature_set, cols in sets.items():
            for outcome in OUTCOMES:
                report = grouped_logistic_report(
                    split,
                    cols,
                    outcome,
                    bootstrap_iters=bootstrap_iters,
                    seed=seed,
                )
                row = {
                    "split": split_name,
                    "feature_set": feature_set,
                    "outcome": outcome,
                    "n_features": len(cols),
                    "available": bool(report.get("available")),
                    "reason": report.get("reason", ""),
                }
                if report.get("available"):
                    row.update(
                        {
                            "n": report["n"],
                            "positive_rate": report["positive_rate"],
                            "cv_auroc": report["cv_auroc"],
                            "auroc_ci_low": report.get("auroc_ci_low", ""),
                            "auroc_ci_high": report.get("auroc_ci_high", ""),
                            "cv_average_precision": report["cv_average_precision"],
                            "ap_ci_low": report.get("ap_ci_low", ""),
                            "ap_ci_high": report.get("ap_ci_high", ""),
                        }
                    )
                out.append(row)
    return out


def top_rows_by_outcome(rows: List[Dict[str, Any]], split: str, outcome: str, n: int = 10) -> List[Dict[str, Any]]:
    sub = [
        row
        for row in rows
        if row.get("source_dataset") == split
        and to_float(row.get(outcome)) is not None
    ]
    sub.sort(key=lambda row: float(to_float(row.get(outcome))), reverse=True)
    top = sub[:n]
    bottom = list(reversed(sub[-n:]))
    out = []
    for label, group in [("top", top), ("bottom", bottom)]:
        for row in group:
            out.append(
                {
                    "split": split,
                    "outcome": outcome,
                    "rank_group": label,
                    "id": row.get("id"),
                    "value": row.get(outcome),
                    "prompt": row.get("prompt"),
                    "target_new": row.get("target_new"),
                    "ground_truth": row.get("ground_truth"),
                    "pre_rewrite_logprob": row.get("pre_rewrite_logprob"),
                    "pre_old_logprob": row.get("pre_old_logprob"),
                    "target_new_words": row.get("target_new_words"),
                    "prompt_words": row.get("prompt_words"),
                }
            )
    return out


def summarize_sources(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for source, source_rows in split_rows(rows).items():
        if source == "all":
            continue
        item: Dict[str, Any] = {"n": len(source_rows)}
        for outcome in OUTCOMES:
            vals = [to_float(row.get(outcome)) for row in source_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                item[outcome] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "positive_rate": float(np.mean(np.array(vals) >= 0.5)),
                }
        summary[source] = item
    return summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = load_diagnostics(args.editing_results_jsonl)
    source_map = source_by_id(args.editing_results_jsonl)
    merged_feature_rows = merge_rows(
        read_csv(args.baseline_joined_csv),
        read_csv(args.agop_joined_csv),
        source_map,
    )
    rows = merge_diagnostics_with_features(diagnostics, merged_feature_rows)

    write_csv(out_dir / "rome500_error_rows.csv", rows)

    diag_cols = [
        col
        for col in numeric_columns(rows)
        if not col.startswith("feature_")
        and not col.startswith("agop_feature_")
        and not col.startswith("baseline_feature_")
        and col not in OUTCOMES
    ]
    corr_rows = diagnostic_correlations(rows, diag_cols, OUTCOMES)
    write_csv(out_dir / "diagnostic_correlations.csv", corr_rows)

    quartile_rows: List[Dict[str, Any]] = []
    for score_key in [
        "rewrite_margin_delta",
        "efficacy",
        "locality_logprob_delta",
        "agop_feature_agop_layer_neg5_top1_ratio",
        "agop_feature_agop_layer_neg13_top_vec_cohen_d",
        "agop_feature_agop_spectral_entropy_mean",
        "agop_feature_proj_layer_-1_cohen_d",
    ]:
        if any(score_key in row for row in rows):
            quartile_rows.extend(
                quantile_group_report(
                    rows,
                    score_key,
                    OUTCOMES,
                    [
                        "pre_rewrite_logprob",
                        "pre_old_logprob",
                        "pre_rewrite_num_tokens",
                        "target_new_words",
                        "prompt_words",
                        "subject_prompt_char_ratio",
                    ],
                )
            )
    write_csv(out_dir / "quartile_group_report.csv", quartile_rows)

    ablation_rows = ablation_report(rows, bootstrap_iters=args.bootstrap_iters, seed=args.seed)
    write_csv(out_dir / "source_feature_ablation_table.csv", ablation_rows)

    examples: List[Dict[str, Any]] = []
    for split in ["tau/commonsense_qa", "allenai/openbookqa", "allenai/ai2_arc"]:
        for outcome in ["rewrite_margin_delta", "locality", "locality_logprob_delta"]:
            examples.extend(top_rows_by_outcome(rows, split, outcome, n=8))
    write_csv(out_dir / "top_bottom_examples.csv", examples)

    summary = {
        "editing_results_jsonl": args.editing_results_jsonl,
        "n_rows": len(rows),
        "source_summary": summarize_sources(rows),
        "diagnostic_correlation_top_abs": corr_rows[:30],
        "ablation_top_by_split_outcome": {},
    }
    for split in sorted({row["split"] for row in ablation_rows if row.get("available")}):
        summary["ablation_top_by_split_outcome"][split] = {}
        for outcome in OUTCOMES:
            sub = [
                row
                for row in ablation_rows
                if row.get("split") == split
                and row.get("outcome") == outcome
                and row.get("available")
                and row.get("cv_auroc") not in ("", None)
            ]
            sub.sort(key=lambda row: float(row["cv_auroc"]), reverse=True)
            summary["ablation_top_by_split_outcome"][split][outcome] = sub[:8]

    (out_dir / "rome500_error_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "n_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
