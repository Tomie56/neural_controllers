from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parent))

from predictive_memory_localization.analysis.analyze_path_editing_transfer import (
    build_feature_matrix,
    to_bool,
)


DEFAULT_OUTCOMES = [
    "standard_success_bool",
    "robust_success_bool",
    "fragile_success_bool",
    "locality_damage_bool",
    "weak_post_margin_bool",
]

DEFAULT_FEATURE_SETS = [
    "diagnostic",
    "agop_top1",
    "agop_topk_project",
    "both",
    "diagnostic+agop_top1",
    "diagnostic+agop_topk_project",
    "diagnostic+both",
]

DEFAULT_COMPARISONS = [
    ("fragile_success_bool", "diagnostic", "diagnostic+agop_top1"),
    ("fragile_success_bool", "diagnostic", "diagnostic+both"),
    ("standard_success_bool", "diagnostic", "diagnostic+agop_top1"),
    ("robust_success_bool", "diagnostic", "diagnostic+agop_top1"),
    ("locality_damage_bool", "diagnostic", "diagnostic+both"),
    ("weak_post_margin_bool", "diagnostic", "diagnostic+agop_top1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap and paired-permutation stability checks for path-to-editing "
            "transfer metrics. This reuses already joined rows; it does not run a model."
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
            "rome/path_editing_transfer/stability"
        ),
    )
    parser.add_argument("--methods", nargs="+", default=["agop_top1", "agop_topk_project"])
    parser.add_argument("--outcomes", nargs="+", default=DEFAULT_OUTCOMES)
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
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


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def safe_metric(y_true: Sequence[int], score: Sequence[float], metric: str) -> Optional[float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return None
    if len(set(y_true)) < 2:
        return None
    if metric == "auroc":
        return float(roc_auc_score(y_true, score))
    if metric == "ap":
        return float(average_precision_score(y_true, score))
    raise ValueError(metric)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def valid_rows_and_labels(rows: List[Dict[str, Any]], outcome: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    valid = []
    labels = []
    for row in rows:
        value = to_bool(row.get(outcome))
        if value is None:
            continue
        valid.append(row)
        labels.append(1 if value else 0)
    return valid, labels


def oof_scores(
    rows: List[Dict[str, Any]],
    y: List[int],
    methods: List[str],
    feature_set: str,
    seed: int,
) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return None, {"available": False, "reason": f"sklearn_unavailable:{exc}"}

    counts = Counter(y)
    if len(counts) < 2 or min(counts.values()) < 5:
        return None, {"available": False, "reason": "insufficient_labels", "n": len(y), "positive_rate": sum(y) / len(y) if y else None}
    x, names = build_feature_matrix(rows, methods, feature_set)
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=int)
    folds = min(5, min(counts.values()))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probs = cross_val_predict(model, x_arr, y_arr, cv=cv, method="predict_proba")[:, 1]
    return probs.astype(float).tolist(), {
        "available": True,
        "n": len(y),
        "positive_rate": float(y_arr.mean()),
        "n_features": len(names),
        "cv_folds": folds,
        "feature_set": feature_set,
        "auroc": safe_metric(y, probs, "auroc"),
        "ap": safe_metric(y, probs, "ap"),
    }


def stratified_bootstrap_indices(y: Sequence[int], rng: Any) -> List[int]:
    pos = [idx for idx, value in enumerate(y) if value == 1]
    neg = [idx for idx, value in enumerate(y) if value == 0]
    return [rng.choice(pos) for _ in pos] + [rng.choice(neg) for _ in neg]


def bootstrap_single(
    y: List[int],
    score: List[float],
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    import random

    observed = safe_metric(y, score, metric)
    if observed is None:
        return {"observed": None}
    rng = random.Random(seed)
    vals = []
    for _ in range(n_bootstrap):
        idxs = stratified_bootstrap_indices(y, rng)
        metric_value = safe_metric([y[i] for i in idxs], [score[i] for i in idxs], metric)
        if metric_value is not None:
            vals.append(metric_value)
    return {
        "observed": observed,
        "boot_mean": sum(vals) / len(vals) if vals else None,
        "ci_low": percentile(vals, 0.025),
        "ci_high": percentile(vals, 0.975),
        "n_bootstrap_valid": len(vals),
    }


def bootstrap_delta(
    y: List[int],
    base_score: List[float],
    plus_score: List[float],
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    import random

    base_obs = safe_metric(y, base_score, metric)
    plus_obs = safe_metric(y, plus_score, metric)
    if base_obs is None or plus_obs is None:
        return {"base_observed": base_obs, "plus_observed": plus_obs, "delta_observed": None}
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_bootstrap):
        idxs = stratified_bootstrap_indices(y, rng)
        base_metric = safe_metric([y[i] for i in idxs], [base_score[i] for i in idxs], metric)
        plus_metric = safe_metric([y[i] for i in idxs], [plus_score[i] for i in idxs], metric)
        if base_metric is not None and plus_metric is not None:
            deltas.append(plus_metric - base_metric)
    delta_obs = plus_obs - base_obs
    return {
        "base_observed": base_obs,
        "plus_observed": plus_obs,
        "delta_observed": delta_obs,
        "delta_boot_mean": sum(deltas) / len(deltas) if deltas else None,
        "delta_ci_low": percentile(deltas, 0.025),
        "delta_ci_high": percentile(deltas, 0.975),
        "p_bootstrap_delta_le_zero": (sum(1 for v in deltas if v <= 0.0) / len(deltas)) if deltas else None,
        "n_bootstrap_valid": len(deltas),
    }


def paired_permutation_delta(
    y: List[int],
    base_score: List[float],
    plus_score: List[float],
    metric: str,
    n_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    import random

    base_obs = safe_metric(y, base_score, metric)
    plus_obs = safe_metric(y, plus_score, metric)
    if base_obs is None or plus_obs is None:
        return {"p_two_sided": None}
    observed = plus_obs - base_obs
    rng = random.Random(seed)
    extreme = 0
    valid = 0
    n = len(y)
    for _ in range(n_permutations):
        perm_base = list(base_score)
        perm_plus = list(plus_score)
        for i in range(n):
            if rng.random() < 0.5:
                perm_base[i], perm_plus[i] = perm_plus[i], perm_base[i]
        base_metric = safe_metric(y, perm_base, metric)
        plus_metric = safe_metric(y, perm_plus, metric)
        if base_metric is None or plus_metric is None:
            continue
        valid += 1
        if abs(plus_metric - base_metric) >= abs(observed):
            extreme += 1
    return {
        "p_two_sided": (extreme + 1) / (valid + 1) if valid else None,
        "n_permutations_valid": valid,
    }


def build_markdown(
    summary: Dict[str, Any],
    metric_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Path-to-Editing Transfer Stability Checks",
        "",
        "This report estimates uncertainty for the held-out 144 path-to-editing transfer result using out-of-fold predictions from existing joined rows.",
        "",
        "## Inputs",
        "",
        f"- joined rows: `{summary['joined_csv']}`",
        f"- n rows: `{summary['n_rows']}`",
        f"- bootstrap samples: `{summary['n_bootstrap']}`",
        f"- paired permutations: `{summary['n_permutations']}`",
        "",
        "## Single-Model Metric Intervals",
        "",
        "| Outcome | Feature set | n | positive rate | AUROC | AUROC 95% CI | AP | AP 95% CI |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in metric_rows:
        if row.get("metric") != "summary":
            continue
        lines.append(
            "| {outcome} | {feature_set} | {n} | {positive_rate:.4f} | {auroc:.4f} | [{auroc_ci_low:.4f}, {auroc_ci_high:.4f}] | {ap:.4f} | [{ap_ci_low:.4f}, {ap_ci_high:.4f}] |".format(
                outcome=row["outcome"],
                feature_set=row["feature_set"],
                n=row["n"],
                positive_rate=float(row["positive_rate"]),
                auroc=float(row["auroc"]),
                auroc_ci_low=float(row["auroc_ci_low"]),
                auroc_ci_high=float(row["auroc_ci_high"]),
                ap=float(row["ap"]),
                ap_ci_low=float(row["ap_ci_low"]),
                ap_ci_high=float(row["ap_ci_high"]),
            )
        )
    lines.extend(
        [
            "",
            "## Controlled Delta Stability",
            "",
            "| Outcome | Comparison | Metric | Base | Plus | Delta | Delta 95% CI | P(delta <= 0) | Paired permutation p |",
            "|---|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in delta_rows:
        lines.append(
            "| {outcome} | {comparison} | {metric} | {base_observed:.4f} | {plus_observed:.4f} | {delta_observed:.4f} | [{delta_ci_low:.4f}, {delta_ci_high:.4f}] | {p_bootstrap_delta_le_zero:.4f} | {p_permutation_two_sided:.4f} |".format(
                outcome=row["outcome"],
                comparison=row["comparison"],
                metric=row["metric"],
                base_observed=float(row["base_observed"]),
                plus_observed=float(row["plus_observed"]),
                delta_observed=float(row["delta_observed"]),
                delta_ci_low=float(row["delta_ci_low"]),
                delta_ci_high=float(row["delta_ci_high"]),
                p_bootstrap_delta_le_zero=float(row["p_bootstrap_delta_le_zero"]),
                p_permutation_two_sided=float(row["p_permutation_two_sided"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These intervals measure stability of the current 144-record transfer analysis; they do not replace a larger held-out run.",
            "- A useful PML signal should show positive controlled deltas with confidence intervals mostly above zero, especially for fragile / superficial risk.",
            "- If the fragile-risk delta remains positive but the interval crosses zero, the result should be framed as a promising diagnostic signal rather than a settled claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.joined_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores: Dict[Tuple[str, str], Dict[str, Any]] = {}
    metric_rows: List[Dict[str, Any]] = []
    for outcome in args.outcomes:
        valid_rows, y = valid_rows_and_labels(rows, outcome)
        for feature_set in args.feature_sets:
            score, meta = oof_scores(valid_rows, y, args.methods, feature_set, args.seed)
            if score is None:
                metric_rows.append({"metric": "summary", "outcome": outcome, "feature_set": feature_set, **meta})
                continue
            scores[(outcome, feature_set)] = {"y": y, "score": score, "meta": meta}
            auroc = bootstrap_single(y, score, "auroc", args.n_bootstrap, args.seed + 101)
            ap = bootstrap_single(y, score, "ap", args.n_bootstrap, args.seed + 202)
            metric_rows.append(
                {
                    "metric": "summary",
                    "outcome": outcome,
                    "feature_set": feature_set,
                    "n": meta["n"],
                    "positive_rate": meta["positive_rate"],
                    "n_features": meta["n_features"],
                    "cv_folds": meta["cv_folds"],
                    "auroc": auroc["observed"],
                    "auroc_ci_low": auroc["ci_low"],
                    "auroc_ci_high": auroc["ci_high"],
                    "ap": ap["observed"],
                    "ap_ci_low": ap["ci_low"],
                    "ap_ci_high": ap["ci_high"],
                }
            )

    delta_rows: List[Dict[str, Any]] = []
    for outcome, base, plus in DEFAULT_COMPARISONS:
        if outcome not in args.outcomes:
            continue
        base_item = scores.get((outcome, base))
        plus_item = scores.get((outcome, plus))
        if not base_item or not plus_item:
            continue
        y = base_item["y"]
        for metric in ["auroc", "ap"]:
            boot = bootstrap_delta(
                y,
                base_item["score"],
                plus_item["score"],
                metric,
                args.n_bootstrap,
                args.seed + 303,
            )
            perm = paired_permutation_delta(
                y,
                base_item["score"],
                plus_item["score"],
                metric,
                args.n_permutations,
                args.seed + 404,
            )
            delta_rows.append(
                {
                    "outcome": outcome,
                    "comparison": f"{plus}_over_{base}",
                    "metric": metric,
                    **boot,
                    "p_permutation_two_sided": perm["p_two_sided"],
                    "n_permutations_valid": perm["n_permutations_valid"],
                }
            )

    write_csv(output_dir / "stability_metric_intervals.csv", metric_rows)
    write_csv(output_dir / "stability_controlled_deltas.csv", delta_rows)
    summary = {
        "joined_csv": args.joined_csv,
        "n_rows": len(rows),
        "methods": args.methods,
        "outcomes": args.outcomes,
        "feature_sets": args.feature_sets,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "metric_rows": metric_rows,
        "delta_rows": delta_rows,
    }
    (output_dir / "stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "PATH_EDITING_TRANSFER_STABILITY.md").write_text(
        build_markdown(summary, metric_rows, delta_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "n_rows": len(rows), "n_delta_rows": len(delta_rows)}, indent=2))


if __name__ == "__main__":
    main()
