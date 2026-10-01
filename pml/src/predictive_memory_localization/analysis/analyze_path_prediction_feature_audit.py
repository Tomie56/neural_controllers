from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_SUMMARIES = [
    "pml/results/path_geometry_audit/strength_prediction/strength_prediction_summary.json",
    "pml/results/path_geometry_audit/strength_prediction_with_method/strength_prediction_summary.json",
    "pml/results/heldout_100_dense_alpha_baselines/path_geometry_audit/strength_prediction/strength_prediction_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract top path-prediction coefficients into paper-friendly feature audit tables."
    )
    parser.add_argument("--summary-json", nargs="+", default=DEFAULT_SUMMARIES)
    parser.add_argument(
        "--output-dir",
        default="pml/results/path_geometry_audit/feature_audit",
    )
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


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


def f4(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def feature_family(feature: str) -> str:
    if feature.startswith("method="):
        return "method_indicator"
    if "agop" in feature:
        if "top_vec" in feature:
            return "agop_topvec"
        if any(token in feature for token in ["top1_ratio", "topk_ratio", "spectral_entropy", "entropy", "effective_rank", "eigengap", "trace", "fro_norm", "condition_number"]):
            return "agop_spectrum"
        return "agop_other"
    if "saliency" in feature:
        return "saliency"
    if "direction_pairwise_cosine" in feature:
        return "direction_agreement"
    if "threshold_accuracy" in feature:
        return "probe_accuracy"
    if "cohen_d" in feature or "gap" in feature or "margin" in feature:
        return "separation_margin"
    if "top_saliency_layer" in feature:
        return "layer_location"
    return "other"


def iter_classification_models(summary: Dict[str, Any], source_file: str) -> Iterable[Dict[str, Any]]:
    include_method = bool(summary.get("include_method"))
    if "heldout_100_dense_alpha_baselines" in source_file:
        summary_source = "heldout_100_dense_alpha_baselines"
    elif source_file.endswith("strength_prediction_with_method/strength_prediction_summary.json"):
        summary_source = "path_geometry_with_method"
    else:
        summary_source = "path_geometry"
    for group_name, group in summary.get("model_summaries", {}).items():
        if group_name == "pooled":
            dataset = "pooled"
        elif group_name.startswith("dataset="):
            dataset = group_name.split("=", 1)[1]
        elif "|method=" in group_name:
            dataset = group_name.split("|", 1)[0].split("=", 1)[1]
        else:
            dataset = group_name
        for feature_group, feature_summary in group.get("feature_groups", {}).items():
            for outcome, model in feature_summary.get("classification", {}).items():
                if not model.get("available"):
                    continue
                yield {
                    "source_file": source_file,
                    "summary_source": summary_source,
                    "include_method": include_method,
                    "group_name": group_name,
                    "dataset": dataset,
                    "feature_group": feature_group,
                    "outcome": outcome,
                    "n": model.get("n"),
                    "positive_rate": model.get("positive_rate"),
                    "cv_auroc": model.get("cv_auroc"),
                    "cv_ap": model.get("cv_average_precision"),
                    "top_coefficients": model.get("top_coefficients", []),
                }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows: List[Dict[str, Any]] = []
    family_counter: Counter[tuple[str, str, str, bool]] = Counter()
    family_weight: defaultdict[tuple[str, str, str, bool], float] = defaultdict(float)
    for summary_path_text in args.summary_json:
        summary_path = Path(summary_path_text)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_file = str(summary_path)
        for model in iter_classification_models(summary, source_file):
            coeffs = sorted(
                model["top_coefficients"],
                key=lambda item: abs(float(item.get("coefficient", 0.0))),
                reverse=True,
            )[: args.top_k]
            for rank, item in enumerate(coeffs, start=1):
                feature = item.get("feature", "")
                coefficient = float(item.get("coefficient", 0.0))
                family = feature_family(feature)
                key = (str(model["dataset"]), str(model["outcome"]), family, bool(model["include_method"]))
                family_counter[key] += 1
                family_weight[key] += abs(coefficient)
                rows.append(
                    {
                        "source_file": source_file,
                        "summary_source": model["summary_source"],
                        "include_method": model["include_method"],
                        "dataset": model["dataset"],
                        "group_name": model["group_name"],
                        "feature_group": model["feature_group"],
                        "outcome": model["outcome"],
                        "n": model["n"],
                        "positive_rate": f4(model["positive_rate"]),
                        "cv_auroc": f4(model["cv_auroc"]),
                        "cv_ap": f4(model["cv_ap"]),
                        "rank": rank,
                        "feature": feature,
                        "feature_family": family,
                        "coefficient": f4(coefficient),
                        "abs_coefficient": f4(abs(coefficient)),
                    }
                )
    family_rows = []
    for (dataset, outcome, family, include_method), count in sorted(family_counter.items()):
        family_rows.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "feature_family": family,
                "include_method": include_method,
                "topk_count": count,
                "abs_coefficient_sum": f4(family_weight[(dataset, outcome, family, include_method)]),
            }
        )
    write_csv(output_dir / "path_prediction_top_coefficients.csv", rows)
    write_csv(output_dir / "path_prediction_feature_family_summary.csv", family_rows)
    summary = {
        "summary_json": args.summary_json,
        "top_k": args.top_k,
        "n_top_coefficient_rows": len(rows),
        "n_family_rows": len(family_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "feature_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Path Prediction Feature Audit",
        "",
        "This audit extracts top logistic coefficients from path-type prediction models.",
        "",
        "## Files",
        "",
        "- `path_prediction_top_coefficients.csv`",
        "- `path_prediction_feature_family_summary.csv`",
        "",
        "## Notes",
        "",
        "- Coefficients are from standardized logistic models in the existing strength-prediction summaries.",
        "- They are explanatory diagnostics, not causal feature attributions.",
        "- Feature-family counts are aggregated over top-k coefficients and should be read qualitatively.",
        "",
    ]
    (output_dir / "PATH_PREDICTION_FEATURE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
