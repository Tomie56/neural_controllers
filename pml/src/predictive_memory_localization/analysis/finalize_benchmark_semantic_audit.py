from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


DEFAULT_SAMPLE_PATH = Path(
    "/data/neural_controllers/pml/results/fresh_multidomain_3000/"
    "benchmark_semantic_audit_20260722/semantic_audit_sample.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SAMPLE_PATH.parent

DIMENSIONS = (
    "source_grounding",
    "desired_contrast_validity",
    "target_consistency",
    "neighbor_locality",
    "capability_independence",
    "construction_quality",
)
SEVERITY = {"pass": 0, "minor": 1, "fail": 2}


def issue(status: str, note: str, **dimensions: str) -> Dict[str, str]:
    row = {dimension: "pass" for dimension in DIMENSIONS}
    row.update(dimensions)
    row["overall"] = status
    row["notes"] = note
    return row


ISSUES: Dict[int, Dict[str, str]] = {
    1: issue(
        "minor",
        "The capacity neighbor states that a canoe holds up to two people, which is not generally true.",
        neighbor_locality="minor",
    ),
    2: issue(
        "minor",
        "Several negative constructions and one contrast are fragmentary or unnatural (for example, 'walk towards').",
        target_consistency="minor",
        construction_quality="minor",
    ),
    4: issue(
        "minor",
        "The source completion describes rackets on an indoor court but does not explicitly establish squash.",
        source_grounding="minor",
    ),
    7: issue(
        "minor",
        "The polarity is clear, but several bleach substitutions are linguistically awkward as matched negatives.",
        construction_quality="minor",
    ),
    11: issue(
        "minor",
        "One neighbor generalizes that caregivers should live close, while the source only requires immediate access.",
        neighbor_locality="minor",
    ),
    13: issue(
        "minor",
        "Counterfactual belt-width neighbors use simple proportional approximations that are not fully justified by the stated mechanics.",
        neighbor_locality="minor",
    ),
    15: issue(
        "minor",
        "The target is valid, but several art neighbors use subjective or weakly exclusive contrasts.",
        neighbor_locality="minor",
    ),
    20: issue(
        "minor",
        "Calling the oscillator weakly damped is imprecise for the reported coefficient, although the numerical target matches the source.",
        desired_contrast_validity="minor",
    ),
    25: issue(
        "minor",
        "Air is more precisely separable into component gases or substances, not simply component elements.",
        desired_contrast_validity="minor",
        target_consistency="minor",
    ),
    27: issue(
        "minor",
        "The target overgeneralizes rivers as responsible for most valleys and canyons; glaciers are also major valley-forming agents.",
        target_consistency="minor",
    ),
    32: issue(
        "minor",
        "Water/solution capability items are not fully independent of the salt-dissolution record.",
        capability_independence="minor",
    ),
    33: issue(
        "minor",
        "The lake-to-swamp endpoint is a simplified succession account and is phrased too deterministically.",
        desired_contrast_validity="minor",
        target_consistency="minor",
    ),
    36: issue(
        "minor",
        "A solar cell does not necessarily replace a battery; many devices retain rechargeable storage.",
        neighbor_locality="minor",
    ),
    39: issue(
        "minor",
        "Water and fertilizer are not mutually exclusive plant requirements, and one photosynthesis neighbor also contrasts two required inputs.",
        desired_contrast_validity="minor",
        target_consistency="minor",
        neighbor_locality="minor",
    ),
    43: issue(
        "minor",
        "The source answer is a tire, while the constructed fact switches to wood; the concept is related but only indirectly grounded.",
        source_grounding="minor",
    ),
    45: issue(
        "minor",
        "The wording compresses evaporation and condensation into a single cloud-formation step, and the capability set is heavily water-themed.",
        desired_contrast_validity="minor",
        target_consistency="minor",
        capability_independence="minor",
    ),
    48: issue(
        "minor",
        "'Optimal angles' follows the source label but is vague without specifying flower orientation or the measured criterion.",
        desired_contrast_validity="minor",
        target_consistency="minor",
    ),
    49: issue(
        "minor",
        "Three of four capability prompts concern water phase behavior, weakening independence from the dew-point target.",
        capability_independence="minor",
    ),
    51: issue(
        "minor",
        "The first neighbor restates the same river-flood fact rather than testing a distinct related fact.",
        neighbor_locality="minor",
    ),
    56: issue(
        "minor",
        "Sugar also stores energy, so the negative construction is not an unconditional semantic opposite.",
        desired_contrast_validity="minor",
        construction_quality="minor",
    ),
    57: issue(
        "minor",
        "Cooking oil transfers heat in a pan but is not itself the heat source, and water can also heat food.",
        desired_contrast_validity="minor",
        target_consistency="minor",
    ),
    60: issue(
        "minor",
        "Wind is also a primary erosion agent, so the binary target and matched negatives are too exclusive; several capabilities are water-related.",
        desired_contrast_validity="minor",
        target_consistency="minor",
        capability_independence="minor",
        construction_quality="minor",
    ),
    69: issue(
        "minor",
        "The neighbor assigning general support and protection to the integumentary system is underspecified relative to the skeletal system.",
        neighbor_locality="minor",
    ),
    74: issue(
        "minor",
        "The main statistic follows the source, but two neighboring 2019 global-access statistics require independent source verification.",
        neighbor_locality="minor",
    ),
    77: issue(
        "minor",
        "The 13C gyromagnetic-ratio neighbor uses approximately 67 MHz/T, conflating angular-frequency and frequency units.",
        neighbor_locality="minor",
    ),
    78: issue(
        "minor",
        "The main statistic follows the source, while the neighboring survey percentages are time- and source-sensitive.",
        neighbor_locality="minor",
    ),
    82: issue(
        "minor",
        "Private/public-key neighbors assume asymmetric encryption although the source statement is encryption-scheme agnostic.",
        neighbor_locality="minor",
    ),
    83: issue(
        "minor",
        "The target follows the source, but the three unrelated 2018 survey percentages are not grounded by the record.",
        neighbor_locality="minor",
    ),
    86: issue(
        "minor",
        "At least one neighboring yearly-average claim depends on quiz-count weighting and is not established by the compact statement alone.",
        neighbor_locality="minor",
    ),
    92: issue(
        "minor",
        "The source item is a corrupted missing-expression geometry prompt; the angle-bisector fact is valid but weakly tied to the requested answer.",
        source_grounding="minor",
    ),
    100: issue(
        "minor",
        "The octagon fact is true and source-related, but it replaces the source question's requested piece count with a background property.",
        source_grounding="minor",
    ),
    101: issue(
        "minor",
        "The target matches the source answer, while the three derived logic-puzzle neighbors should be retained only with solver-backed verification.",
        neighbor_locality="minor",
    ),
    38: issue(
        "fail",
        "The advice to drink sparingly in a hot arid environment is unsafe and semantically misleading as a general survival rule.",
        desired_contrast_validity="fail",
        target_consistency="fail",
    ),
    50: issue(
        "fail",
        "Oxygen-rich blood does not universally flow away from the heart; pulmonary veins carry oxygenated blood toward it.",
        desired_contrast_validity="fail",
        target_consistency="fail",
        construction_quality="minor",
    ),
    52: issue(
        "fail",
        "Crevasses are not generally an erosion product, and sediment is itself commonly produced or transported by erosion, breaking the polarity.",
        desired_contrast_validity="fail",
        target_consistency="fail",
        construction_quality="fail",
    ),
    55: issue(
        "fail",
        "The statement that prolonged contact specifically smooths shale is too context-free to define a valid factual target.",
        desired_contrast_validity="fail",
        target_consistency="fail",
    ),
    73: issue(
        "fail",
        "The description conflates forward stagewise and forward stepwise selection; the stated feature-addition criterion is not uniquely diagnostic.",
        desired_contrast_validity="fail",
        target_consistency="fail",
    ),
    76: issue(
        "fail",
        "One neighbor is a direct inversion of the target relation and another introduces an unsupported high-income-country comparison.",
        neighbor_locality="fail",
    ),
    81: issue(
        "fail",
        "The NMR target reverses the usual distinction: T1 is strongly governed by spectral density near the Larmor frequency, while T2 also includes low-frequency dephasing.",
        desired_contrast_validity="fail",
        target_consistency="fail",
    ),
    93: issue(
        "fail",
        "Two probability neighbors are numerically wrong: hitting cumulative totals 4 and 2 is not 7/27 and 1/6, respectively.",
        neighbor_locality="fail",
    ),
    96: issue(
        "fail",
        "The target contrast is the alternate standard characteristic-polynomial sign convention, and the determinant neighbor has the wrong sign.",
        desired_contrast_validity="fail",
        neighbor_locality="fail",
    ),
    107: issue(
        "fail",
        "The hexagon neighbor reports 16 pieces despite the same three-parallel-line constraint, whose planar maximum is 13.",
        neighbor_locality="fail",
    ),
    108: issue(
        "fail",
        "The source is a cube-cutting problem with answer 0, but the record, constructions, and probes concern the Parthenon Marbles.",
        source_grounding="fail",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the frozen semantic audit.")
    parser.add_argument("--sample-path", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]

    rows = []
    for sample in samples:
        audit_index = int(sample["audit_index"])
        judgment = ISSUES.get(
            audit_index,
            issue("pass", "No material semantic issue identified in the audited fields."),
        )
        dimension_severity = max(SEVERITY[judgment[name]] for name in DIMENSIONS)
        if dimension_severity != SEVERITY[judgment["overall"]]:
            raise ValueError(f"Overall status mismatch for audit index {audit_index}")
        rows.append(
            {
                "audit_index": audit_index,
                "dataset": sample["dataset"],
                "record_id": sample["record_id"],
                **judgment,
            }
        )

    frame = pd.DataFrame(rows).sort_values("audit_index")
    judgment_path = output_dir / "semantic_audit_judgments.csv"
    frame.to_csv(judgment_path, index=False)

    overall = frame["overall"].value_counts().reindex(["pass", "minor", "fail"], fill_value=0)
    by_dataset = (
        frame.groupby(["dataset", "overall"]).size().unstack(fill_value=0).reindex(
            columns=["pass", "minor", "fail"], fill_value=0
        )
    )
    by_dataset["n"] = by_dataset.sum(axis=1)
    by_dataset["strict_pass_rate"] = by_dataset["pass"] / by_dataset["n"]
    by_dataset["acceptable_rate"] = (by_dataset["pass"] + by_dataset["minor"]) / by_dataset["n"]
    by_dataset_path = output_dir / "semantic_audit_by_dataset.csv"
    by_dataset.reset_index().to_csv(by_dataset_path, index=False)

    dimension_summary = {}
    for dimension in DIMENSIONS:
        counts = frame[dimension].value_counts().reindex(["pass", "minor", "fail"], fill_value=0)
        dimension_summary[dimension] = {key: int(value) for key, value in counts.items()}

    summary = {
        "sample_path": str(sample_path.resolve()),
        "judgment_path": str(judgment_path.resolve()),
        "by_dataset_path": str(by_dataset_path.resolve()),
        "n_records": int(len(frame)),
        "n_target_prompts": int(len(frame) * 3),
        "n_neighbor_prompts": int(len(frame) * 3),
        "n_capability_prompts": int(len(frame) * 4),
        "n_positive_constructions": int(sum(len(row["train_positive"]) for row in samples)),
        "n_negative_constructions": int(sum(len(row["train_negative"]) for row in samples)),
        "overall": {key: int(value) for key, value in overall.items()},
        "strict_pass_rate": float(overall["pass"] / len(frame)),
        "acceptable_rate": float((overall["pass"] + overall["minor"]) / len(frame)),
        "dimension_summary": dimension_summary,
    }
    summary_path = output_dir / "semantic_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
