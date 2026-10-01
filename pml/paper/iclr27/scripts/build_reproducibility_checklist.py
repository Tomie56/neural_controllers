#!/usr/bin/env python3
"""Generate the PML AAAI reproducibility checklist from the official template."""

from __future__ import annotations

import json
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PAPER_ROOT / "official/AuthorKit27/ReproducibilityChecklist.tex"
OUTPUT = PAPER_ROOT / "ReproducibilityChecklist_PML.tex"
NOTES = PAPER_ROOT / "data/REPRODUCIBILITY_CHECKLIST_NOTES.md"
ENVIRONMENT = PAPER_ROOT / "data/computational_environment.json"
ENVIRONMENT_TEX = PAPER_ROOT / "sections/computational_environment_generated.tex"

ANSWERS = [
    "yes",
    "yes",
    "yes",
    "no",
    "no",
    "no",
    "no",
    "no",
    "no",
    "NA",
    "NA",
    "yes",
    "yes",
    "partial",
    "partial",
    "yes",
    "partial",
    "NA",
    "yes",
    "partial",
    "no",
    "no",
    "partial",
    "partial",
    "yes",
    "yes",
    "partial",
    "yes",
    "yes",
    "partial",
    "partial",
]

RATIONALES = [
    "The pipeline figure and formal PML setup give the conceptual outline.",
    "Results, interpretations, and boundary evidence are separated explicitly.",
    "Related Work provides background references for localization, steering, and editing.",
    "The contribution is empirical and methodological rather than theoretical.",
    "Not applicable because the paper makes no theoretical contribution.",
    "Not applicable because the paper makes no theorem-level claims.",
    "Not applicable because no proofs are claimed.",
    "Not applicable because no theoretical derivation requires a proof sketch.",
    "Not applicable to a non-theoretical contribution.",
    "No theoretical claims are asserted.",
    "No theoretical claim-elimination code is applicable.",
    "The frozen benchmark is assembled from multiple public datasets.",
    "The manuscript motivates multidomain coverage and separate target/control probes.",
    "Dataset composition and construction are documented, but the full record artifact is not embedded in the paper.",
    "A public release is intended, but source-dataset licensing and redistribution constraints still require final review.",
    "Existing source datasets are cited in the manuscript and dataset documentation.",
    "The constituent sources are public, but access and redistribution conditions vary.",
    "No non-public source dataset is currently claimed.",
    "The paper reports GPU intervention sweeps and CPU prediction analyses.",
    "Final grids and pilot-selected layers are documented, but every development range is not exhaustively listed.",
    "Preprocessing code exists in the repository rather than as literal code in the supplementary document.",
    "Experiment code exists in the repository rather than as a code supplement.",
    "A release is intended, but the final public license and packaging are pending.",
    "Implementation code is structured and documented, but not every new-method block links back to a paper section.",
    "Seed 113 and grouped split behavior are stated and stored in generated configurations.",
    "The supplementary document records OS, CPU, RAM, GPU, driver, CUDA toolkit, Python, and principal library versions.",
    "Outcome and prediction metrics are defined, but the motivation for every secondary metric is not equally detailed.",
    "The manuscript identifies the frozen method-layer runs, prediction folds, and bootstrap resamples.",
    "Paired intervals, grouped-fold variation, and boundary analyses supplement averages.",
    "Record-paired bootstrap intervals support the main comparisons, but not every descriptive replication delta has a formal test.",
    "Core intervention and selector settings are listed; complete library/model defaults remain in configuration artifacts.",
]


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    marker = "Type your response here"
    question_start = "% The questions start here"
    if question_start not in text:
        raise ValueError("Official checklist is missing the question-start marker")
    prefix, questions = text.split(question_start, 1)
    count = questions.count(marker)
    if count != len(ANSWERS):
        raise ValueError(f"Checklist question count changed: {count} != {len(ANSWERS)}")
    for answer in ANSWERS:
        questions = questions.replace(marker, answer, 1)
    text = prefix + question_start + questions
    OUTPUT.write_text(text, encoding="utf-8")

    environment = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    environment_text = (
        f"Experiments and paper-facing analyses run on {environment['operating_system']} "
        f"with an {environment['cpu']} CPU ({environment['cpu_cores']} allocated cores), "
        f"{environment['ram_gib']} GiB RAM, and one {environment['gpu']} GPU "
        f"(driver {environment['nvidia_driver']}; CUDA toolkit "
        f"{environment['cuda_toolkit']}). The software environment uses Python "
        f"{environment['python']}, PyTorch {environment['pytorch']}, Transformers "
        f"{environment['transformers']}, scikit-learn {environment['scikit_learn']}, "
        f"pandas {environment['pandas']}, NumPy {environment['numpy']}, and SciPy "
        f"{environment['scipy']}.\n"
    )
    ENVIRONMENT_TEX.write_text(environment_text, encoding="utf-8")

    lines = [
        "# PML Reproducibility Checklist Notes",
        "",
        "Generated from the official AAAI-27 checklist template. The response file",
        "is kept separate from `main.tex` until the live submission policy specifies",
        "whether it should be included or uploaded independently.",
        "",
        "| Item | Answer | Evidence/rationale |",
        "| ---: | --- | --- |",
    ]
    for index, (answer, rationale) in enumerate(zip(ANSWERS, RATIONALES), 1):
        lines.append(f"| {index} | {answer} | {rationale} |")
    lines.extend(
        [
            "",
            "## Remaining Artifact Actions",
            "",
            "- Export a lock file or environment specification matching the recorded versions.",
            "- Finalize dataset/model/code license and redistribution notes.",
            "- Replace repository-internal model paths with public model identifiers or environment variables.",
            "- Revisit partial/no answers after the public artifact package is frozen.",
            "",
        ]
    )
    NOTES.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {NOTES}")
    print(f"Wrote {ENVIRONMENT_TEX}")


if __name__ == "__main__":
    main()
