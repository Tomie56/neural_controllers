# Predictive Memory Localization

This directory is the cleaned project root for the PML experiments.

## Layout

```text
pml/
  src/predictive_memory_localization/   Python package: data, methods, analysis
  data/                                 Required PML datasets and task JSONL files
  results/                              Local result artifacts (not in release commits)
  docs/                                 Research notes, paper matrices, and archived drafts
  scripts/                              Top-level experiment/report entrypoints
```

New work should use the `pml/scripts/` entrypoints below. The repository
root no longer keeps duplicate `data/`, `results/`, or source symlinks.

## Main Commands

Run the next main experiment for minimum effective steering strength, path
quality, side effects, and capability damage:

```bash
./pml/scripts/run_pml_min_strength_main.sh
```

Rebuild all report/table artifacts from existing results:

```bash
./pml/scripts/rebuild_reports.sh
```

Check the representative capability-onset probe status:

```bash
PYTHON_BIN=/data/miniconda3/envs/rfm/bin/python \
  PYTHONPATH=/data/neural_controllers/pml/src \
  python -m predictive_memory_localization.analysis.check_path_representative_probe_status
```

Rerun or extend capability-onset scoring if new model scoring is needed:

```bash
./pml/scripts/run_path_representative_capability_probe.sh
```

## Main Artifacts

- `pml/docs/PML_GEOMETRY_AWARE_IDEA_AND_EXPERIMENT_PLAN_中文.md`
- `pml/docs/PML_NEXT_EXPERIMENT_PLAN_中文.md`
- `pml/docs/PML_PAPER_RESULTS_MATRIX_中文.md`
- `pml/results/final_summary/FINAL_RESULTS_SUMMARY.md`
- `pml/results/paper_tables/PML_PAPER_TABLES.md`
- `pml/results/path_representative_50_capability_onset_probe/`

## Current Scope

The active line is not a generic RFM reproduction. It is:

> Predictive Memory Localization: testing when internal memory/localization
> signals predict intervention strength, side effects, capability damage, and
> editing fragility.

## Public release contents

The outer repository contains the current experiment source under `pml/src/`,
the shell entrypoints under `pml/scripts/`, the ICLR 2027 manuscript source
and deliverable PDFs under `pml/paper/iclr27/`, and the frozen record/task
inputs under `pml/data/`. The paper data directory also contains the compact
tables, environment manifest, and downloaded citation BibTeX sources needed
to audit the submitted manuscript.

Large generated result trees, model checkpoints, compiler caches, local logs,
and internal review reports remain outside the release commit. The source
code in `pml/src/` is the active implementation used for the ICLR workspace;
`pml/codebase/` is a separate historical/public code repository and is not
embedded as a nested Git repository here.
