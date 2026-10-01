# PML ICLR 2027 frozen result snapshot

This directory contains the compact result files needed to audit and
reproduce the paper-facing ICLR assets. It is a selected snapshot of the
local experiment tree, not the complete 9.5 GB working directory.

Included files cover:

- the primary Qwen3-1.7B strict-later path dataset, prediction summaries and
  fold-level predictions;
- the strength-selector validation table, selected decisions, task metrics,
  and paired policy comparisons;
- endpoint-validation per-evaluation rows and summaries; and
- the corresponding path datasets, summaries, and fold predictions for the
  three 500-record residual-norm-matched model confirmations.

The exact source locations and file hashes are recorded in
`pml/paper/iclr27/data/raw_results_manifest.json`. The paper-facing audit can
be run with:

```bash
cd pml/paper/iclr27
/data/miniconda3/envs/rfm/bin/python scripts/audit_frozen_results_assets.py
```

Model checkpoints, token-level activation dumps, raw generation logs, and
large training matrices are intentionally excluded. The included path-level
tables contain the derived intervention outcomes and predictive features used
by the manuscript; the original benchmark records and evaluation assignments
are in `pml/data/pml_fresh_multidomain_3000_v1_frozen/`.
