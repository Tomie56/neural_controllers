# Frozen PML Fresh Multidomain 3000 v1

This directory is the frozen dataset for the next PML bidirectional steering experiments.

Do not edit files in this directory in-place. If the dataset changes, create a new frozen directory with a new version suffix.

## Files

- `records.jsonl`: 3000 intervention records. These contain training positive/negative statements and no eval fields.
- `eval_bank.jsonl`: global contrastive eval bank with `prompt`, `correct`, and `contrast`.
- `eval_assignments.jsonl`: record-to-eval mapping. Target and neighbor assignments are record-specific; capability assignments are globally rebalanced.
- `validation.json`: schema, reference, distribution, and capability usage checks.
- `manifest.json`: source file paths and generation settings.
- `sample_report.jsonl`: human QA sample by dataset.

## Capability Rebalancing

Capability probes are assigned by uniform round-robin over the global capability pool.

- records: 3000
- capability pool size: 2834
- capability assignments: 12000
- mean usage: 4.234297812279464
- min usage: 4
- max usage: 5

## Readiness

`is_freeze_ready`: `True`

## Source

Source prefix: `/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000_v1`

Created by:

```bash
PYTHONPATH=/data/neural_controllers/pml/src /data/miniconda3/envs/rfm/bin/python -m predictive_memory_localization.data.rebalance_capability_assignments --src-prefix /data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000_v1 --output-dir /data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen --n-capability-per-record 4 --seed 113 --overwrite
```
