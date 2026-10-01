# PML claim–evidence ledger

Updated for ICLR: 2026-09-23.

Partial `raw_scores.jsonl` files are execution progress only and cannot support
a paper claim. This ledger records the supplied frozen evidence, not an
independent replication of the experiments.

| Claim | Frozen evidence | Scope |
| --- | --- | --- |
| 3,000 primary records, 30,000 distinct paths, 210,000 distinct path-strength evaluations | experiment_scale.json; dataset_composition.csv; appendix counting explanation | 36,000 logged paths and 252,000 logged evaluations include a duplicate random entry |
| Learned directions improve primary target and clean incidence | frozen_outcome_summary.csv; paired_bootstrap_vs_random.csv | Two selected blocks; no instance-level locality guarantee |
| Layer-7 RFM/AGOP gains 3.6 Target and 3.4 Clean percentage points over random | paired_bootstrap_vs_random.csv; main Table 1 | Paired record bootstrap; not AUROC differences |
| Strength-disjoint low-dose response is the main predictive contribution | prediction_split_macro.csv; cross_model_prediction_heatmap.csv | Primary all-method predictions retain duplicate-random weighting; not a deduplicated rerun |
| Selector improves utility over a train-selected fixed policy | strength_selector_summary.csv; selector_paired_bootstrap.csv | 100 held-out records; declared utility and shared-cost exclusions |
| Width-corrected residual-norm confirmation across three models. | cross_model_replication_summary.csv | 24-path scientific gate: complete, 3 models × 2 blocks × 4 methods; completion does not certify every scientific claim |
| Free-generation endpoint behavior is boundary evidence only. | endpoint_validation_summary.csv | Supplement only; no reliable wrong-to-right control claim |
| Activation controllability is not persistent editability | Supplied ROME transfer analyses | Separate protocol; not pooled with primary evidence |

## Remaining checks

- The frozen primary all-method prediction artifacts still contain 36,000
  logged rows. Record-grouped folds keep duplicate entries together but do not
  remove their training/evaluation weight. No completed replacement is claimed.
- Author confirmation of the AI-use inventory and human-audit process is pending.
- Current ICLR format checks are recorded in `iclr_pdf_checks.json` and
  `VISUAL_QA_LEDGER.md`; copied AAAI checklist notes are historical only.
