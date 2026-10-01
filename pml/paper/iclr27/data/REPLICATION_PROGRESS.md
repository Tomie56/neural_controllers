# Compact Replication Execution Progress

Snapshot time: `2026-07-19T14:40:59+08:00`.

This file reports execution progress only. Partial raw-score rows are not
scientific results and are never used by the manuscript updater.

| Model | Record-method-layer units | Configs | Active config | Naive current ETA | Strict summary |
| --- | ---: | ---: | --- | ---: | --- |
| Qwen3.5-2B | 3,565/4,000 (89.1%) | 7/8 | rfm_agop_top1 / -15: 65/500 | 10h 17m | pending |
| Ministral-3-3B | 0/4,000 (0.0%) | 0/8 | -- | -- | pending |

## Configuration Detail

| Model | Method | Layer | Records | Status | Failures |
| --- | --- | ---: | ---: | --- | ---: |
| Qwen3.5-2B | random | -18 | 500/500 | complete | 0 |
| Qwen3.5-2B | random | -15 | 500/500 | complete | 0 |
| Qwen3.5-2B | mean_difference | -18 | 500/500 | complete | 0 |
| Qwen3.5-2B | mean_difference | -15 | 500/500 | complete | 0 |
| Qwen3.5-2B | logistic | -18 | 500/500 | complete | 0 |
| Qwen3.5-2B | logistic | -15 | 500/500 | complete | 0 |
| Qwen3.5-2B | rfm_agop_top1 | -18 | 500/500 | complete | 0 |
| Qwen3.5-2B | rfm_agop_top1 | -15 | 65/500 | running | 0 |
| Ministral-3-3B | random | -20 | 0/500 | pending | 0 |
| Ministral-3-3B | random | -16 | 0/500 | pending | 0 |
| Ministral-3-3B | mean_difference | -20 | 0/500 | pending | 0 |
| Ministral-3-3B | mean_difference | -16 | 0/500 | pending | 0 |
| Ministral-3-3B | logistic | -20 | 0/500 | pending | 0 |
| Ministral-3-3B | logistic | -16 | 0/500 | pending | 0 |
| Ministral-3-3B | rfm_agop_top1 | -20 | 0/500 | pending | 0 |
| Ministral-3-3B | rfm_agop_top1 | -16 | 0/500 | pending | 0 |

The ETA is a naive extrapolation from the current configuration's wall
time and should not be used as a paper result or a guarantee for later
methods, which may have different direction-fitting costs.
