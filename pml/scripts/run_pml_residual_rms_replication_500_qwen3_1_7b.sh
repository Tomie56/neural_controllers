#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
export MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"
export MODEL_LOADER="${MODEL_LOADER:-causal_lm}"
export BATCH_SIZE="${BATCH_SIZE:-4}"

exec /data/neural_controllers/pml/scripts/run_pml_residual_rms_replication_500_model.sh "$@"
