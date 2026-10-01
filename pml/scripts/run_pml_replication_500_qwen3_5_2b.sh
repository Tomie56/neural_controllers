#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/data/neural_controllers/model/Qwen3.5-2B-Base}"
export MODEL_NAME="${MODEL_NAME:-qwen3_5_2b_base}"
export MODEL_LOADER="${MODEL_LOADER:-image_text_to_text}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500/qwen3_5_2b_base}"

exec /data/neural_controllers/pml/scripts/run_pml_compact_replication_500_model.sh "$@"
