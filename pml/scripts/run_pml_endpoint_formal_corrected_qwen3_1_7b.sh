#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
export MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"
export MODEL_LOADER="${MODEL_LOADER:-causal_lm}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LAYERS="${LAYERS:--21 -17}"
export PATH_OUTCOME_CSV="${PATH_OUTCOME_CSV:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b/prediction_dataset/pml_path_prediction_dataset.csv}"
export BASE_ROOT="${BASE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/endpoint_validation_v3/qwen3_1_7b}"

exec /data/neural_controllers/pml/scripts/run_pml_endpoint_validation_model.sh formal "$@"
