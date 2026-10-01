#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/data/neural_controllers/model/Qwen3.5-2B-Base}"
export MODEL_NAME="${MODEL_NAME:-qwen3_5_2b_base}"
export MODEL_LOADER="${MODEL_LOADER:-image_text_to_text}"
export BATCH_SIZE="${BATCH_SIZE:-2}"

exec /data/neural_controllers/pml/scripts/run_pml_residual_rms_replication_500_model.sh "$@"
