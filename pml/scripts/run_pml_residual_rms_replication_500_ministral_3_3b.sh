#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/data/neural_controllers/model/Ministral-3-3B-Base-2512}"
export MODEL_NAME="${MODEL_NAME:-ministral_3_3b_base_2512}"
export MODEL_LOADER="${MODEL_LOADER:-image_text_to_text}"
export BATCH_SIZE="${BATCH_SIZE:-1}"

exec /data/neural_controllers/pml/scripts/run_pml_residual_rms_replication_500_model.sh "$@"
