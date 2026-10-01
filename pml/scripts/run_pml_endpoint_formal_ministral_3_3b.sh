#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/data/neural_controllers/model/Ministral-3-3B-Base-2512}"
export MODEL_NAME="${MODEL_NAME:-ministral_3_3b_base_2512}"
export MODEL_LOADER="${MODEL_LOADER:-image_text_to_text}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export PATH_OUTCOME_CSV="${PATH_OUTCOME_CSV:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500/ministral_3_3b_base_2512/prediction_dataset/pml_path_prediction_dataset.csv}"
export BASE_ROOT="${BASE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/endpoint_validation_v3/ministral_3_3b_base_2512}"

exec /data/neural_controllers/pml/scripts/run_pml_endpoint_validation_model.sh formal "$@"
