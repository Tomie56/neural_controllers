#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b/selector_dataset}"
VALIDATION_DATASET_DIR="${VALIDATION_DATASET_DIR:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2b_selector_validation_100/qwen3_1_7b/selector_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_strength_selector_models/qwen3_1_7b}"
FEATURE_SETS="${FEATURE_SETS:-L LB LBA LBAR}"
MODELS="${MODELS:-shared_logistic shared_gbdt per_layer_logistic per_layer_gbdt}"
OVERWRITE="${OVERWRITE:-1}"

ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.train_strength_selector \
  --train-dataset-dir "${TRAIN_DATASET_DIR}" \
  --validation-dataset-dir "${VALIDATION_DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --feature-sets ${FEATURE_SETS} \
  --models ${MODELS} \
  "${ARGS[@]}"

echo "Selector model output: ${OUTPUT_DIR}"
