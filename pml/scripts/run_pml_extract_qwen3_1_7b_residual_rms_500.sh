#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b/outcomes}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500_residual_rms_v2/qwen3_1_7b}"
OUTCOME_ROOT="${OUTCOME_ROOT:-${EXPERIMENT_ROOT}/outcomes}"
RECORD_IDS_FILE="${RECORD_IDS_FILE:-/data/neural_controllers/pml/data/pml_replication_500_stratified_seed113/record_ids.txt}"
PREDICTION_DATASET_DIR="${PREDICTION_DATASET_DIR:-${EXPERIMENT_ROOT}/prediction_dataset}"
LATER_DATASET_DIR="${LATER_DATASET_DIR:-${EXPERIMENT_ROOT}/prediction_dataset_later_strength}"
SUMMARY_DIR="${SUMMARY_DIR:-${EXPERIMENT_ROOT}/outcome_analysis_later_strength}"
METHODS="${METHODS:-random mean_difference logistic rfm_agop_top1}"
LAYERS="${LAYERS:--21 -17}"
OVERWRITE="${OVERWRITE:-0}"

EXTRACT_ARGS=()
BUILD_ARGS=()
SUMMARY_ARGS=(--overwrite)
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRACT_ARGS+=(--overwrite)
  BUILD_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.extract_qwen3_reference_replication_subset \
  --source-outcome-root "${SOURCE_ROOT}" \
  --output-outcome-root "${OUTCOME_ROOT}" \
  --record-ids-file "${RECORD_IDS_FILE}" \
  --methods ${METHODS} \
  --layers ${LAYERS} \
  "${EXTRACT_ARGS[@]}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset \
  --stage-root "${OUTCOME_ROOT}" \
  --output-dir "${PREDICTION_DATASET_DIR}" \
  --random-method random \
  --null-quantile 0.95 \
  --early-alpha 0.1 \
  --medium-alpha 0.5 \
  "${BUILD_ARGS[@]}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_strict_later_dataset \
  --source-dataset-dir "${PREDICTION_DATASET_DIR}" \
  --stage-root "${OUTCOME_ROOT}" \
  --output-dir "${LATER_DATASET_DIR}" \
  --early-alpha 0.1 \
  --later-alphas 0.25 0.5 \
  "${BUILD_ARGS[@]}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.summarize_strict_later_outcomes \
  --dataset-csv "${LATER_DATASET_DIR}/pml_path_prediction_dataset.csv" \
  --output-dir "${SUMMARY_DIR}" \
  "${SUMMARY_ARGS[@]}"

echo "[reference extract] complete: ${EXPERIMENT_ROOT}"
