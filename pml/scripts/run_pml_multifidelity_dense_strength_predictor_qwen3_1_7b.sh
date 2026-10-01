#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000}"
MAIN_ROOT="${MAIN_ROOT:-${RESULTS_ROOT}/stage_activation_pml/main_3000/qwen3_1_7b}"
DENSE_TRAIN_ROOT="${DENSE_TRAIN_ROOT:-${RESULTS_ROOT}/stage2_bidirectional_pilot_500/qwen3_1_7b}"
DENSE_VALIDATION_ROOT="${DENSE_VALIDATION_ROOT:-${RESULTS_ROOT}/stage2b_selector_validation_100/qwen3_1_7b}"

TAU="${TAU:-0.15258546272913603}"
DENSE_TRAIN_DATASET_DIR="${DENSE_TRAIN_DATASET_DIR:-${DENSE_TRAIN_ROOT}/selector_dataset_tau_main_q95}"
DENSE_VALIDATION_DATASET_DIR="${DENSE_VALIDATION_DATASET_DIR:-${DENSE_VALIDATION_ROOT}/selector_dataset_tau_main_q95}"
SPARSE_AUX_DATASET_DIR="${SPARSE_AUX_DATASET_DIR:-${MAIN_ROOT}/strength_selector_sparse_dataset_tau_main_q95}"
MULTIFIDELITY_DATASET_DIR="${MULTIFIDELITY_DATASET_DIR:-${MAIN_ROOT}/strength_selector_multifidelity_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${MAIN_ROOT}/strength_predictor_multifidelity_dense}"

SPARSE_METHODS="${SPARSE_METHODS:-mean_difference logistic random}"
SPARSE_LAYERS="${SPARSE_LAYERS:--21 -17}"
FEATURE_SETS="${FEATURE_SETS:-BMA MAR BMAR BMLAR}"
MODELS="${MODELS:-linear hist_gbdt}"
DENSE_WEIGHT="${DENSE_WEIGHT:-1.0}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.5}"
LINEAR_N_JOBS="${LINEAR_N_JOBS:-8}"
OVERWRITE="${OVERWRITE:-0}"

trap 'echo; echo "Interrupted. Rerun the same command to resume completed predictor tasks." >&2' INT TERM

build_selector_dataset() {
  local stage_root="$1"
  local output_dir="$2"
  shift 2
  if [[ "${OVERWRITE}" != "1" && -s "${output_dir}/alpha_level_selector_dataset.csv" ]]; then
    echo "[strength] reuse selector dataset: ${output_dir}"
    return
  fi
  echo "[strength] build selector dataset: ${output_dir}"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_strength_selector_dataset \
    --stage-root "${stage_root}" \
    --output-dir "${output_dir}" \
    --tau "${TAU}" \
    --overwrite \
    "$@"
}

build_selector_dataset "${DENSE_TRAIN_ROOT}" "${DENSE_TRAIN_DATASET_DIR}"
build_selector_dataset "${DENSE_VALIDATION_ROOT}" "${DENSE_VALIDATION_DATASET_DIR}"
build_selector_dataset "${MAIN_ROOT}/outcomes" "${SPARSE_AUX_DATASET_DIR}" --early-alphas 0.1

if [[ "${OVERWRITE}" == "1" || ! -s "${MULTIFIDELITY_DATASET_DIR}/alpha_level_selector_dataset.csv" ]]; then
  MULTIFIDELITY_ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    MULTIFIDELITY_ARGS+=(--overwrite)
  fi
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_multifidelity_strength_dataset \
    --dense-train-dataset-dir "${DENSE_TRAIN_DATASET_DIR}" \
    --dense-validation-dataset-dir "${DENSE_VALIDATION_DATASET_DIR}" \
    --sparse-aux-dataset-dir "${SPARSE_AUX_DATASET_DIR}" \
    --output-dir "${MULTIFIDELITY_DATASET_DIR}" \
    --sparse-methods ${SPARSE_METHODS} \
    --sparse-layers ${SPARSE_LAYERS} \
    --dense-weight "${DENSE_WEIGHT}" \
    --sparse-weight "${SPARSE_WEIGHT}" \
    "${MULTIFIDELITY_ARGS[@]}"
else
  echo "[strength] reuse multifidelity dataset: ${MULTIFIDELITY_DATASET_DIR}"
fi

TRAIN_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.train_multifidelity_dense_strength_predictor \
  --dense-train-dataset-dir "${DENSE_TRAIN_DATASET_DIR}" \
  --multifidelity-train-dataset-dir "${MULTIFIDELITY_DATASET_DIR}" \
  --validation-dataset-dir "${DENSE_VALIDATION_DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --feature-sets ${FEATURE_SETS} \
  --models ${MODELS} \
  --linear-n-jobs "${LINEAR_N_JOBS}" \
  "${TRAIN_ARGS[@]}"

echo "[strength] complete"
echo "Report: ${OUTPUT_DIR}/STRENGTH_PREDICTOR_REPORT.md"
echo "Summary: ${OUTPUT_DIR}/strength_selector_summary.csv"
