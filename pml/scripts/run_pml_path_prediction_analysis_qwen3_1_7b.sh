#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
STAGE_ROOT="${STAGE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b}"
SELECTOR_DATASET_DIR="${SELECTOR_DATASET_DIR:-${STAGE_ROOT}/selector_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGE_ROOT}/pml_path_prediction}"
OVERWRITE="${OVERWRITE:-1}"
BUILD_SELECTOR_DATASET="${BUILD_SELECTOR_DATASET:-auto}"

echo "[PML path prediction] STAGE_ROOT=${STAGE_ROOT}"
echo "[PML path prediction] SELECTOR_DATASET_DIR=${SELECTOR_DATASET_DIR}"
echo "[PML path prediction] OUTPUT_DIR=${OUTPUT_DIR}"

if [[ "${BUILD_SELECTOR_DATASET}" == "1" ]] || [[ "${BUILD_SELECTOR_DATASET}" == "auto" && ! -s "${SELECTOR_DATASET_DIR}/path_level_selector_dataset.csv" ]]; then
  echo "[PML path prediction] Building selector/path dataset from Stage 2 dense curves"
  BUILD_ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    BUILD_ARGS+=(--overwrite)
  fi
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_strength_selector_dataset \
    --stage-root "${STAGE_ROOT}" \
    --output-dir "${SELECTOR_DATASET_DIR}" \
    --tau-source global_random_q95 \
    "${BUILD_ARGS[@]}"
fi

RUN_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  RUN_ARGS+=(--overwrite)
fi

echo "[PML path prediction] Running path outcome prediction and ablation"
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_pml_path_prediction_analysis \
  --selector-dataset-dir "${SELECTOR_DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 113 \
  --test-size 0.2 \
  "${RUN_ARGS[@]}"

echo "[PML path prediction] Done."
echo "Report: ${OUTPUT_DIR}/PML_PATH_PREDICTION_REPORT.md"
