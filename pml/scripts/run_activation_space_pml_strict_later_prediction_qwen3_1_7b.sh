#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MAIN_ROOT="${MAIN_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b}"
STAGE_ROOT="${STAGE_ROOT:-${MAIN_ROOT}/outcomes}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-${MAIN_ROOT}/prediction_dataset}"
DATASET_DIR="${DATASET_DIR:-${MAIN_ROOT}/prediction_dataset_strict_later}"
OUTPUT_DIR="${OUTPUT_DIR:-${MAIN_ROOT}/prediction_analysis_strict_later}"

EARLY_ALPHA="${EARLY_ALPHA:-0.1}"
LATER_ALPHAS="${LATER_ALPHAS:-0.25 0.5}"

TARGETS="${TARGETS:-later_suppression_path later_enhancement_path later_target_any_path later_neighbor_damage_any_path later_capability_damage_any_path later_clean_suppression_path later_clean_enhancement_path later_clean_any_path suppression_success_medium enhancement_success_medium clean_suppression_medium clean_enhancement_medium}"
SPLITS="${SPLITS:-record dataset domain}"
COHORTS="${COHORTS:-all learned rfm}"
MODELS="${MODELS:-logistic random_forest}"
N_SPLITS="${N_SPLITS:-5}"
MIN_POSITIVE="${MIN_POSITIVE:-20}"
SEED="${SEED:-113}"
MAX_RECORDS="${MAX_RECORDS:-}"

BUILD_DATASET="${BUILD_DATASET:-auto}"
RUN_PREDICTION="${RUN_PREDICTION:-1}"
OVERWRITE="${OVERWRITE:-0}"

echo "[PML strict later] main_root=${MAIN_ROOT}"
echo "[PML strict later] stage_root=${STAGE_ROOT}"
echo "[PML strict later] source_dataset_dir=${SOURCE_DATASET_DIR}"
echo "[PML strict later] dataset_dir=${DATASET_DIR}"
echo "[PML strict later] output_dir=${OUTPUT_DIR}"
echo "[PML strict later] early_alpha=${EARLY_ALPHA} later_alphas=${LATER_ALPHAS}"
echo "[PML strict later] overwrite=${OVERWRITE}"

BUILD_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
fi

if [[ "${BUILD_DATASET}" == "1" ]] || \
   [[ "${BUILD_DATASET}" == "auto" && ! -s "${DATASET_DIR}/pml_path_prediction_dataset.csv" ]]; then
  echo "[PML strict later] Building leakage-free later-label dataset"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_strict_later_dataset \
    --source-dataset-dir "${SOURCE_DATASET_DIR}" \
    --stage-root "${STAGE_ROOT}" \
    --output-dir "${DATASET_DIR}" \
    --early-alpha "${EARLY_ALPHA}" \
    --later-alphas ${LATER_ALPHAS} \
    "${BUILD_ARGS[@]}"
else
  echo "[PML strict later] Reusing existing strict-later dataset"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_strict_later_dataset \
    --source-dataset-dir "${SOURCE_DATASET_DIR}" \
    --stage-root "${STAGE_ROOT}" \
    --output-dir "${DATASET_DIR}" \
    --early-alpha "${EARLY_ALPHA}" \
    --later-alphas ${LATER_ALPHAS}
fi

if [[ "${RUN_PREDICTION}" != "1" ]]; then
  echo "[PML strict later] Dataset build complete; prediction disabled"
  exit 0
fi

PREDICTION_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  PREDICTION_ARGS+=(--overwrite)
fi
if [[ -n "${MAX_RECORDS}" ]]; then
  PREDICTION_ARGS+=(--max-records "${MAX_RECORDS}")
fi

echo "[PML strict later] Running grouped prediction tasks with resume support"
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_activation_space_pml_main_prediction \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --targets ${TARGETS} \
  --splits ${SPLITS} \
  --cohorts ${COHORTS} \
  --models ${MODELS} \
  --n-splits "${N_SPLITS}" \
  --min-positive "${MIN_POSITIVE}" \
  --seed "${SEED}" \
  "${PREDICTION_ARGS[@]}"

echo "[PML strict later] complete"
echo "Dataset: ${DATASET_DIR}/pml_path_prediction_dataset.csv"
echo "Manifest: ${DATASET_DIR}/prediction_dataset_manifest.json"
echo "Report: ${OUTPUT_DIR}/PML_MAIN_PREDICTION_REPORT.md"
echo "Fold results: ${OUTPUT_DIR}/fold_results.jsonl"
