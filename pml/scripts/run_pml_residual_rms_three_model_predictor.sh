#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers

ROOT="${ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500_residual_rms_v2}"
RUNNER="${RUNNER:-/data/neural_controllers/pml/scripts/run_activation_space_pml_strict_later_prediction_qwen3_1_7b.sh}"

MODELS="${PREDICTOR_MODELS:-qwen3_1_7b qwen3_5_2b_base ministral_3_3b_base_2512}"
PREDICTION_MODELS="${PREDICTION_MODELS:-logistic random_forest}"
SPLITS="${SPLITS:-record dataset domain}"
COHORTS="${COHORTS:-all learned rfm}"
TARGETS="${TARGETS:-later_suppression_path later_enhancement_path later_target_any_path later_neighbor_damage_any_path later_capability_damage_any_path later_clean_suppression_path later_clean_enhancement_path later_clean_any_path}"
N_SPLITS="${N_SPLITS:-5}"
MIN_POSITIVE="${MIN_POSITIVE:-20}"
SEED="${SEED:-113}"
OVERWRITE="${OVERWRITE:-0}"

for MODEL in ${MODELS}; do
  MODEL_ROOT="${ROOT}/${MODEL}"
  DATASET_DIR="${MODEL_ROOT}/prediction_dataset_later_strength"

  if [[ ! -s "${DATASET_DIR}/pml_path_prediction_dataset.csv" ]]; then
    echo "ERROR: missing strict-later dataset for ${MODEL}: ${DATASET_DIR}" >&2
    exit 2
  fi

  echo "[three-model predictor] starting model=${MODEL}"
  MAIN_ROOT="${MODEL_ROOT}" \
  STAGE_ROOT="${MODEL_ROOT}/outcomes" \
  SOURCE_DATASET_DIR="${MODEL_ROOT}/prediction_dataset" \
  DATASET_DIR="${DATASET_DIR}" \
  OUTPUT_DIR="${MODEL_ROOT}/prediction_analysis_strict_later" \
  BUILD_DATASET=0 \
  OVERWRITE="${OVERWRITE}" \
  MODELS="${PREDICTION_MODELS}" \
  SPLITS="${SPLITS}" \
  COHORTS="${COHORTS}" \
  TARGETS="${TARGETS}" \
  N_SPLITS="${N_SPLITS}" \
  MIN_POSITIVE="${MIN_POSITIVE}" \
  SEED="${SEED}" \
  bash "${RUNNER}"
  echo "[three-model predictor] completed model=${MODEL}"
done

echo "[three-model predictor] complete"
