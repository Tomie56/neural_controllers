#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MAIN_ROOT="${MAIN_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b}"
DATASET_DIR="${DATASET_DIR:-${MAIN_ROOT}/prediction_dataset_strict_later}"
OUTPUT_DIR="${OUTPUT_DIR:-${MAIN_ROOT}/prediction_analysis_strict_later_r_ablation}"

TARGETS="${TARGETS:-later_suppression_path later_enhancement_path later_target_any_path later_neighbor_damage_any_path later_capability_damage_any_path later_clean_suppression_path later_clean_enhancement_path later_clean_any_path}"
FEATURE_SETS="${FEATURE_SETS:-M+R B+M+R B+M+L+R}"
SPLITS="${SPLITS:-record dataset domain}"
COHORTS="${COHORTS:-all learned rfm}"
MODELS="${MODELS:-logistic random_forest}"
N_SPLITS="${N_SPLITS:-5}"
RF_N_JOBS="${RF_N_JOBS:-8}"
MIN_POSITIVE="${MIN_POSITIVE:-20}"
SEED="${SEED:-113}"
OVERWRITE="${OVERWRITE:-0}"

RUN_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  RUN_ARGS+=(--overwrite)
fi

echo "[strict-later R ablation] dataset=${DATASET_DIR}"
echo "[strict-later R ablation] output=${OUTPUT_DIR}"
echo "[strict-later R ablation] feature_sets=${FEATURE_SETS} rf_n_jobs=${RF_N_JOBS}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_activation_space_pml_main_prediction \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --targets ${TARGETS} \
  --feature-sets ${FEATURE_SETS} \
  --splits ${SPLITS} \
  --cohorts ${COHORTS} \
  --models ${MODELS} \
  --n-splits "${N_SPLITS}" \
  --rf-n-jobs "${RF_N_JOBS}" \
  --min-positive "${MIN_POSITIVE}" \
  --seed "${SEED}" \
  "${RUN_ARGS[@]}"

echo "[strict-later R ablation] complete"
echo "Report: ${OUTPUT_DIR}/PML_MAIN_PREDICTION_REPORT.md"
echo "Fold results: ${OUTPUT_DIR}/fold_results.jsonl"
