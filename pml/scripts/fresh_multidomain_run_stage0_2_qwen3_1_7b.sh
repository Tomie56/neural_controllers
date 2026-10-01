#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000}"
MODEL_SLUG="${MODEL_SLUG:-qwen3_1_7b}"
METHODS="${METHODS:-mean_difference logistic random}"
STAGE1_LAYERS="${STAGE1_LAYERS:--1 -9 -17}"
STAGE2_LAYERS="${STAGE2_LAYERS:--1 -5 -9 -13 -17 -21}"
STAGE1_RECORDS="${STAGE1_RECORDS:-50}"
STAGE2_RECORDS="${STAGE2_RECORDS:-500}"

RUN_STAGE0="${RUN_STAGE0:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"

if [[ "${RUN_STAGE0}" == "1" ]]; then
  echo "[Stage 0] Margin sanity (${MODEL_SLUG})"
  ./pml/scripts/fresh_multidomain_stage0_margin_sanity_qwen3_1_7b.sh
fi

if [[ "${RUN_STAGE1}" == "1" ]]; then
  echo "[Stage 1] Normalized evaluator smoke (${MODEL_SLUG})"
  METHODS="${METHODS}" LAYERS="${STAGE1_LAYERS}" MAX_RECORDS="${STAGE1_RECORDS}" ./pml/scripts/fresh_multidomain_stage1_smoke_qwen3_1_7b.sh
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  echo "[Stage 2] Bidirectional pilot (${MODEL_SLUG})"
  METHODS="${METHODS}" LAYERS="${STAGE2_LAYERS}" MAX_RECORDS="${STAGE2_RECORDS}" ./pml/scripts/fresh_multidomain_stage2_pilot_qwen3_1_7b.sh
fi

echo "[Validate] Checking Stage 0-2 outputs"
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.validate_fresh_multidomain_stage_outputs \
  --results-root "${RESULTS_ROOT}" \
  --model-slug "${MODEL_SLUG}" \
  --methods ${METHODS} \
  --stage1-layers ${STAGE1_LAYERS} \
  --stage2-layers ${STAGE2_LAYERS} \
  --stage1-records "${STAGE1_RECORDS}" \
  --stage2-records "${STAGE2_RECORDS}" \
  --output-json "${RESULTS_ROOT}/stage0_2_validation.json"

echo "[Report] Building Stage 0-2 report"
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_fresh_multidomain_stage_report \
  --results-root "${RESULTS_ROOT}" \
  --model-slug "${MODEL_SLUG}" \
  --output-md "${RESULTS_ROOT}/FRESH_MULTIDOMAIN_STAGE0_2_REPORT.md"

echo "Report: ${RESULTS_ROOT}/FRESH_MULTIDOMAIN_STAGE0_2_REPORT.md"
