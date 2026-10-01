#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
STAGE_ROOT="${STAGE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b/threshold_reanalysis}"
FIXED_TAUS="${FIXED_TAUS:-0.05 0.1 0.2 0.5 1.0}"
TINY_ALPHA_MAX="${TINY_ALPHA_MAX:-0.02}"
OVERWRITE="${OVERWRITE:-1}"

ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.reanalyze_bidirectional_thresholds \
  --stage-root "${STAGE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --fixed-taus ${FIXED_TAUS} \
  --tiny-alpha-max "${TINY_ALPHA_MAX}" \
  "${ARGS[@]}"
