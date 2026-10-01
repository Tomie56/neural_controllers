#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
STAGE_ROOT="${STAGE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGE_ROOT}/selector_dataset}"
TAU_SOURCE="${TAU_SOURCE:-global_random_q95}"
OVERWRITE="${OVERWRITE:-1}"

ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_strength_selector_dataset \
  --stage-root "${STAGE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --tau-source "${TAU_SOURCE}" \
  "${ARGS[@]}"
