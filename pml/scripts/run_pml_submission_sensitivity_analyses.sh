#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
RF_N_JOBS="${RF_N_JOBS:-8}"
OVERWRITE="${OVERWRITE:-0}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

echo "[submission sensitivity] semantic-audit exclusions"
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.analyze_semantic_audit_exclusion_sensitivity \
  --rf-n-jobs "${RF_N_JOBS}" \
  "${OVERWRITE_ARGS[@]}"

echo "[submission sensitivity] utility-weight robustness"
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.analyze_strength_utility_weight_sensitivity \
  "${OVERWRITE_ARGS[@]}"

echo "[submission sensitivity] complete"
