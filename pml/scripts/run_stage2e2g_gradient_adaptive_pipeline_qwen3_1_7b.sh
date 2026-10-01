#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MAX_RECORDS="${MAX_RECORDS:-30}"
METHODS="${METHODS:-mean_difference logistic}"
LAYERS="${LAYERS:--21 -17}"
OVERWRITE="${OVERWRITE:-1}"
VALIDATE_AGAINST_DENSE="${VALIDATE_AGAINST_DENSE:-1}"

STAGE2E_ROOT="${STAGE2E_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2e_gradient_response/qwen3_1_7b}"
STAGE2G_ROOT="${STAGE2G_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2g_adaptive_alpha_scan/qwen3_1_7b}"
DECISION_OUT="${DECISION_OUT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2g_adaptive_alpha_scan/qwen3_1_7b/go_3000_decision}"

echo "[Pipeline] Stage 2e gradient response"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_RECORDS="${MAX_RECORDS}" \
METHODS="${METHODS}" \
LAYERS="${LAYERS}" \
OVERWRITE="${OVERWRITE}" \
VALIDATE_AGAINST_DENSE="${VALIDATE_AGAINST_DENSE}" \
./pml/scripts/run_stage2e_gradient_response_qwen3_1_7b.sh

echo "[Pipeline] Stage 2g adaptive alpha scan"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_RECORDS="${MAX_RECORDS}" \
METHODS_FILTER="${METHODS}" \
LAYERS_FILTER="${LAYERS}" \
OVERWRITE="${OVERWRITE}" \
VALIDATE_AGAINST_DENSE="${VALIDATE_AGAINST_DENSE}" \
./pml/scripts/run_stage2g_adaptive_alpha_scan_qwen3_1_7b.sh

if [[ "${VALIDATE_AGAINST_DENSE}" == "1" ]]; then
  echo "[Pipeline] GO/NO-GO decision"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.decide_gradient_response_readiness \
    --stage2e-root "${STAGE2E_ROOT}" \
    --stage2g-root "${STAGE2G_ROOT}" \
    --output-dir "${DECISION_OUT}"
  echo "Decision report: ${DECISION_OUT}/GO_3000_DECISION.md"
fi

echo "Pipeline complete."
