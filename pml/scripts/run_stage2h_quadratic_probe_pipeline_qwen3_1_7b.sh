#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"

ROOT_OUT="${ROOT_OUT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2h_quadratic_probe/qwen3_1_7b}"
DENSE_ROOT="${DENSE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b}"

MAX_RECORDS="${MAX_RECORDS:-500}"
RECORD_OFFSET="${RECORD_OFFSET:-0}"
METHODS="${METHODS:-mean_difference logistic random}"
LAYERS="${LAYERS:--21 -17 -13}"
PROBE_ALPHAS="${PROBE_ALPHAS:--0.05 0.0 0.05}"
PREDICT_ALPHAS="${PREDICT_ALPHAS:--1.0 -0.75 -0.5 -0.35 -0.25 -0.2 -0.15 -0.1 -0.08 -0.05 -0.03 -0.02 -0.01 0.0 0.01 0.02 0.03 0.05 0.08 0.1 0.15 0.2 0.25 0.35 0.5 0.75 1.0}"
TAU="${TAU:-0.1522890289624531}"

BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-113}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OVERWRITE="${OVERWRITE:-0}"
VALIDATE_AGAINST_DENSE="${VALIDATE_AGAINST_DENSE:-1}"

if [[ "${DEVICE_MAP}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot see CUDA. Run this script from a shell with GPU access.", file=sys.stderr)
    print(f"torch_version={torch.__version__}", file=sys.stderr)
    print(f"cuda_device_count={torch.cuda.device_count()}", file=sys.stderr)
    raise SystemExit(2)
print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}")
PY
fi

mkdir -p "${ROOT_OUT}"

echo "Stage 2h quadratic probe"
echo "ROOT_OUT=${ROOT_OUT}"
echo "MAX_RECORDS=${MAX_RECORDS}"
echo "METHODS=${METHODS}"
echo "LAYERS=${LAYERS}"
echo "PROBE_ALPHAS=${PROBE_ALPHAS}"

for METHOD in ${METHODS}; do
  for LAYER in ${LAYERS}; do
    LAYER_NAME="${LAYER/-/neg}"
    OUT_DIR="${ROOT_OUT}/${METHOD}/layer_${LAYER_NAME}"
    RUN_MODE_ARGS=()
    if [[ "${OVERWRITE}" == "1" ]]; then
      RUN_MODE_ARGS+=(--overwrite)
    elif [[ -s "${OUT_DIR}/raw_scores.jsonl" ]]; then
      RUN_MODE_ARGS+=(--resume)
    else
      RUN_MODE_ARGS+=(--overwrite)
    fi

    echo "[Stage 2h probe scan] method=${METHOD} layer=${LAYER} out=${OUT_DIR} mode=${RUN_MODE_ARGS[*]}"
    "${PYTHON_BIN}" -m predictive_memory_localization.methods.run_bidirectional_three_table \
      --data-dir "${DATA_DIR}" \
      --model-local-path "${MODEL_PATH}" \
      --model-name "${MODEL_NAME}" \
      --output-dir "${OUT_DIR}" \
      --control-method "${METHOD}" \
      --max-records "${MAX_RECORDS}" \
      --record-offset "${RECORD_OFFSET}" \
      --layers "${LAYER}" \
      --alphas ${PROBE_ALPHAS} \
      --batch-size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      --target-threshold "${TAU}" \
      --neighbor-threshold "${TAU}" \
      --capability-threshold "${TAU}" \
      "${RUN_MODE_ARGS[@]}"
  done
done

COMPARE_ARGS=()
if [[ "${VALIDATE_AGAINST_DENSE}" == "1" ]]; then
  COMPARE_ARGS+=(--dense-root "${DENSE_ROOT}")
fi

echo "[Stage 2h] Fitting quadratic response from probe scan"
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.fit_quadratic_response_from_probe_scan \
  --probe-root "${ROOT_OUT}" \
  --output-dir "${ROOT_OUT}/quadratic_response" \
  --predict-alphas ${PREDICT_ALPHAS} \
  --tau "${TAU}" \
  "${COMPARE_ARGS[@]}"

echo "Stage 2h quadratic response report: ${ROOT_OUT}/quadratic_response/QUADRATIC_RESPONSE_REPORT.md"
