#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"
ROOT_OUT="${ROOT_OUT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage1_evaluator_smoke_50/qwen3_1_7b}"
MAX_RECORDS="${MAX_RECORDS:-50}"
METHODS="${METHODS:-mean_difference logistic random}"
LAYERS="${LAYERS:--1 -9 -17}"
ALPHAS="${ALPHAS:--0.1 -0.05 -0.02 0 0.02 0.05 0.1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-113}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

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
    "${PYTHON_BIN}" -m predictive_memory_localization.methods.run_bidirectional_three_table \
      --data-dir "${DATA_DIR}" \
      --model-local-path "${MODEL_PATH}" \
      --model-name "${MODEL_NAME}" \
      --output-dir "${OUT_DIR}" \
      --control-method "${METHOD}" \
      --max-records "${MAX_RECORDS}" \
      --layers "${LAYER}" \
      --alphas ${ALPHAS} \
      --batch-size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      "${RUN_MODE_ARGS[@]}"
  done
done
