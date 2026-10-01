#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"
ROOT_OUT="${ROOT_OUT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/stage1_vector_family_pilot/qwen3_1_7b}"

MAX_RECORDS="${MAX_RECORDS:-300}"
RFM_MAX_RECORDS="${RFM_MAX_RECORDS:-100}"
RECORD_OFFSET="${RECORD_OFFSET:-0}"
METHODS="${METHODS:-random matched_norm_random mean_difference logistic linear pca_diff rfm_agop_top1}"
LAYERS="${LAYERS:--21 -17 -13 -9}"
ALPHAS="${ALPHAS:--0.5 -0.25 -0.1 0 0.1 0.25 0.5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-113}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

RFM_ITERS="${RFM_ITERS:-3}"
RFM_REG="${RFM_REG:-1e-3}"
RFM_BANDWIDTH="${RFM_BANDWIDTH:-10.0}"
RFM_KERNEL="${RFM_KERNEL:-laplace}"
RFM_MEM_GB="${RFM_MEM_GB:-8.0}"

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

echo "[Activation-Space PML Stage 1]"
echo "root=${ROOT_OUT}"
echo "methods=${METHODS}"
echo "layers=${LAYERS}"
echo "alphas=${ALPHAS}"
echo "max_records=${MAX_RECORDS}; rfm_max_records=${RFM_MAX_RECORDS}; offset=${RECORD_OFFSET}"

for METHOD in ${METHODS}; do
  METHOD_MAX_RECORDS="${MAX_RECORDS}"
  if [[ "${METHOD}" == rfm_agop_* ]]; then
    METHOD_MAX_RECORDS="${RFM_MAX_RECORDS}"
  fi

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

    echo "[Stage 1] method=${METHOD} layer=${LAYER} records=${METHOD_MAX_RECORDS} out=${OUT_DIR}"
    "${PYTHON_BIN}" -m predictive_memory_localization.methods.run_bidirectional_three_table \
      --data-dir "${DATA_DIR}" \
      --model-local-path "${MODEL_PATH}" \
      --model-name "${MODEL_NAME}" \
      --output-dir "${OUT_DIR}" \
      --control-method "${METHOD}" \
      --max-records "${METHOD_MAX_RECORDS}" \
      --record-offset "${RECORD_OFFSET}" \
      --layers "${LAYER}" \
      --alphas ${ALPHAS} \
      --batch-size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      --rfm-iters "${RFM_ITERS}" \
      --rfm-reg "${RFM_REG}" \
      --rfm-bandwidth "${RFM_BANDWIDTH}" \
      --rfm-kernel "${RFM_KERNEL}" \
      --rfm-mem-gb "${RFM_MEM_GB}" \
      "${RUN_MODE_ARGS[@]}"
  done
done

echo "Stage 1 vector family pilot complete: ${ROOT_OUT}"
