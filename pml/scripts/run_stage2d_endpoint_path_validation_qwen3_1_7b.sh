#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"

PATH_PREDICTION_DIR="${PATH_PREDICTION_DIR:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b/pml_path_prediction}"
PATH_OUTCOME_CSV="${PATH_OUTCOME_CSV:-${PATH_PREDICTION_DIR}/pml_path_outcome_dataset.csv}"
ROOT_OUT="${ROOT_OUT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2d_endpoint_path_validation/qwen3_1_7b}"

MAX_SELECTED_RECORDS="${MAX_SELECTED_RECORDS:-60}"
PER_BUCKET="${PER_BUCKET:-16}"
METHODS="${METHODS:-mean_difference logistic random}"
LAYERS="${LAYERS:--21 -17}"
ALPHAS="${ALPHAS:--2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0}"

BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
TARGET_EVALS_PER_RECORD="${TARGET_EVALS_PER_RECORD:-1}"
NEIGHBOR_EVALS_PER_RECORD="${NEIGHBOR_EVALS_PER_RECORD:-1}"
CAPABILITY_EVALS_PER_RECORD="${CAPABILITY_EVALS_PER_RECORD:-1}"
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

mkdir -p "${ROOT_OUT}/selection"

echo "[Stage 2d] Selecting endpoint validation records"
echo "PATH_OUTCOME_CSV=${PATH_OUTCOME_CSV}"
echo "ROOT_OUT=${ROOT_OUT}"
echo "MAX_SELECTED_RECORDS=${MAX_SELECTED_RECORDS}"
echo "PER_BUCKET=${PER_BUCKET}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.select_endpoint_validation_records \
  --path-outcome-csv "${PATH_OUTCOME_CSV}" \
  --output-dir "${ROOT_OUT}/selection" \
  --max-records "${MAX_SELECTED_RECORDS}" \
  --per-bucket "${PER_BUCKET}" \
  --seed "${SEED}"

RECORD_IDS_FILE="${ROOT_OUT}/selection/endpoint_validation_record_ids.txt"
echo "RECORD_IDS_FILE=${RECORD_IDS_FILE}"
echo "METHODS=${METHODS}"
echo "LAYERS=${LAYERS}"
echo "ALPHAS=${ALPHAS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"

for METHOD in ${METHODS}; do
  for LAYER in ${LAYERS}; do
    LAYER_NAME="${LAYER/-/neg}"
    OUT_DIR="${ROOT_OUT}/${METHOD}/layer_${LAYER_NAME}"
    RUN_MODE_ARGS=()
    if [[ "${OVERWRITE}" == "1" ]]; then
      RUN_MODE_ARGS+=(--overwrite)
    elif [[ -s "${OUT_DIR}/raw_endpoint_generations.jsonl" ]]; then
      RUN_MODE_ARGS+=(--resume)
    else
      RUN_MODE_ARGS+=(--overwrite)
    fi

    echo "[Stage 2d endpoint] method=${METHOD} layer=${LAYER} out=${OUT_DIR} mode=${RUN_MODE_ARGS[*]}"
    "${PYTHON_BIN}" -m predictive_memory_localization.methods.run_endpoint_generation_three_table \
      --data-dir "${DATA_DIR}" \
      --model-local-path "${MODEL_PATH}" \
      --model-name "${MODEL_NAME}" \
      --output-dir "${OUT_DIR}" \
      --control-method "${METHOD}" \
      --max-records "${MAX_SELECTED_RECORDS}" \
      --record-ids-file "${RECORD_IDS_FILE}" \
      --layers "${LAYER}" \
      --alphas ${ALPHAS} \
      --batch-size "${BATCH_SIZE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --target-evals-per-record "${TARGET_EVALS_PER_RECORD}" \
      --neighbor-evals-per-record "${NEIGHBOR_EVALS_PER_RECORD}" \
      --capability-evals-per-record "${CAPABILITY_EVALS_PER_RECORD}" \
      --seed "${SEED}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      "${RUN_MODE_ARGS[@]}"
  done
done

echo "Stage 2d endpoint path validation root: ${ROOT_OUT}"
