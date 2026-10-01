#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
INPUT_JSONL="${INPUT_JSONL:-/data/neural_controllers/pml/data/suppression_commonsenseqa_balanced_followup_144.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-/data/neural_controllers/pml/results/commonsenseqa_balanced_followup_144/agop_direction_suppression}"
MAX_RECORDS="${MAX_RECORDS:-144}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LAYERS="${LAYERS:--1 -5 -9 -13 -17 -21}"
SUPPRESSION_COEFS="${SUPPRESSION_COEFS:--0.1 -0.25 -0.5 -1.0}"
METHODS="${METHODS:-agop_top1 agop_topk_project}"
TOP_K="${TOP_K:-5}"
RFM_ITERS="${RFM_ITERS:-3}"
RFM_REG="${RFM_REG:-1e-3}"
RFM_BANDWIDTH="${RFM_BANDWIDTH:-10.0}"
RFM_KERNEL="${RFM_KERNEL:-laplace}"
RFM_MEM_GB="${RFM_MEM_GB:-8.0}"
SEED="${SEED:-113}"
OVERWRITE="${OVERWRITE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
DEVICE="${DEVICE:-0}"

CUDA_AVAILABLE="$("${PYTHON_BIN}" - <<'PY'
import torch
print(1 if torch.cuda.is_available() else 0)
PY
)"
if [[ "${CUDA_AVAILABLE}" == "1" ]]; then
  DEFAULT_DEVICE_MAP="cuda:${DEVICE}"
else
  DEFAULT_DEVICE_MAP="cpu"
fi
DEVICE_MAP="${DEVICE_MAP:-${DEFAULT_DEVICE_MAP}}"

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: input JSONL does not exist or is empty: ${INPUT_JSONL}" >&2
  exit 1
fi

read -r -a LAYER_ARGS <<< "${LAYERS}"
read -r -a COEF_ARGS <<< "${SUPPRESSION_COEFS}"
read -r -a METHOD_ARGS <<< "${METHODS}"

mkdir -p "${RESULT_ROOT}"

RESULT_FILES=()
for METHOD in "${METHOD_ARGS[@]}"; do
  OUT_DIR="${RESULT_ROOT}/${METHOD}"
  RESULT_PATH="${OUT_DIR}/suppression_results.jsonl"
  if [[ "${OVERWRITE}" != "1" && -s "${RESULT_PATH}" ]]; then
    echo "Skipping ${METHOD}; existing result: ${RESULT_PATH}"
  else
    echo "Running ${METHOD}; output: ${OUT_DIR}"
    EXTRA_ARGS=()
    if [[ "${OVERWRITE}" == "1" ]]; then
      EXTRA_ARGS+=(--overwrite)
    fi
    "${PYTHON_BIN}" pml/src/predictive_memory_localization/methods/run_agop_suppression.py \
      --model-local-path "${MODEL_PATH}" \
      --input-jsonl "${INPUT_JSONL}" \
      --output-dir "${OUT_DIR}" \
      --control-method "${METHOD}" \
      --layers "${LAYER_ARGS[@]}" \
      --suppression-coefs "${COEF_ARGS[@]}" \
      --max-records "${MAX_RECORDS}" \
      --batch-size "${BATCH_SIZE}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      --top-k "${TOP_K}" \
      --rfm-iters "${RFM_ITERS}" \
      --rfm-reg "${RFM_REG}" \
      --rfm-bandwidth "${RFM_BANDWIDTH}" \
      --rfm-kernel "${RFM_KERNEL}" \
      --rfm-mem-gb "${RFM_MEM_GB}" \
      --seed "${SEED}" \
      "${EXTRA_ARGS[@]}"
  fi
  RESULT_FILES+=("${RESULT_PATH}")
done

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  SUMMARY_DIR="${RESULT_ROOT}/summary"
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/summarize_suppression_results.py \
    --results-jsonl "${RESULT_FILES[@]}" \
    --output-dir "${SUMMARY_DIR}"

  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_suppression_prediction.py \
    --per-record-csv "${SUMMARY_DIR}/per_record.csv" \
    --output-dir "${RESULT_ROOT}/prediction_analysis"
fi

echo "AGOP direction suppression root: ${RESULT_ROOT}"
