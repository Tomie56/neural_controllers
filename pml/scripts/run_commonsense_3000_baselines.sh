#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
INPUT_JSONL="${INPUT_JSONL:-/data/neural_controllers/pml/data/suppression_commonsense_3000.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-/data/neural_controllers/pml/results/commonsense_3000_baselines}"
MAX_RECORDS="${MAX_RECORDS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LAYERS="${LAYERS:--1 -5 -9 -13 -17 -21}"
SUPPRESSION_COEFS="${SUPPRESSION_COEFS:--0.1 -0.25 -0.5 -1.0}"
METHODS="${METHODS:-logistic mean_difference linear random}"
OVERWRITE="${OVERWRITE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

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
    "${PYTHON_BIN}" pml/src/predictive_memory_localization/methods/run_activation_suppression.py \
      --model-local-path "${MODEL_PATH}" \
      --input-jsonl "${INPUT_JSONL}" \
      --output-dir "${OUT_DIR}" \
      --control-method "${METHOD}" \
      --layers "${LAYER_ARGS[@]}" \
      --suppression-coefs "${COEF_ARGS[@]}" \
      --max-records "${MAX_RECORDS}" \
      --batch-size "${BATCH_SIZE}" \
      --torch-dtype "${TORCH_DTYPE}" \
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

echo "Baseline results root: ${RESULT_ROOT}"
