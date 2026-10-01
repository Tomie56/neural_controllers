#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
INPUT_JSONL="${INPUT_JSONL:-/data/neural_controllers/pml/data/suppression_commonsenseqa_balanced_followup_144.jsonl}"
TASKS_JSONL="${TASKS_JSONL:-/data/neural_controllers/pml/data/editing_tasks_commonsenseqa_balanced_followup_144.jsonl}"
REPORT_JSON="${REPORT_JSON:-/data/neural_controllers/pml/data/commonsenseqa_balanced_followup_144_report.json}"
EXCLUDE_JSONL="${EXCLUDE_JSONL:-/data/neural_controllers/pml/data/editing_tasks_commonsense_500_agop.jsonl}"
SOURCE_JSONL="${SOURCE_JSONL:-/data/neural_controllers/pml/data/suppression_commonsense_3000_valid.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-/data/neural_controllers/pml/results/commonsenseqa_balanced_followup_144}"
BASELINE_PER_RECORD_CSV="${BASELINE_PER_RECORD_CSV:-/data/neural_controllers/pml/results/commonsense_3000_baselines/summary/per_record.csv}"

BUILD_DATA="${BUILD_DATA:-0}"
RUN_AGOP="${RUN_AGOP:-1}"
RUN_ROME="${RUN_ROME:-1}"
RUN_GROUPED_ANALYSIS="${RUN_GROUPED_ANALYSIS:-1}"
RUN_ERROR_ABLATION="${RUN_ERROR_ABLATION:-${RUN_GROUPED_ANALYSIS}}"
RUN_CONTROLLED_ANALYSIS="${RUN_CONTROLLED_ANALYSIS:-${RUN_GROUPED_ANALYSIS}}"
RUN_BALANCED_ANALYSIS="${RUN_BALANCED_ANALYSIS:-1}"
RUN_MEMIT="${RUN_MEMIT:-0}"
MAX_RECORDS="${MAX_RECORDS:-144}"
PER_BIN="${PER_BIN:-48}"
SEED="${SEED:-101}"
TORCH_DTYPE_EDIT="${TORCH_DTYPE_EDIT:-float32}"
TORCH_DTYPE_AGOP="${TORCH_DTYPE_AGOP:-bfloat16}"
DEVICE="${DEVICE:-0}"
AGOP_LAYERS="${AGOP_LAYERS:--21 -17 -13 -9 -5 -1}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-300}"
CUDA_AVAILABLE="$("${PYTHON_BIN}" - <<'PY'
import torch
print(1 if torch.cuda.is_available() else 0)
PY
)"
if [[ "${CUDA_AVAILABLE}" == "1" ]]; then
  DEFAULT_AGOP_DEVICE_MAP="cuda:${DEVICE}"
else
  DEFAULT_AGOP_DEVICE_MAP="cpu"
fi
AGOP_DEVICE_MAP="${AGOP_DEVICE_MAP:-${DEFAULT_AGOP_DEVICE_MAP}}"

mkdir -p "${RESULT_ROOT}"

if [[ "${BUILD_DATA}" == "1" || ! -s "${INPUT_JSONL}" || ! -s "${TASKS_JSONL}" ]]; then
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/data/build_commonsenseqa_balanced_followup.py \
    --input-jsonl "${SOURCE_JSONL}" \
    --output-jsonl "${INPUT_JSONL}" \
    --tasks-jsonl "${TASKS_JSONL}" \
    --report-json "${REPORT_JSON}" \
    --exclude-jsonl "${EXCLUDE_JSONL}" \
    --per-bin "${PER_BIN}" \
    --seed "${SEED}" \
    --require-subject-in-prompt
fi

AGOP_DIR="${RESULT_ROOT}/agop_features"
AGOP_FEATURE_ROWS="${AGOP_DIR}/editing_feature_rows.csv"

if [[ "${RUN_AGOP}" == "1" ]]; then
  read -r -a LAYER_ARGS <<< "${AGOP_LAYERS}"
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/methods/extract_agop_features.py \
    --model-local-path "${MODEL_PATH}" \
    --input-jsonl "${INPUT_JSONL}" \
    --output-dir "${AGOP_DIR}" \
    --max-records "${MAX_RECORDS}" \
    --seed "${SEED}" \
    --layers "${LAYER_ARGS[@]}" \
    --batch-size 2 \
    --torch-dtype "${TORCH_DTYPE_AGOP}" \
    --device-map "${AGOP_DEVICE_MAP}" \
    --overwrite

  "${PYTHON_BIN}" pml/src/predictive_memory_localization/methods/export_raw_agop_feature_rows.py \
    --input-csv "${AGOP_DIR}/agop_features.csv" \
    --output-csv "${AGOP_FEATURE_ROWS}"
fi

if [[ "${RUN_ROME}" == "1" ]]; then
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/methods/run_easyedit_prediction.py \
    --tasks-jsonl "${TASKS_JSONL}" \
    --output-dir "${RESULT_ROOT}/rome" \
    --method ROME \
    --model-local-path "${MODEL_PATH}" \
    --max-records "${MAX_RECORDS}" \
    --torch-dtype "${TORCH_DTYPE_EDIT}" \
    --device "${DEVICE}" \
    --overwrite
fi

if [[ "${RUN_MEMIT}" == "1" ]]; then
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/methods/run_easyedit_prediction.py \
    --tasks-jsonl "${TASKS_JSONL}" \
    --output-dir "${RESULT_ROOT}/memit" \
    --method MEMIT \
    --model-local-path "${MODEL_PATH}" \
    --max-records "${MAX_RECORDS}" \
    --torch-dtype "${TORCH_DTYPE_EDIT}" \
    --device "${DEVICE}" \
    --overwrite
fi

if [[ "${RUN_GROUPED_ANALYSIS}" == "1" ]]; then
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_editing_prediction.py \
    --editing-results-jsonl "${RESULT_ROOT}/rome/rome_editing_results.jsonl" \
    --suppression-per-record-csv "${BASELINE_PER_RECORD_CSV}" \
    --output-dir "${RESULT_ROOT}/rome/prediction_analysis_baseline_features"

  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_editing_prediction.py \
    --editing-results-jsonl "${RESULT_ROOT}/rome/rome_editing_results.jsonl" \
    --suppression-per-record-csv "${AGOP_FEATURE_ROWS}" \
    --output-dir "${RESULT_ROOT}/rome/prediction_analysis_agop_features"

  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_editing_grouped_prediction.py \
    --editing-results-jsonl "${RESULT_ROOT}/rome/rome_editing_results.jsonl" \
    --baseline-joined-csv "${RESULT_ROOT}/rome/prediction_analysis_baseline_features/editing_joined_rows.csv" \
    --agop-joined-csv "${RESULT_ROOT}/rome/prediction_analysis_agop_features/editing_joined_rows.csv" \
    --output-dir "${RESULT_ROOT}/rome/grouped_prediction_analysis" \
    --bootstrap-iters "${BOOTSTRAP_ITERS}" \
    --seed "${SEED}"
fi

ERROR_ROWS_CSV="${RESULT_ROOT}/rome/error_ablation_analysis/rome500_error_rows.csv"

if [[ "${RUN_ERROR_ABLATION}" == "1" ]]; then
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_rome500_error_ablation.py \
    --editing-results-jsonl "${RESULT_ROOT}/rome/rome_editing_results.jsonl" \
    --baseline-joined-csv "${RESULT_ROOT}/rome/prediction_analysis_baseline_features/editing_joined_rows.csv" \
    --agop-joined-csv "${RESULT_ROOT}/rome/prediction_analysis_agop_features/editing_joined_rows.csv" \
    --output-dir "${RESULT_ROOT}/rome/error_ablation_analysis" \
    --bootstrap-iters "${BOOTSTRAP_ITERS}" \
    --seed "${SEED}"
fi

if [[ "${RUN_CONTROLLED_ANALYSIS}" == "1" ]]; then
  if [[ ! -s "${ERROR_ROWS_CSV}" ]]; then
    echo "Missing ${ERROR_ROWS_CSV}; run RUN_ERROR_ABLATION=1 first." >&2
    exit 1
  fi
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_rome500_controlled_incremental.py \
    --error-rows-csv "${ERROR_ROWS_CSV}" \
    --output-dir "${RESULT_ROOT}/rome/controlled_incremental_analysis" \
    --bootstrap-iters "${BOOTSTRAP_ITERS}" \
    --seed "${SEED}"
fi

if [[ "${RUN_BALANCED_ANALYSIS}" == "1" ]]; then
  if [[ ! -s "${ERROR_ROWS_CSV}" ]]; then
    echo "Missing ${ERROR_ROWS_CSV}; run RUN_ERROR_ABLATION=1 first." >&2
    exit 1
  fi
  for OUTCOME in efficacy rewrite_margin_delta rephrase_margin_delta; do
    "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_commonsenseqa_balanced_agop.py \
      --error-rows-csv "${ERROR_ROWS_CSV}" \
      --output-dir "${RESULT_ROOT}/rome/balanced_cv_full_rows/${OUTCOME}" \
      --outcome "${OUTCOME}" \
      --n-repeats 200 \
      --cv-folds 5 \
      --seed "${SEED}"
  done
fi

echo "CommonsenseQA balanced follow-up root: ${RESULT_ROOT}"
