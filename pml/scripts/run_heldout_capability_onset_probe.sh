#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
INPUT_JSONL="${INPUT_JSONL:-/data/neural_controllers/pml/data/suppression_commonsenseqa_balanced_followup_144_with_capability.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-/data/neural_controllers/pml/results/heldout_50_capability_onset_probe}"
MAX_RECORDS="${MAX_RECORDS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LAYERS="${LAYERS:--1 -5 -9 -13 -17 -21}"
SUPPRESSION_COEFS="${SUPPRESSION_COEFS:--0.02 -0.05 -0.08 -0.1 -0.15 -0.2 -0.25 -0.35 -0.5 -0.75 -1.0}"
METHODS="${METHODS:-mean_difference logistic}"
OVERWRITE="${OVERWRITE:-1}"
DATASET_LABEL="${DATASET_LABEL:-heldout_50_capability_onset_probe}"
DIAGNOSTICS_CSV="${DIAGNOSTICS_CSV:-}"

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: input JSONL does not exist or is empty: ${INPUT_JSONL}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot see CUDA. Run this script in an environment where torch.cuda.is_available() is True.", file=sys.stderr)
    print(f"torch_version={torch.__version__}", file=sys.stderr)
    print(f"cuda_device_count={torch.cuda.device_count()}", file=sys.stderr)
    raise SystemExit(2)

print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}")
PY

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

SUMMARY_DIR="${RESULT_ROOT}/summary"
"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/summarize_suppression_results.py \
  --results-jsonl "${RESULT_FILES[@]}" \
  --output-dir "${SUMMARY_DIR}"

"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_strength_curves.py \
  --per-record-csv "${SUMMARY_DIR}/per_record.csv" \
  --label "${DATASET_LABEL}" \
  --output-dir "${RESULT_ROOT}/path_geometry_audit" \
  --plot

"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_strength_prediction.py \
  --path-rows-csv "${RESULT_ROOT}/path_geometry_audit/strength_path_rows.csv" \
  --output-dir "${RESULT_ROOT}/path_geometry_audit/strength_prediction" \
  --groups pooled dataset="${DATASET_LABEL}" \
  --label-keys clean_window_label usable_path_label damage_first_label capability_first_label no_effect_label \
  --numeric-keys target_onset_strength damage_onset_strength capability_onset_strength clean_window_width pareto_score_with_capability \
  --feature-groups baseline all

if [[ -n "${DIAGNOSTICS_CSV}" ]]; then
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_representative_subset_outcomes.py \
    --path-rows-csv "${RESULT_ROOT}/path_geometry_audit/strength_path_rows.csv" \
    --diagnostics-csv "${DIAGNOSTICS_CSV}" \
    --output-dir "${RESULT_ROOT}/path_geometry_audit/representative_outcomes" \
    --dataset-label "${DATASET_LABEL}"
fi

echo "Capability-onset probe results root: ${RESULT_ROOT}"
