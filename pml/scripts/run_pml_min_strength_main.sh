#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"

# Main path experiment. This input has target, neighbor, and capability probes.
INPUT_JSONL="${INPUT_JSONL:-/data/neural_controllers/pml/data/suppression_commonsenseqa_balanced_followup_144_with_capability.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-/data/neural_controllers/pml/results/pml_min_strength_main}"
DATASET_LABEL="${DATASET_LABEL:-pml_min_strength_main}"

# Default is a compact but meaningful run. Increase MAX_RECORDS or set it empty
# only when you want the full input file.
MAX_RECORDS="${MAX_RECORDS:-144}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
LAYERS="${LAYERS:--1 -5 -9 -13 -17 -21}"

# Dense alpha grid: designed to estimate onset/clean-window instead of only
# endpoint suppression at {-0.1,-0.25,-0.5,-1.0}.
SUPPRESSION_COEFS="${SUPPRESSION_COEFS:--0.01 -0.02 -0.03 -0.05 -0.08 -0.1 -0.15 -0.2 -0.25 -0.35 -0.5 -0.75 -1.0}"

# Main baselines. Keep random for calibration; set METHODS="mean_difference logistic"
# if you want a cheaper first run.
METHODS="${METHODS:-mean_difference logistic linear random}"

# Optional AGOP/RFM diagnostic branch. It is off by default because the current
# AGOP runner measures target/neighbor but not capability probes.
RUN_AGOP="${RUN_AGOP:-0}"
AGOP_METHODS="${AGOP_METHODS:-agop_top1 agop_topk_project}"
TOP_K="${TOP_K:-5}"
RFM_ITERS="${RFM_ITERS:-3}"
RFM_REG="${RFM_REG:-1e-3}"
RFM_BANDWIDTH="${RFM_BANDWIDTH:-10.0}"
RFM_KERNEL="${RFM_KERNEL:-laplace}"
RFM_MEM_GB="${RFM_MEM_GB:-8.0}"
SEED="${SEED:-113}"

OVERWRITE="${OVERWRITE:-1}"
PLOT="${PLOT:-1}"

TARGET_SUCCESS_DELTA="${TARGET_SUCCESS_DELTA:--0.05}"
NEIGHBOR_DAMAGE_DELTA="${NEIGHBOR_DAMAGE_DELTA:--0.05}"
CAPABILITY_DAMAGE_DELTA="${CAPABILITY_DAMAGE_DELTA:--0.05}"
MIN_CLEAN_ALPHA_POINTS="${MIN_CLEAN_ALPHA_POINTS:-1}"

if [[ ! -s "${INPUT_JSONL}" ]]; then
  echo "ERROR: input JSONL does not exist or is empty: ${INPUT_JSONL}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot see CUDA. Set DEVICE_MAP=cpu only for debugging; main runs should use GPU.", file=sys.stderr)
    print(f"torch_version={torch.__version__}", file=sys.stderr)
    print(f"cuda_device_count={torch.cuda.device_count()}", file=sys.stderr)
    raise SystemExit(2)

print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}")
PY

read -r -a LAYER_ARGS <<< "${LAYERS}"
read -r -a COEF_ARGS <<< "${SUPPRESSION_COEFS}"
read -r -a METHOD_ARGS <<< "${METHODS}"
read -r -a AGOP_METHOD_ARGS <<< "${AGOP_METHODS}"

mkdir -p "${RESULT_ROOT}"

RESULT_FILES=()
for METHOD in "${METHOD_ARGS[@]}"; do
  OUT_DIR="${RESULT_ROOT}/baselines/${METHOD}"
  RESULT_PATH="${OUT_DIR}/suppression_results.jsonl"
  if [[ "${OVERWRITE}" != "1" && -s "${RESULT_PATH}" ]]; then
    echo "Skipping baseline ${METHOD}; existing result: ${RESULT_PATH}"
  else
    echo "Running baseline ${METHOD}; output: ${OUT_DIR}"
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
      --device-map "${DEVICE_MAP}" \
      --seed "${SEED}" \
      "${EXTRA_ARGS[@]}"
  fi
  RESULT_FILES+=("${RESULT_PATH}")
done

BASELINE_SUMMARY_DIR="${RESULT_ROOT}/baselines/summary"
"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/summarize_suppression_results.py \
  --results-jsonl "${RESULT_FILES[@]}" \
  --output-dir "${BASELINE_SUMMARY_DIR}" \
  --success-delta "${TARGET_SUCCESS_DELTA}" \
  --neighbor-damage-delta "${NEIGHBOR_DAMAGE_DELTA}"

PLOT_ARGS=()
if [[ "${PLOT}" == "1" ]]; then
  PLOT_ARGS+=(--plot)
fi

"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_strength_curves.py \
  --per-record-csv "${BASELINE_SUMMARY_DIR}/per_record.csv" \
  --label "${DATASET_LABEL}" \
  --output-dir "${RESULT_ROOT}/path_geometry_audit" \
  --target-success-threshold "${TARGET_SUCCESS_DELTA}" \
  --neighbor-damage-threshold "${NEIGHBOR_DAMAGE_DELTA}" \
  --capability-damage-threshold "${CAPABILITY_DAMAGE_DELTA}" \
  --min-clean-alpha-points "${MIN_CLEAN_ALPHA_POINTS}" \
  "${PLOT_ARGS[@]}"

"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_minimum_effective_strength.py \
  --path-rows-csv "${RESULT_ROOT}/path_geometry_audit/strength_path_rows.csv" \
  --output-dir "${RESULT_ROOT}/minimum_effective_strength" \
  --dataset-label "${DATASET_LABEL}"

"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_strength_prediction.py \
  --path-rows-csv "${RESULT_ROOT}/path_geometry_audit/strength_path_rows.csv" \
  --output-dir "${RESULT_ROOT}/strength_prediction" \
  --include-method \
  --groups pooled dataset="${DATASET_LABEL}" \
  --label-keys clean_window_label usable_path_label damage_first_label capability_first_label no_effect_label \
  --numeric-keys target_onset_strength damage_onset_strength capability_onset_strength clean_window_width pareto_score_with_capability \
  --feature-groups baseline all

if [[ "${RUN_AGOP}" == "1" ]]; then
  AGOP_RESULT_FILES=()
  for METHOD in "${AGOP_METHOD_ARGS[@]}"; do
    OUT_DIR="${RESULT_ROOT}/agop/${METHOD}"
    RESULT_PATH="${OUT_DIR}/suppression_results.jsonl"
    if [[ "${OVERWRITE}" != "1" && -s "${RESULT_PATH}" ]]; then
      echo "Skipping AGOP ${METHOD}; existing result: ${RESULT_PATH}"
    else
      echo "Running AGOP ${METHOD}; output: ${OUT_DIR}"
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
    AGOP_RESULT_FILES+=("${RESULT_PATH}")
  done

  AGOP_SUMMARY_DIR="${RESULT_ROOT}/agop/summary"
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/summarize_suppression_results.py \
    --results-jsonl "${AGOP_RESULT_FILES[@]}" \
    --output-dir "${AGOP_SUMMARY_DIR}" \
    --success-delta "${TARGET_SUCCESS_DELTA}" \
    --neighbor-damage-delta "${NEIGHBOR_DAMAGE_DELTA}"

  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_strength_curves.py \
    --per-record-csv "${AGOP_SUMMARY_DIR}/per_record.csv" \
    --label "${DATASET_LABEL}_agop" \
    --output-dir "${RESULT_ROOT}/agop/path_geometry_audit" \
    --target-success-threshold "${TARGET_SUCCESS_DELTA}" \
    --neighbor-damage-threshold "${NEIGHBOR_DAMAGE_DELTA}" \
    --capability-damage-threshold "${CAPABILITY_DAMAGE_DELTA}" \
    --min-clean-alpha-points "${MIN_CLEAN_ALPHA_POINTS}" \
    "${PLOT_ARGS[@]}"

  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_minimum_effective_strength.py \
    --path-rows-csv "${RESULT_ROOT}/agop/path_geometry_audit/strength_path_rows.csv" \
    --output-dir "${RESULT_ROOT}/agop/minimum_effective_strength" \
    --dataset-label "${DATASET_LABEL}_agop"

  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_strength_prediction.py \
    --path-rows-csv "${RESULT_ROOT}/agop/path_geometry_audit/strength_path_rows.csv" \
    --output-dir "${RESULT_ROOT}/agop/strength_prediction" \
    --groups pooled dataset="${DATASET_LABEL}_agop" \
    --label-keys clean_window_label usable_path_label damage_first_label no_effect_label \
    --numeric-keys target_onset_strength damage_onset_strength clean_window_width pareto_score \
    --feature-groups agop agop_spectrum agop_topvec all
fi

echo "Main result root: ${RESULT_ROOT}"
echo "Baseline path rows: ${RESULT_ROOT}/path_geometry_audit/strength_path_rows.csv"
echo "Minimum strength report: ${RESULT_ROOT}/minimum_effective_strength/MINIMUM_EFFECTIVE_STRENGTH.md"
echo "Strength prediction root: ${RESULT_ROOT}/strength_prediction"
if [[ "${RUN_AGOP}" == "1" ]]; then
  echo "AGOP path rows: ${RESULT_ROOT}/agop/path_geometry_audit/strength_path_rows.csv"
fi
