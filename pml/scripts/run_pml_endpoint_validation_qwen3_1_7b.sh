#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODE="${1:-${MODE:-formal}}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"
PATH_OUTCOME_CSV="${PATH_OUTCOME_CSV:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b/prediction_dataset/pml_path_prediction_dataset.csv}"
BASE_ROOT="${BASE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/endpoint_validation_v2/qwen3_1_7b}"

case "${MODE}" in
  smoke)
    ROOT_OUT="${ROOT_OUT:-${BASE_ROOT}/smoke}"
    MAX_SELECTED_RECORDS="${MAX_SELECTED_RECORDS:-8}"
    PER_BUCKET="${PER_BUCKET:-2}"
    METHODS="${METHODS:-mean_difference}"
    LAYERS="${LAYERS:--17}"
    ALPHAS="${ALPHAS:--0.1 0 0.1}"
    MIN_BASE_TARGET_RATE="${MIN_BASE_TARGET_RATE:-0.40}"
    ;;
  formal)
    ROOT_OUT="${ROOT_OUT:-${BASE_ROOT}/formal}"
    MAX_SELECTED_RECORDS="${MAX_SELECTED_RECORDS:-100}"
    PER_BUCKET="${PER_BUCKET:-25}"
    METHODS="${METHODS:-random mean_difference logistic rfm_agop_top1}"
    LAYERS="${LAYERS:--21 -17}"
    ALPHAS="${ALPHAS:--0.5 -0.25 -0.1 0 0.1 0.25 0.5}"
    MIN_BASE_TARGET_RATE="${MIN_BASE_TARGET_RATE:-0.0}"
    ;;
  *)
    echo "Usage: $0 [smoke|formal]" >&2
    exit 2
    ;;
esac

BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
TARGET_EVALS_PER_RECORD="${TARGET_EVALS_PER_RECORD:-1}"
NEIGHBOR_EVALS_PER_RECORD="${NEIGHBOR_EVALS_PER_RECORD:-1}"
CAPABILITY_EVALS_PER_RECORD="${CAPABILITY_EVALS_PER_RECORD:-1}"
ANSWER_PREFIX_TOKENS="${ANSWER_PREFIX_TOKENS:-24}"
SEED="${SEED:-113}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OVERWRITE="${OVERWRITE:-0}"
RFM_ITERS="${RFM_ITERS:-3}"
RFM_REG="${RFM_REG:-1e-3}"
RFM_BANDWIDTH="${RFM_BANDWIDTH:-10.0}"
RFM_KERNEL="${RFM_KERNEL:-laplace}"
RFM_MEM_GB="${RFM_MEM_GB:-8.0}"

trap 'echo; echo "Interrupted. Rerun the same command to resume." >&2' INT TERM

"${PYTHON_BIN}" - <<'PY'
from predictive_memory_localization.methods.run_endpoint_generation_three_table import contains_answer

assert contains_answer("100 V. What follows", "100 V")
assert not contains_answer("100 V. What follows", "10 V")
assert contains_answer("The answer is 3.14 meters because", "3.14 m")
assert not contains_answer("-10 V", "10 V")
print("Endpoint answer matcher preflight OK")
PY

if [[ "${DEVICE_MAP}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot see CUDA.", file=sys.stderr)
    raise SystemExit(2)
print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}")
PY
fi

mkdir -p "${ROOT_OUT}/selection"
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.select_endpoint_validation_records \
  --path-outcome-csv "${PATH_OUTCOME_CSV}" \
  --output-dir "${ROOT_OUT}/selection" \
  --max-records "${MAX_SELECTED_RECORDS}" \
  --per-bucket "${PER_BUCKET}" \
  --seed "${SEED}"

RECORD_IDS_FILE="${ROOT_OUT}/selection/endpoint_validation_record_ids.txt"
EXPECTED_RECORDS="$(wc -l < "${RECORD_IDS_FILE}")"
echo "[endpoint ${MODE}] records=${EXPECTED_RECORDS} methods=${METHODS} layers=${LAYERS} alphas=${ALPHAS}"

for METHOD in ${METHODS}; do
  for LAYER in ${LAYERS}; do
    LAYER_NAME="${LAYER/-/neg}"
    OUT_DIR="${ROOT_OUT}/${METHOD}/layer_${LAYER_NAME}"
    COMPLETED_RECORDS=0
    if [[ -s "${OUT_DIR}/endpoint_summary.json" ]]; then
      COMPLETED_RECORDS="$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("n_records", 0)))' "${OUT_DIR}/endpoint_summary.json")"
    fi
    UNRESOLVED_FAILURES=0
    if [[ -s "${OUT_DIR}/failures.jsonl" ]]; then
      UNRESOLVED_FAILURES="$(wc -l < "${OUT_DIR}/failures.jsonl")"
    fi
    if [[ "${OVERWRITE}" != "1" ]] && (( COMPLETED_RECORDS >= EXPECTED_RECORDS && UNRESOLVED_FAILURES == 0 )); then
      echo "[endpoint] skip completed method=${METHOD} layer=${LAYER} records=${COMPLETED_RECORDS}"
      continue
    fi

    RUN_MODE_ARGS=(--overwrite)
    if [[ "${OVERWRITE}" != "1" && -s "${OUT_DIR}/raw_endpoint_generations.jsonl" ]]; then
      RUN_MODE_ARGS=(--resume)
    fi
    echo "[endpoint] method=${METHOD} layer=${LAYER} mode=${RUN_MODE_ARGS[*]}"
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
      --answer-prefix-tokens "${ANSWER_PREFIX_TOKENS}" \
      --seed "${SEED}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      --rfm-iters "${RFM_ITERS}" \
      --rfm-reg "${RFM_REG}" \
      --rfm-bandwidth "${RFM_BANDWIDTH}" \
      --rfm-kernel "${RFM_KERNEL}" \
      --rfm-mem-gb "${RFM_MEM_GB}" \
      "${RUN_MODE_ARGS[@]}"

    COMPLETED_RECORDS="$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("n_records", 0)))' "${OUT_DIR}/endpoint_summary.json")"
    UNRESOLVED_FAILURES=0
    if [[ -s "${OUT_DIR}/failures.jsonl" ]]; then
      UNRESOLVED_FAILURES="$(wc -l < "${OUT_DIR}/failures.jsonl")"
    fi
    if (( COMPLETED_RECORDS < EXPECTED_RECORDS || UNRESOLVED_FAILURES > 0 )); then
      echo "ERROR: incomplete endpoint method=${METHOD} layer=${LAYER} completed=${COMPLETED_RECORDS}/${EXPECTED_RECORDS} unresolved_failures=${UNRESOLVED_FAILURES}" >&2
      exit 3
    fi
  done
done

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.analyze_endpoint_validation \
  --root-dir "${ROOT_OUT}" \
  --selection-csv "${ROOT_OUT}/selection/endpoint_validation_record_selection.csv"

"${PYTHON_BIN}" - "${ROOT_OUT}/endpoint_validation_analysis_manifest.json" "${MIN_BASE_TARGET_RATE}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
rate = manifest.get("baseline_target_endpoint_correct_rate")
minimum = float(sys.argv[2])
print(f"Endpoint baseline target accuracy: {rate}")
if rate is None or rate < minimum:
    raise SystemExit(f"ERROR: baseline target accuracy {rate} is below smoke threshold {minimum}")
PY

echo "[endpoint ${MODE}] complete: ${ROOT_OUT}"
