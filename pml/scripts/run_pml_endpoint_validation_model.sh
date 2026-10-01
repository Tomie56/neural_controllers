#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODE="${1:-${MODE:-formal}}"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"

: "${MODEL_PATH:?Set MODEL_PATH to the downloaded model directory}"
: "${MODEL_NAME:?Set MODEL_NAME to a filesystem-safe experiment name}"
: "${PATH_OUTCOME_CSV:?Set PATH_OUTCOME_CSV to the model-specific pml_path_prediction_dataset.csv}"

MODEL_LOADER="${MODEL_LOADER:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
BASE_ROOT="${BASE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/endpoint_validation_v2/${MODEL_NAME}}"
REFERENCE_NUM_LAYERS="${REFERENCE_NUM_LAYERS:-28}"
REFERENCE_LAYERS="${REFERENCE_LAYERS:--21 -17}"
LAYERS="${LAYERS:-auto}"

case "${MODE}" in
  smoke)
    ROOT_OUT="${ROOT_OUT:-${BASE_ROOT}/smoke}"
    MAX_SELECTED_RECORDS="${MAX_SELECTED_RECORDS:-8}"
    PER_BUCKET="${PER_BUCKET:-2}"
    METHODS="${METHODS:-mean_difference}"
    ALPHAS="${ALPHAS:--0.1 0 0.1}"
    MIN_BASE_TARGET_RATE="${MIN_BASE_TARGET_RATE:-0.0}"
    ;;
  formal)
    ROOT_OUT="${ROOT_OUT:-${BASE_ROOT}/formal}"
    MAX_SELECTED_RECORDS="${MAX_SELECTED_RECORDS:-100}"
    PER_BUCKET="${PER_BUCKET:-25}"
    METHODS="${METHODS:-random mean_difference logistic rfm_agop_top1}"
    ALPHAS="${ALPHAS:--1 -0.5 -0.25 -0.1 0 0.1 0.25 0.5 1}"
    MIN_BASE_TARGET_RATE="${MIN_BASE_TARGET_RATE:-0.0}"
    ;;
  *)
    echo "Usage: $0 [smoke|formal]" >&2
    exit 2
    ;;
esac

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
TARGET_EVALS_PER_RECORD="${TARGET_EVALS_PER_RECORD:-3}"
NEIGHBOR_EVALS_PER_RECORD="${NEIGHBOR_EVALS_PER_RECORD:-3}"
CAPABILITY_EVALS_PER_RECORD="${CAPABILITY_EVALS_PER_RECORD:-4}"
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

if [[ ! -s "${PATH_OUTCOME_CSV}" ]]; then
  echo "ERROR: missing model-specific path outcome CSV: ${PATH_OUTCOME_CSV}" >&2
  echo "Run the 500-record replication through postprocessing first." >&2
  exit 2
fi

TRUST_ARGS=()
MODEL_ARGS=(--model-loader "${MODEL_LOADER}")
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_ARGS+=(--trust-remote-code)
  MODEL_ARGS+=(--trust-remote-code)
fi

if [[ "${LAYERS}" == "auto" ]]; then
  LAYER_MAPPING_JSON="${BASE_ROOT}/relative_layer_mapping.json"
  LAYERS="$("${PYTHON_BIN}" -m predictive_memory_localization.analysis.resolve_relative_intervention_layers \
    --model-path "${MODEL_PATH}" \
    --reference-num-layers "${REFERENCE_NUM_LAYERS}" \
    --reference-layers ${REFERENCE_LAYERS} \
    --output-json "${LAYER_MAPPING_JSON}" \
    --format shell \
    "${TRUST_ARGS[@]}")"
fi

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
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.select_endpoint_validation_paths \
  --path-outcome-csv "${PATH_OUTCOME_CSV}" \
  --output-dir "${ROOT_OUT}/selection" \
  --max-paths "${MAX_SELECTED_RECORDS}" \
  --per-bucket "${PER_BUCKET}" \
  --methods ${METHODS} \
  --layers ${LAYERS} \
  --seed "${SEED}"

TOTAL_RECORD_IDS_FILE="${ROOT_OUT}/selection/endpoint_validation_record_ids.txt"
TOTAL_EXPECTED_RECORDS="$(wc -l < "${TOTAL_RECORD_IDS_FILE}")"
echo "[endpoint ${MODEL_NAME} ${MODE}] selected_paths=${TOTAL_EXPECTED_RECORDS} methods=${METHODS} layers=${LAYERS} alphas=${ALPHAS}"

for METHOD in ${METHODS}; do
  for LAYER in ${LAYERS}; do
    LAYER_NAME="${LAYER/-/neg}"
    RECORD_IDS_FILE="${ROOT_OUT}/selection/path_record_ids/${METHOD}/layer_${LAYER_NAME}.txt"
    if [[ ! -s "${RECORD_IDS_FILE}" ]]; then
      echo "[endpoint] skip unselected model=${MODEL_NAME} method=${METHOD} layer=${LAYER}"
      continue
    fi
    EXPECTED_RECORDS="$(wc -l < "${RECORD_IDS_FILE}")"
    OUT_DIR="${ROOT_OUT}/${METHOD}/layer_${LAYER_NAME}"
    COMPLETED_RECORDS=0
    if [[ -s "${OUT_DIR}/endpoint_summary.json" ]]; then
      COMPLETED_RECORDS="$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("n_records", 0)))' "${OUT_DIR}/endpoint_summary.json")"
    elif [[ -s "${OUT_DIR}/raw_endpoint_generations.jsonl" ]]; then
      COMPLETED_RECORDS="$(wc -l < "${OUT_DIR}/raw_endpoint_generations.jsonl")"
    fi
    UNRESOLVED_FAILURES=0
    if [[ -s "${OUT_DIR}/failures.jsonl" ]]; then
      UNRESOLVED_FAILURES="$(wc -l < "${OUT_DIR}/failures.jsonl")"
    fi
    if [[ "${OVERWRITE}" != "1" && -s "${OUT_DIR}/endpoint_summary.json" ]] && (( COMPLETED_RECORDS >= EXPECTED_RECORDS && UNRESOLVED_FAILURES == 0 )); then
      echo "[endpoint] skip completed model=${MODEL_NAME} method=${METHOD} layer=${LAYER} records=${COMPLETED_RECORDS}"
      continue
    fi

    RUN_MODE_ARGS=(--overwrite)
    if [[ "${OVERWRITE}" != "1" ]] && (( COMPLETED_RECORDS >= EXPECTED_RECORDS )); then
      RUN_MODE_ARGS=(--postprocess-only)
    elif [[ "${OVERWRITE}" != "1" && -s "${OUT_DIR}/raw_endpoint_generations.jsonl" ]]; then
      RUN_MODE_ARGS=(--resume)
    fi
    echo "[endpoint] model=${MODEL_NAME} method=${METHOD} layer=${LAYER} mode=${RUN_MODE_ARGS[*]}"
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
      "${MODEL_ARGS[@]}" \
      "${RUN_MODE_ARGS[@]}"

    COMPLETED_RECORDS="$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("n_records", 0)))' "${OUT_DIR}/endpoint_summary.json")"
    UNRESOLVED_FAILURES=0
    if [[ -s "${OUT_DIR}/failures.jsonl" ]]; then
      UNRESOLVED_FAILURES="$(wc -l < "${OUT_DIR}/failures.jsonl")"
    fi
    if (( COMPLETED_RECORDS < EXPECTED_RECORDS || UNRESOLVED_FAILURES > 0 )); then
      echo "ERROR: incomplete endpoint model=${MODEL_NAME} method=${METHOD} layer=${LAYER} completed=${COMPLETED_RECORDS}/${EXPECTED_RECORDS} unresolved_failures=${UNRESOLVED_FAILURES}" >&2
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
    raise SystemExit(f"ERROR: baseline target accuracy {rate} is below threshold {minimum}")
PY

echo "[endpoint ${MODEL_NAME} ${MODE}] complete: ${ROOT_OUT}"
