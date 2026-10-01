#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
RECORD_IDS_FILE="${RECORD_IDS_FILE:-/data/neural_controllers/pml/data/pml_replication_500_stratified_seed113/record_ids.txt}"
SCALE_MANIFEST="${SCALE_MANIFEST:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/scale_calibration_residual_rms_v2/residual_rms_reference_scale.json}"

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MODEL_NAME:?Set MODEL_NAME to the scale-manifest model key}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500_residual_rms_v2/${MODEL_NAME}}"
OUTCOME_ROOT="${OUTCOME_ROOT:-${EXPERIMENT_ROOT}/outcomes}"
PREDICTION_DATASET_DIR="${PREDICTION_DATASET_DIR:-${EXPERIMENT_ROOT}/prediction_dataset}"
STRICT_DATASET_DIR="${STRICT_DATASET_DIR:-${EXPERIMENT_ROOT}/prediction_dataset_later_strength}"
STRICT_SUMMARY_DIR="${STRICT_SUMMARY_DIR:-${EXPERIMENT_ROOT}/outcome_analysis_later_strength}"

MAX_RECORDS="${MAX_RECORDS:-500}"
ALPHAS="${ALPHAS:--0.5 -0.25 -0.1 0 0.1 0.25 0.5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-113}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
MODEL_LOADER="${MODEL_LOADER:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
RFM_ITERS="${RFM_ITERS:-3}"
RFM_REG="${RFM_REG:-1e-3}"
RFM_BANDWIDTH="${RFM_BANDWIDTH:-10.0}"
RFM_KERNEL="${RFM_KERNEL:-laplace}"
RFM_MEM_GB="${RFM_MEM_GB:-8.0}"
NULL_QUANTILE="${NULL_QUANTILE:-0.95}"
EARLY_ALPHA="${EARLY_ALPHA:-0.1}"
MEDIUM_ALPHA="${MEDIUM_ALPHA:-0.5}"
LATER_ALPHAS="${LATER_ALPHAS:-0.25 0.5}"
RUN_OUTCOMES="${RUN_OUTCOMES:-1}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
OVERWRITE="${OVERWRITE:-0}"
CROSS_MODEL_METHODS="${CROSS_MODEL_METHODS:-random mean_difference logistic rfm_agop_top1}"

if [[ ! -s "${SCALE_MANIFEST}" ]]; then
  echo "ERROR: missing scale manifest: ${SCALE_MANIFEST}" >&2
  echo "Run pml/scripts/run_pml_residual_rms_calibration.sh first." >&2
  exit 2
fi

mapfile -t PATH_SPECS < <("${PYTHON_BIN}" - "${SCALE_MANIFEST}" "${MODEL_NAME}" "${CROSS_MODEL_METHODS}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
model_key = sys.argv[2]
try:
    layers = payload["models"][model_key]["layers"]
except KeyError as exc:
    raise SystemExit(f"Model key {model_key!r} is absent from scale manifest") from exc
by_reference = {int(row["reference_layer"]): row for row in layers.values()}
methods = sys.argv[3].split()
if not methods:
    raise SystemExit("CROSS_MODEL_METHODS must contain at least one method")
for reference_layer in [-21, -17]:
    row = by_reference[reference_layer]
    for method in methods:
        print(f'{method}\t{int(row["target_layer"])}\t{float(row["control_scale"]):.17g}\t{reference_layer}')
PY
)

mkdir -p "${EXPERIMENT_ROOT}" "${OUTCOME_ROOT}"
trap 'echo; echo "Interrupted. Rerun the same command to resume." >&2' INT TERM

if [[ "${RUN_OUTCOMES}" == "1" && "${DEVICE_MAP}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot see CUDA.", file=sys.stderr)
    raise SystemExit(2)
print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}")
PY
fi

SCALE_SHA256="$(${PYTHON_BIN} -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${SCALE_MANIFEST}")"
cat > "${EXPERIMENT_ROOT}/replication_config.env" <<EOF
PROTOCOL=residual_rms_reference_matching_dimension_corrected_v2
DATA_DIR=${DATA_DIR}
RECORD_IDS_FILE=${RECORD_IDS_FILE}
MODEL_PATH=${MODEL_PATH}
MODEL_NAME=${MODEL_NAME}
SCALE_MANIFEST=${SCALE_MANIFEST}
SCALE_MANIFEST_SHA256=${SCALE_SHA256}
MAX_RECORDS=${MAX_RECORDS}
ALPHAS=${ALPHAS}
BATCH_SIZE=${BATCH_SIZE}
SEED=${SEED}
MODEL_LOADER=${MODEL_LOADER}
CROSS_MODEL_METHODS=${CROSS_MODEL_METHODS}
EOF

echo "[RMS replication] model=${MODEL_NAME} root=${EXPERIMENT_ROOT}"
printf '[RMS replication] paths:\n'
printf '  %s\n' "${PATH_SPECS[@]}"

EXPECTED_RECORDS="$(awk 'NF {count += 1} END {print count + 0}' "${RECORD_IDS_FILE}")"
if (( EXPECTED_RECORDS > MAX_RECORDS )); then
  EXPECTED_RECORDS="${MAX_RECORDS}"
fi

if [[ "${RUN_OUTCOMES}" == "1" ]]; then
  for SPEC in "${PATH_SPECS[@]}"; do
    IFS=$'\t' read -r METHOD LAYER CONTROL_SCALE REFERENCE_LAYER <<<"${SPEC}"
    LAYER_NAME="${LAYER/-/neg}"
    OUT_DIR="${OUTCOME_ROOT}/${METHOD}/layer_${LAYER_NAME}"
    mkdir -p "${OUT_DIR}"
    if [[ "${OVERWRITE}" != "1" && -s "${OUT_DIR}/method_summary.json" ]]; then
      read -r COMPLETED_RECORDS SAVED_SCALE N_FAILURES < <("${PYTHON_BIN}" - "${OUT_DIR}/method_summary.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
print(int(row.get("n_records",0)), float(row.get("control_scale",1.0)), int(row.get("n_failures",0)))
PY
)
      if (( COMPLETED_RECORDS >= EXPECTED_RECORDS && N_FAILURES == 0 )) && \
        "${PYTHON_BIN}" -c 'import math,sys; raise SystemExit(0 if math.isclose(float(sys.argv[1]),float(sys.argv[2]),rel_tol=1e-12,abs_tol=1e-12) else 1)' "${SAVED_SCALE}" "${CONTROL_SCALE}"; then
        echo "[outcome] skip completed method=${METHOD} layer=${LAYER} records=${COMPLETED_RECORDS}"
        continue
      fi
    fi

    RUN_MODE_ARGS=(--overwrite)
    if [[ "${OVERWRITE}" != "1" && -s "${OUT_DIR}/raw_scores.jsonl" ]]; then
      RUN_MODE_ARGS=(--resume)
    fi
    MODEL_ARGS=(--model-loader "${MODEL_LOADER}")
    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
      MODEL_ARGS+=(--trust-remote-code)
    fi
    echo "[outcome] method=${METHOD} reference_layer=${REFERENCE_LAYER} layer=${LAYER} scale=${CONTROL_SCALE} mode=${RUN_MODE_ARGS[*]}"
    "${PYTHON_BIN}" -m predictive_memory_localization.methods.run_bidirectional_three_table \
      --data-dir "${DATA_DIR}" \
      --model-local-path "${MODEL_PATH}" \
      --model-name "${MODEL_NAME}" \
      --output-dir "${OUT_DIR}" \
      --control-method "${METHOD}" \
      --control-scale "${CONTROL_SCALE}" \
      --max-records "${MAX_RECORDS}" \
      --record-ids-file "${RECORD_IDS_FILE}" \
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
      "${MODEL_ARGS[@]}" \
      "${RUN_MODE_ARGS[@]}"
  done
fi

if [[ "${RUN_POSTPROCESS}" != "1" ]]; then
  echo "[RMS replication] outcomes complete; postprocessing disabled"
  exit 0
fi

BUILD_ARGS=()
SUMMARY_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
  SUMMARY_ARGS+=(--overwrite)
elif [[ -s "${STRICT_SUMMARY_DIR}/strict_later_outcome_summary.csv" ]]; then
  echo "[RMS replication] postprocessing already complete: ${STRICT_SUMMARY_DIR}"
  exit 0
else
  SUMMARY_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset \
  --stage-root "${OUTCOME_ROOT}" \
  --output-dir "${PREDICTION_DATASET_DIR}" \
  --random-method random \
  --null-quantile "${NULL_QUANTILE}" \
  --early-alpha "${EARLY_ALPHA}" \
  --medium-alpha "${MEDIUM_ALPHA}" \
  "${BUILD_ARGS[@]}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_strict_later_dataset \
  --source-dataset-dir "${PREDICTION_DATASET_DIR}" \
  --stage-root "${OUTCOME_ROOT}" \
  --output-dir "${STRICT_DATASET_DIR}" \
  --early-alpha "${EARLY_ALPHA}" \
  --later-alphas ${LATER_ALPHAS} \
  "${BUILD_ARGS[@]}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.summarize_strict_later_outcomes \
  --dataset-csv "${STRICT_DATASET_DIR}/pml_path_prediction_dataset.csv" \
  --output-dir "${STRICT_SUMMARY_DIR}" \
  "${SUMMARY_ARGS[@]}"

echo "[RMS replication] complete model=${MODEL_NAME}"
echo "Outcome summary: ${STRICT_SUMMARY_DIR}/strict_later_outcome_summary.csv"
