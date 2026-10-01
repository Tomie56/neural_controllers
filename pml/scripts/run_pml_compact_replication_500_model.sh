#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
SELECTION_DIR="${SELECTION_DIR:-/data/neural_controllers/pml/data/pml_replication_500_stratified_seed113}"
RECORD_IDS_FILE="${RECORD_IDS_FILE:-${SELECTION_DIR}/record_ids.txt}"

: "${MODEL_PATH:?Set MODEL_PATH to the downloaded model directory}"
: "${MODEL_NAME:?Set MODEL_NAME to a filesystem-safe experiment name}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500/${MODEL_NAME}}"
OUTCOME_ROOT="${OUTCOME_ROOT:-${EXPERIMENT_ROOT}/outcomes}"
PREDICTION_DATASET_DIR="${PREDICTION_DATASET_DIR:-${EXPERIMENT_ROOT}/prediction_dataset}"
PREDICTION_OUTPUT_DIR="${PREDICTION_OUTPUT_DIR:-${EXPERIMENT_ROOT}/prediction_analysis}"
STRICT_DATASET_DIR="${STRICT_DATASET_DIR:-${EXPERIMENT_ROOT}/prediction_dataset_strict_later}"
STRICT_OUTPUT_DIR="${STRICT_OUTPUT_DIR:-${EXPERIMENT_ROOT}/prediction_analysis_strict_later}"

MAX_RECORDS="${MAX_RECORDS:-500}"
METHODS="${METHODS:-random mean_difference logistic rfm_agop_top1}"
LAYERS="${LAYERS:-auto}"
REFERENCE_NUM_LAYERS="${REFERENCE_NUM_LAYERS:-28}"
REFERENCE_LAYERS="${REFERENCE_LAYERS:--21 -17}"
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
RUN_STANDARD_PREDICTION="${RUN_STANDARD_PREDICTION:-1}"
RUN_STRICT_LATER="${RUN_STRICT_LATER:-1}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${EXPERIMENT_ROOT}" "${OUTCOME_ROOT}"
trap 'echo; echo "Interrupted. Rerun the same command to resume." >&2' INT TERM

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.select_stratified_replication_records \
  --data-dir "${DATA_DIR}" \
  --output-dir "${SELECTION_DIR}" \
  --n-records 500 \
  --seed "${SEED}" >/dev/null

"${PYTHON_BIN}" - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = root / "config.json"
if not config_path.exists():
    raise SystemExit(f"ERROR: model download incomplete; missing {config_path}")
index_paths = sorted(root.glob("*.safetensors.index.json"))
if index_paths:
    for index_path in index_paths:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        files = sorted(set(payload.get("weight_map", {}).values()))
        missing = [name for name in files if not (root / name).is_file()]
        if missing:
            raise SystemExit(
                f"ERROR: model download incomplete; {len(missing)} missing shards, first={missing[:3]}"
            )
        expected_size = int((payload.get("metadata") or {}).get("total_size") or 0)
        actual_size = sum((root / name).stat().st_size for name in files)
        if expected_size and actual_size < expected_size:
            raise SystemExit(
                f"ERROR: model download incomplete; weight bytes={actual_size} expected={expected_size}"
            )
elif not any(root.glob("*.safetensors")):
    raise SystemExit(f"ERROR: model download incomplete; no safetensors under {root}")
print(f"Model file preflight OK: {root}")
PY

if [[ "${LAYERS}" == "auto" ]]; then
  LAYER_MAPPING_JSON="${EXPERIMENT_ROOT}/relative_layer_mapping.json"
  TRUST_ARGS=()
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    TRUST_ARGS+=(--trust-remote-code)
  fi
  LAYERS="$("${PYTHON_BIN}" -m predictive_memory_localization.analysis.resolve_relative_intervention_layers \
    --model-path "${MODEL_PATH}" \
    --reference-num-layers "${REFERENCE_NUM_LAYERS}" \
    --reference-layers ${REFERENCE_LAYERS} \
    --output-json "${LAYER_MAPPING_JSON}" \
    --format shell \
    "${TRUST_ARGS[@]}")"
fi

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

cat > "${EXPERIMENT_ROOT}/replication_config.env" <<EOF
DATA_DIR=${DATA_DIR}
SELECTION_DIR=${SELECTION_DIR}
RECORD_IDS_FILE=${RECORD_IDS_FILE}
MODEL_PATH=${MODEL_PATH}
MODEL_NAME=${MODEL_NAME}
MAX_RECORDS=${MAX_RECORDS}
METHODS=${METHODS}
LAYERS=${LAYERS}
REFERENCE_NUM_LAYERS=${REFERENCE_NUM_LAYERS}
REFERENCE_LAYERS=${REFERENCE_LAYERS}
ALPHAS=${ALPHAS}
BATCH_SIZE=${BATCH_SIZE}
SEED=${SEED}
TORCH_DTYPE=${TORCH_DTYPE}
DEVICE_MAP=${DEVICE_MAP}
MODEL_LOADER=${MODEL_LOADER}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
EOF

echo "[PML replication] model=${MODEL_NAME} path=${MODEL_PATH}"
echo "[PML replication] records=${MAX_RECORDS} ids=${RECORD_IDS_FILE}"
echo "[PML replication] methods=${METHODS} layers=${LAYERS} alphas=${ALPHAS}"
echo "[PML replication] root=${EXPERIMENT_ROOT}"

if [[ "${RUN_OUTCOMES}" == "1" ]]; then
  EXPECTED_RECORDS="$(awk 'NF {count += 1} END {print count + 0}' "${RECORD_IDS_FILE}")"
  if (( EXPECTED_RECORDS > MAX_RECORDS )); then
    EXPECTED_RECORDS="${MAX_RECORDS}"
  fi
  for METHOD in ${METHODS}; do
    for LAYER in ${LAYERS}; do
      LAYER_NAME="${LAYER/-/neg}"
      OUT_DIR="${OUTCOME_ROOT}/${METHOD}/layer_${LAYER_NAME}"
      mkdir -p "${OUT_DIR}"
      if [[ "${OVERWRITE}" != "1" && -s "${OUT_DIR}/method_summary.json" ]]; then
        COMPLETED_RECORDS="$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("n_records", 0)))' "${OUT_DIR}/method_summary.json")"
        UNRESOLVED_FAILURES=0
        if [[ -s "${OUT_DIR}/failures.jsonl" ]]; then
          UNRESOLVED_FAILURES="$(wc -l < "${OUT_DIR}/failures.jsonl")"
        fi
        if (( COMPLETED_RECORDS >= EXPECTED_RECORDS && UNRESOLVED_FAILURES == 0 )); then
          echo "[outcome] skip completed method=${METHOD} layer=${LAYER} records=${COMPLETED_RECORDS}"
          continue
        fi
      fi

      RUN_MODE_ARGS=()
      if [[ "${OVERWRITE}" == "1" ]]; then
        RUN_MODE_ARGS+=(--overwrite)
      elif [[ -s "${OUT_DIR}/raw_scores.jsonl" ]]; then
        RUN_MODE_ARGS+=(--resume)
      else
        RUN_MODE_ARGS+=(--overwrite)
      fi
      MODEL_ARGS=(--model-loader "${MODEL_LOADER}")
      if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
        MODEL_ARGS+=(--trust-remote-code)
      fi

      echo "[outcome] method=${METHOD} layer=${LAYER} mode=${RUN_MODE_ARGS[*]}"
      "${PYTHON_BIN}" -m predictive_memory_localization.methods.run_bidirectional_three_table \
        --data-dir "${DATA_DIR}" \
        --model-local-path "${MODEL_PATH}" \
        --model-name "${MODEL_NAME}" \
        --output-dir "${OUT_DIR}" \
        --control-method "${METHOD}" \
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

      COMPLETED_RECORDS="$("${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("n_records", 0)))' "${OUT_DIR}/method_summary.json")"
      UNRESOLVED_FAILURES=0
      if [[ -s "${OUT_DIR}/failures.jsonl" ]]; then
        UNRESOLVED_FAILURES="$(wc -l < "${OUT_DIR}/failures.jsonl")"
      fi
      if (( COMPLETED_RECORDS < EXPECTED_RECORDS || UNRESOLVED_FAILURES > 0 )); then
        echo "ERROR: incomplete path method=${METHOD} layer=${LAYER} completed=${COMPLETED_RECORDS}/${EXPECTED_RECORDS} unresolved_failures=${UNRESOLVED_FAILURES}" >&2
        echo "Fix the error and rerun the same command; postprocessing has not started." >&2
        exit 3
      fi
    done
  done
fi

if [[ "${RUN_POSTPROCESS}" != "1" ]]; then
  echo "[PML replication] Outcomes complete; postprocessing disabled"
  exit 0
fi

BUILD_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset \
  --stage-root "${OUTCOME_ROOT}" \
  --output-dir "${PREDICTION_DATASET_DIR}" \
  --random-method random \
  --null-quantile "${NULL_QUANTILE}" \
  --early-alpha "${EARLY_ALPHA}" \
  --medium-alpha "${MEDIUM_ALPHA}" \
  "${BUILD_ARGS[@]}"

if [[ "${RUN_STANDARD_PREDICTION}" == "1" ]]; then
  PREDICT_ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    PREDICT_ARGS+=(--overwrite)
  fi
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_activation_space_pml_main_prediction \
    --dataset-dir "${PREDICTION_DATASET_DIR}" \
    --output-dir "${PREDICTION_OUTPUT_DIR}" \
    --splits record dataset domain \
    --cohorts all learned rfm \
    --models logistic \
    --n-splits 5 \
    --seed "${SEED}" \
    "${PREDICT_ARGS[@]}"
fi

if [[ "${RUN_STRICT_LATER}" == "1" ]]; then
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_strict_later_dataset \
    --source-dataset-dir "${PREDICTION_DATASET_DIR}" \
    --stage-root "${OUTCOME_ROOT}" \
    --output-dir "${STRICT_DATASET_DIR}" \
    --early-alpha "${EARLY_ALPHA}" \
    --later-alphas ${LATER_ALPHAS} \
    "${BUILD_ARGS[@]}"
  STRICT_ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    STRICT_ARGS+=(--overwrite)
  fi
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_activation_space_pml_main_prediction \
    --dataset-dir "${STRICT_DATASET_DIR}" \
    --output-dir "${STRICT_OUTPUT_DIR}" \
    --targets \
      later_suppression_path later_enhancement_path later_target_any_path \
      later_neighbor_damage_any_path later_capability_damage_any_path \
      later_clean_suppression_path later_clean_enhancement_path later_clean_any_path \
      suppression_success_medium enhancement_success_medium \
      clean_suppression_medium clean_enhancement_medium \
    --splits record dataset domain \
    --cohorts all learned rfm \
    --models logistic \
    --n-splits 5 \
    --seed "${SEED}" \
    "${STRICT_ARGS[@]}"
fi

echo "[PML replication] complete model=${MODEL_NAME}"
echo "Outcome root: ${OUTCOME_ROOT}"
echo "Prediction report: ${PREDICTION_OUTPUT_DIR}/PML_MAIN_PREDICTION_REPORT.md"
echo "Strict-later report: ${STRICT_OUTPUT_DIR}/PML_MAIN_PREDICTION_REPORT.md"
