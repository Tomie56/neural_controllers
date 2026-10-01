#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b}"
OUTCOME_ROOT="${OUTCOME_ROOT:-${EXPERIMENT_ROOT}/outcomes}"
PREDICTION_DATASET_DIR="${PREDICTION_DATASET_DIR:-${EXPERIMENT_ROOT}/prediction_dataset}"
PREDICTION_OUTPUT_DIR="${PREDICTION_OUTPUT_DIR:-${EXPERIMENT_ROOT}/prediction_analysis}"

RUN_OUTCOMES="${RUN_OUTCOMES:-1}"
RUN_DATASET_BUILD="${RUN_DATASET_BUILD:-1}"
RUN_PREDICTION="${RUN_PREDICTION:-1}"

MAX_RECORDS="${MAX_RECORDS:-3000}"
RECORD_OFFSET="${RECORD_OFFSET:-0}"
RECORD_IDS_FILE="${RECORD_IDS_FILE:-}"
METHODS="${METHODS:-random matched_norm_random mean_difference logistic linear rfm_agop_top1}"
LAYERS="${LAYERS:--21 -17}"
ALPHAS="${ALPHAS:--0.5 -0.25 -0.1 0 0.1 0.25 0.5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-113}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

RFM_ITERS="${RFM_ITERS:-3}"
RFM_REG="${RFM_REG:-1e-3}"
RFM_BANDWIDTH="${RFM_BANDWIDTH:-10.0}"
RFM_KERNEL="${RFM_KERNEL:-laplace}"
RFM_MEM_GB="${RFM_MEM_GB:-8.0}"

NULL_QUANTILE="${NULL_QUANTILE:-0.95}"
EARLY_ALPHA="${EARLY_ALPHA:-0.1}"
MEDIUM_ALPHA="${MEDIUM_ALPHA:-0.5}"
PREDICTION_SPLITS="${PREDICTION_SPLITS:-record dataset domain}"
PREDICTION_COHORTS="${PREDICTION_COHORTS:-all learned rfm}"
PREDICTION_MODELS="${PREDICTION_MODELS:-logistic random_forest}"
PREDICTION_FOLDS="${PREDICTION_FOLDS:-5}"
PREDICTION_MAX_RECORDS="${PREDICTION_MAX_RECORDS:-}"

mkdir -p "${EXPERIMENT_ROOT}" "${OUTCOME_ROOT}"

trap 'echo; echo "Interrupted. Rerun the same command to resume from raw_scores.jsonl and fold_results.jsonl." >&2' INT TERM

if [[ "${RUN_OUTCOMES}" == "1" && "${DEVICE_MAP}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot see CUDA.", file=sys.stderr)
    print(f"torch_version={torch.__version__}", file=sys.stderr)
    print(f"cuda_device_count={torch.cuda.device_count()}", file=sys.stderr)
    raise SystemExit(2)
print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}")
PY
fi

cat > "${EXPERIMENT_ROOT}/main_experiment_config.env" <<EOF
DATA_DIR=${DATA_DIR}
MODEL_PATH=${MODEL_PATH}
MODEL_NAME=${MODEL_NAME}
MAX_RECORDS=${MAX_RECORDS}
RECORD_OFFSET=${RECORD_OFFSET}
RECORD_IDS_FILE=${RECORD_IDS_FILE}
METHODS=${METHODS}
LAYERS=${LAYERS}
ALPHAS=${ALPHAS}
BATCH_SIZE=${BATCH_SIZE}
SEED=${SEED}
TORCH_DTYPE=${TORCH_DTYPE}
DEVICE_MAP=${DEVICE_MAP}
NULL_QUANTILE=${NULL_QUANTILE}
EARLY_ALPHA=${EARLY_ALPHA}
MEDIUM_ALPHA=${MEDIUM_ALPHA}
EOF

echo "[PML main] experiment_root=${EXPERIMENT_ROOT}"
echo "[PML main] records=${MAX_RECORDS} methods=${METHODS} layers=${LAYERS} alphas=${ALPHAS}"

if [[ "${RUN_OUTCOMES}" == "1" ]]; then
  EXPECTED_RECORDS="${MAX_RECORDS}"
  if [[ -n "${RECORD_IDS_FILE}" ]]; then
    EXPECTED_RECORDS="$(awk 'NF {count += 1} END {print count + 0}' "${RECORD_IDS_FILE}")"
    if (( EXPECTED_RECORDS > MAX_RECORDS )); then
      EXPECTED_RECORDS="${MAX_RECORDS}"
    fi
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

      RECORD_ARGS=()
      if [[ -n "${RECORD_IDS_FILE}" ]]; then
        RECORD_ARGS+=(--record-ids-file "${RECORD_IDS_FILE}")
      fi

      echo "[outcome] method=${METHOD} layer=${LAYER} out=${OUT_DIR} mode=${RUN_MODE_ARGS[*]}"
      "${PYTHON_BIN}" -m predictive_memory_localization.methods.run_bidirectional_three_table \
        --data-dir "${DATA_DIR}" \
        --model-local-path "${MODEL_PATH}" \
        --model-name "${MODEL_NAME}" \
        --output-dir "${OUT_DIR}" \
        --control-method "${METHOD}" \
        --max-records "${MAX_RECORDS}" \
        --record-offset "${RECORD_OFFSET}" \
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
        "${RECORD_ARGS[@]}" \
        "${RUN_MODE_ARGS[@]}"
    done
  done
fi

if [[ "${RUN_DATASET_BUILD}" == "1" ]]; then
  BUILD_ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    BUILD_ARGS+=(--overwrite)
  fi
  echo "[dataset] building prediction dataset"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_activation_space_pml_prediction_dataset \
    --stage-root "${OUTCOME_ROOT}" \
    --output-dir "${PREDICTION_DATASET_DIR}" \
    --random-method random \
    --null-quantile "${NULL_QUANTILE}" \
    --early-alpha "${EARLY_ALPHA}" \
    --medium-alpha "${MEDIUM_ALPHA}" \
    "${BUILD_ARGS[@]}"
fi

if [[ "${RUN_PREDICTION}" == "1" ]]; then
  PREDICTION_ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    PREDICTION_ARGS+=(--overwrite)
  fi
  if [[ -n "${PREDICTION_MAX_RECORDS}" ]]; then
    PREDICTION_ARGS+=(--max-records "${PREDICTION_MAX_RECORDS}")
  fi
  echo "[prediction] running grouped PML ablations"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_activation_space_pml_main_prediction \
    --dataset-dir "${PREDICTION_DATASET_DIR}" \
    --output-dir "${PREDICTION_OUTPUT_DIR}" \
    --splits ${PREDICTION_SPLITS} \
    --cohorts ${PREDICTION_COHORTS} \
    --models ${PREDICTION_MODELS} \
    --n-splits "${PREDICTION_FOLDS}" \
    --seed "${SEED}" \
    "${PREDICTION_ARGS[@]}"
fi

echo "[PML main] complete"
echo "Outcome root: ${OUTCOME_ROOT}"
echo "Prediction dataset: ${PREDICTION_DATASET_DIR}/pml_path_prediction_dataset.csv"
echo "Prediction report: ${PREDICTION_OUTPUT_DIR}/PML_MAIN_PREDICTION_REPORT.md"
