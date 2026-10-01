#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
FORMAL_IDS="${FORMAL_IDS:-/data/neural_controllers/pml/data/pml_replication_500_stratified_seed113/record_ids.txt}"
CALIBRATION_SELECTION_DIR="${CALIBRATION_SELECTION_DIR:-/data/neural_controllers/pml/data/pml_scale_calibration_100_seed127}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/scale_calibration_residual_rms_v2}"
SCALE_MANIFEST="${SCALE_MANIFEST:-${CALIBRATION_ROOT}/residual_rms_reference_scale.json}"

QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
QWEN35_MODEL_PATH="${QWEN35_MODEL_PATH:-/data/neural_controllers/model/Qwen3.5-2B-Base}"
MINISTRAL_MODEL_PATH="${MINISTRAL_MODEL_PATH:-/data/neural_controllers/model/Ministral-3-3B-Base-2512}"

N_CALIBRATION_RECORDS="${N_CALIBRATION_RECORDS:-100}"
SEED="${SEED:-127}"
REFERENCE_NUM_LAYERS="${REFERENCE_NUM_LAYERS:-28}"
REFERENCE_LAYERS="${REFERENCE_LAYERS:--21 -17}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${CALIBRATION_ROOT}"
trap 'echo; echo "Interrupted. Rerun the same command to resume calibration." >&2' INT TERM

SELECT_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  SELECT_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" -m predictive_memory_localization.analysis.select_disjoint_scale_calibration_records \
  --data-dir "${DATA_DIR}" \
  --exclude-record-ids-file "${FORMAL_IDS}" \
  --output-dir "${CALIBRATION_SELECTION_DIR}" \
  --n-records "${N_CALIBRATION_RECORDS}" \
  --seed "${SEED}" \
  "${SELECT_ARGS[@]}"

CALIBRATION_IDS="${CALIBRATION_SELECTION_DIR}/record_ids.txt"

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

resolve_mapping() {
  local model_path="$1"
  local output_json="$2"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.resolve_relative_intervention_layers \
    --model-path "${model_path}" \
    --reference-num-layers "${REFERENCE_NUM_LAYERS}" \
    --reference-layers ${REFERENCE_LAYERS} \
    --output-json "${output_json}" \
    --format shell
}

measure_model() {
  local model_key="$1"
  local model_path="$2"
  local model_loader="$3"
  local batch_size="$4"
  local mapping_json="$5"
  local output_dir="${CALIBRATION_ROOT}/${model_key}"
  local summary_path="${output_dir}/residual_rms_summary.json"
  local layers
  layers="$(${PYTHON_BIN} -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["layers"])))' "${mapping_json}")"
  if [[ -s "${summary_path}" && "${OVERWRITE}" != "1" ]]; then
    echo "[RMS calibration] skip complete model=${model_key} summary=${summary_path}"
    return
  fi
  local run_mode=(--overwrite)
  if [[ "${OVERWRITE}" != "1" && -s "${output_dir}/residual_rms_records.jsonl" ]]; then
    run_mode=(--resume)
  fi
  echo "[RMS calibration] model=${model_key} layers=${layers} mode=${run_mode[*]}"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.measure_residual_rms_calibration \
    --data-dir "${DATA_DIR}" \
    --record-ids-file "${CALIBRATION_IDS}" \
    --model-local-path "${model_path}" \
    --model-name "${model_key}" \
    --output-dir "${output_dir}" \
    --layers ${layers} \
    --batch-size "${batch_size}" \
    --seed "${SEED}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --model-loader "${model_loader}" \
    "${run_mode[@]}"
}

QWEN3_MAPPING="${CALIBRATION_ROOT}/qwen3_1_7b/relative_layer_mapping.json"
QWEN35_MAPPING="${CALIBRATION_ROOT}/qwen3_5_2b_base/relative_layer_mapping.json"
MINISTRAL_MAPPING="${CALIBRATION_ROOT}/ministral_3_3b_base_2512/relative_layer_mapping.json"

resolve_mapping "${QWEN3_MODEL_PATH}" "${QWEN3_MAPPING}" >/dev/null
resolve_mapping "${QWEN35_MODEL_PATH}" "${QWEN35_MAPPING}" >/dev/null
resolve_mapping "${MINISTRAL_MODEL_PATH}" "${MINISTRAL_MAPPING}" >/dev/null

measure_model qwen3_1_7b "${QWEN3_MODEL_PATH}" causal_lm 4 "${QWEN3_MAPPING}"
measure_model qwen3_5_2b_base "${QWEN35_MODEL_PATH}" image_text_to_text 2 "${QWEN35_MAPPING}"
measure_model ministral_3_3b_base_2512 "${MINISTRAL_MODEL_PATH}" image_text_to_text 1 "${MINISTRAL_MAPPING}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_residual_rms_reference_scale \
  --reference-model-key qwen3_1_7b \
  --reference-summary "${CALIBRATION_ROOT}/qwen3_1_7b/residual_rms_summary.json" \
  --reference-mapping "${QWEN3_MAPPING}" \
  --target-model-keys qwen3_5_2b_base ministral_3_3b_base_2512 \
  --target-summaries \
    "${CALIBRATION_ROOT}/qwen3_5_2b_base/residual_rms_summary.json" \
    "${CALIBRATION_ROOT}/ministral_3_3b_base_2512/residual_rms_summary.json" \
  --target-mappings "${QWEN35_MAPPING}" "${MINISTRAL_MAPPING}" \
  --reference-alphas 0.1 0.25 0.5 \
  --output-json "${SCALE_MANIFEST}" \
  --overwrite

echo "[RMS calibration] complete: ${SCALE_MANIFEST}"
