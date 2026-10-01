#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"

GRADIENT_ROOT="${GRADIENT_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2e_gradient_response/qwen3_1_7b}"
PLAN_ROOT="${PLAN_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2g_adaptive_alpha_plan/qwen3_1_7b}"
ROOT_OUT="${ROOT_OUT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2g_adaptive_alpha_scan/qwen3_1_7b}"
DENSE_ROOT="${DENSE_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b}"

MAX_RECORDS="${MAX_RECORDS:-500}"
RECORD_OFFSET="${RECORD_OFFSET:-0}"
TAU="${TAU:-0.1522890289624531}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-113}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
OVERWRITE="${OVERWRITE:-0}"
BUILD_PLAN="${BUILD_PLAN:-1}"
VALIDATE_AGAINST_DENSE="${VALIDATE_AGAINST_DENSE:-1}"

METHODS_FILTER="${METHODS_FILTER:-}"
LAYERS_FILTER="${LAYERS_FILTER:-}"

if [[ "${BUILD_PLAN}" == "1" ]]; then
  echo "[Stage 2g] Building per-record adaptive alpha plans"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_adaptive_alpha_plan_from_gradient \
    --gradient-root "${GRADIENT_ROOT}" \
    --output-dir "${PLAN_ROOT}" \
    --tau "${TAU}"
fi

if [[ ! -d "${PLAN_ROOT}" ]]; then
  echo "ERROR: PLAN_ROOT not found: ${PLAN_ROOT}" >&2
  echo "Run Stage 2e first, then rerun with BUILD_PLAN=1." >&2
  exit 2
fi

if [[ "${DEVICE_MAP}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot see CUDA. Run this script from a shell with GPU access.", file=sys.stderr)
    print(f"torch_version={torch.__version__}", file=sys.stderr)
    print(f"cuda_device_count={torch.cuda.device_count()}", file=sys.stderr)
    raise SystemExit(2)
print(f"CUDA preflight OK: {torch.cuda.get_device_name(0)}")
PY
fi

mkdir -p "${ROOT_OUT}"
PLAN_TSV="${ROOT_OUT}/adaptive_plan_rows.tsv"
export PLAN_ROOT PLAN_TSV METHODS_FILTER LAYERS_FILTER

"${PYTHON_BIN}" -c '
import os
from pathlib import Path

plan_root = Path(os.environ["PLAN_ROOT"])
out_path = Path(os.environ["PLAN_TSV"])
methods_filter = {x for x in os.environ.get("METHODS_FILTER", "").split() if x}
layers_filter = {x for x in os.environ.get("LAYERS_FILTER", "").split() if x}
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for path in sorted(plan_root.glob("*/*/adaptive_alpha_plan.jsonl")):
        method = path.parts[-3]
        layer_name = path.parts[-2]
        layer = layer_name.replace("layer_neg", "-").replace("layer_", "")
        if methods_filter and method not in methods_filter:
            continue
        if layers_filter and layer not in layers_filter and layer_name not in layers_filter:
            continue
        f.write(f"{method}\t{layer}\t{layer_name}\t{path}\n")
print(f"Wrote adaptive plan TSV to {out_path}")
'

if [[ ! -s "${PLAN_TSV}" ]]; then
  echo "ERROR: no adaptive plan rows after filters." >&2
  exit 2
fi

echo "Stage 2g adaptive alpha scan"
echo "GRADIENT_ROOT=${GRADIENT_ROOT}"
echo "PLAN_ROOT=${PLAN_ROOT}"
echo "ROOT_OUT=${ROOT_OUT}"
echo "MAX_RECORDS=${MAX_RECORDS}"
echo "RECORD_OFFSET=${RECORD_OFFSET}"
echo "TAU=${TAU}"

while IFS=$'\t' read -r METHOD LAYER LAYER_NAME PLAN_JSONL; do
  OUT_DIR="${ROOT_OUT}/${METHOD}/${LAYER_NAME}"
  RUN_MODE_ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    RUN_MODE_ARGS+=(--overwrite)
  elif [[ -s "${OUT_DIR}/raw_scores.jsonl" ]]; then
    RUN_MODE_ARGS+=(--resume)
  else
    RUN_MODE_ARGS+=(--overwrite)
  fi

  echo "[Stage 2g adaptive scan] method=${METHOD} layer=${LAYER} plan=${PLAN_JSONL} out=${OUT_DIR} mode=${RUN_MODE_ARGS[*]}"
  "${PYTHON_BIN}" -m predictive_memory_localization.methods.run_bidirectional_three_table \
    --data-dir "${DATA_DIR}" \
    --model-local-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUT_DIR}" \
    --control-method "${METHOD}" \
    --max-records "${MAX_RECORDS}" \
    --record-offset "${RECORD_OFFSET}" \
    --layers "${LAYER}" \
    --alpha-plan-jsonl "${PLAN_JSONL}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --target-threshold "${TAU}" \
    --neighbor-threshold "${TAU}" \
    --capability-threshold "${TAU}" \
    "${RUN_MODE_ARGS[@]}"
done < "${PLAN_TSV}"

if [[ "${VALIDATE_AGAINST_DENSE}" == "1" ]]; then
  echo "[Stage 2g] Comparing adaptive scan with existing dense curves"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.compare_reduced_scan_to_dense_curve \
    --reduced-root "${ROOT_OUT}" \
    --dense-root "${DENSE_ROOT}" \
    --output-dir "${ROOT_OUT}/validation_vs_stage2_dense" \
    --tau "${TAU}"
fi

echo "Stage 2g adaptive alpha scan root: ${ROOT_OUT}"
