#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
DATA_DIR="${DATA_DIR:-/data/neural_controllers/pml/data/pml_fresh_multidomain_3000_v1_frozen}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-qwen3_1_7b}"
OUT_DIR="${OUT_DIR:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage0_margin_sanity/qwen3_1_7b}"
SAMPLE_PER_TYPE="${SAMPLE_PER_TYPE:-200}"
SAMPLE_SIZE="${SAMPLE_SIZE:-600}"
SEED="${SEED:-113}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"

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

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_margin_sanity_three_table \
  --data-dir "${DATA_DIR}" \
  --model-local-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUT_DIR}" \
  --sample-per-type "${SAMPLE_PER_TYPE}" \
  --sample-size "${SAMPLE_SIZE}" \
  --seed "${SEED}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --device-map "${DEVICE_MAP}" \
  --overwrite
