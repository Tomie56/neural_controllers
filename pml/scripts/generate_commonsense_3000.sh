#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
TARGET_RECORDS="${TARGET_RECORDS:-3000}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
TEMPERATURE="${TEMPERATURE:-0.1}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.05}"
WORKERS="${WORKERS:-6}"
OVERWRITE_SEEDS="${OVERWRITE_SEEDS:-0}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
RESUME="${RESUME:-1}"

OUT_DIR="${OUT_DIR:-/data/neural_controllers/pml/data}"
SEED_JSONL="${SEED_JSONL:-${OUT_DIR}/commonsense_seed_mix_${TARGET_RECORDS}.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUT_DIR}/suppression_commonsense_${TARGET_RECORDS}.jsonl}"
FAILURES_JSONL="${FAILURES_JSONL:-${OUT_DIR}/suppression_commonsense_${TARGET_RECORDS}_failures.jsonl}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

SEED_ARGS=()
if [[ "${OVERWRITE_SEEDS}" == "1" || ! -s "${SEED_JSONL}" ]]; then
  SEED_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" pml/src/predictive_memory_localization/data/build_commonsense_seed_mix.py \
  --output-jsonl "${SEED_JSONL}" \
  --target-records "${TARGET_RECORDS}" \
  --seed "${SEED}" \
  "${SEED_ARGS[@]}"

GEN_ARGS=()
if [[ "${OVERWRITE_OUTPUT}" == "1" ]]; then
  GEN_ARGS+=(--overwrite)
elif [[ "${RESUME}" == "1" ]]; then
  GEN_ARGS+=(--resume)
fi

"${PYTHON_BIN}" pml/src/predictive_memory_localization/data/generate_suppression_data_openrouter.py \
  --seed-jsonl "${SEED_JSONL}" \
  --dataset-name commonsense_seed_mix \
  --dataset-config "" \
  --split train \
  --max-records "${TARGET_RECORDS}" \
  --seed "${SEED}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --failures-jsonl "${FAILURES_JSONL}" \
  --model "${MODEL}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --sleep "${SLEEP_SECONDS}" \
  --workers "${WORKERS}" \
  "${GEN_ARGS[@]}"

"${PYTHON_BIN}" pml/src/predictive_memory_localization/data/validate_suppression_data.py \
  --input-jsonl "${OUTPUT_JSONL}" \
  --output-json "${OUTPUT_JSONL%.jsonl}_validation.json"

echo "Seed data: ${SEED_JSONL}"
echo "Generated suppression data: ${OUTPUT_JSONL}"
echo "Failures, if any: ${FAILURES_JSONL}"
