#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
RAW_ROOT="${RAW_ROOT:-/dev/shm/pml_datas}"
OUT_PREFIX="${OUT_PREFIX:-/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000}"
TARGET_RECORDS="${TARGET_RECORDS:-3000}"
SEED="${SEED:-113}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
WORKERS="${WORKERS:-4}"
TEMPERATURE="${TEMPERATURE:-0.15}"
MAX_TOKENS="${MAX_TOKENS:-2800}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.1}"
OVERWRITE_RAW="${OVERWRITE_RAW:-0}"
OVERWRITE_SEEDS="${OVERWRITE_SEEDS:-0}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
RESUME="${RESUME:-1}"

SEED_JSONL="${SEED_JSONL:-${OUT_PREFIX}.seeds.jsonl}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

DOWNLOAD_ARGS=()
if [[ "${OVERWRITE_RAW}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" -m predictive_memory_localization.data.download_pml_hf_sources \
  --raw-root "${RAW_ROOT}" \
  --skip-errors \
  "${DOWNLOAD_ARGS[@]}"

SEED_ARGS=()
if [[ "${OVERWRITE_SEEDS}" == "1" || ! -s "${SEED_JSONL}" ]]; then
  SEED_ARGS+=(--overwrite)
else
  echo "Reusing existing seed file: ${SEED_JSONL}"
fi

if [[ "${#SEED_ARGS[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" -m predictive_memory_localization.data.build_pml_fresh_seed_pool \
    --raw-root "${RAW_ROOT}" \
    --output-jsonl "${SEED_JSONL}" \
    --target-records "${TARGET_RECORDS}" \
    --seed "${SEED}" \
    "${SEED_ARGS[@]}"
fi

GEN_ARGS=()
if [[ "${OVERWRITE_OUTPUT}" == "1" ]]; then
  GEN_ARGS+=(--overwrite)
elif [[ "${RESUME}" == "1" ]]; then
  GEN_ARGS+=(--resume)
fi

"${PYTHON_BIN}" -m predictive_memory_localization.data.generate_pml_bidirectional_data_openrouter \
  --seed-jsonl "${SEED_JSONL}" \
  --output-prefix "${OUT_PREFIX}" \
  --max-records "${TARGET_RECORDS}" \
  --seed "${SEED}" \
  --model "${MODEL}" \
  --workers "${WORKERS}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --sleep "${SLEEP_SECONDS}" \
  "${GEN_ARGS[@]}"

echo "Seeds: ${SEED_JSONL}"
echo "Records: ${OUT_PREFIX}.records.jsonl"
echo "Eval bank: ${OUT_PREFIX}.eval_bank.jsonl"
echo "Assignments: ${OUT_PREFIX}.eval_assignments.jsonl"
echo "Report: ${OUT_PREFIX}.report.json"
