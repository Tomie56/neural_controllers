#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"

# Raw Hugging Face data is intentionally kept in shared memory for fast local preprocessing.
RAW_ROOT="${RAW_ROOT:-/dev/shm/pml_datas}"

# Final normalized PML data goes under the repo data directory.
OUT_PREFIX="${OUT_PREFIX:-/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000}"
SEED_JSONL="${SEED_JSONL:-${OUT_PREFIX}.seeds.jsonl}"

TARGET_RECORDS="${TARGET_RECORDS:-3000}"
SEED="${SEED:-113}"

MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
WORKERS="${WORKERS:-4}"
TEMPERATURE="${TEMPERATURE:-0.15}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-2800}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.1}"

# Overwrite controls.
# First full run:
#   OVERWRITE_SEEDS=1 OVERWRITE_OUTPUT=1 ./pml/scripts/run_fresh_bidirectional_data_pipeline.sh
# Resume OpenRouter generation after interruption:
#   RESUME=1 ./pml/scripts/run_fresh_bidirectional_data_pipeline.sh
OVERWRITE_RAW="${OVERWRITE_RAW:-0}"
OVERWRITE_SEEDS="${OVERWRITE_SEEDS:-0}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
RESUME="${RESUME:-1}"
SKIP_DOWNLOAD_ERRORS="${SKIP_DOWNLOAD_ERRORS:-1}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  echo "Run: export OPENROUTER_API_KEY=your_key" >&2
  exit 1
fi

mkdir -p "${RAW_ROOT}" "$(dirname "${OUT_PREFIX}")"

echo "[1/3] Downloading raw Hugging Face datasets to ${RAW_ROOT}"
DOWNLOAD_ARGS=()
if [[ "${OVERWRITE_RAW}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--overwrite)
fi
if [[ "${SKIP_DOWNLOAD_ERRORS}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--skip-errors)
fi

"${PYTHON_BIN}" -m predictive_memory_localization.data.download_pml_hf_sources \
  --raw-root "${RAW_ROOT}" \
  "${DOWNLOAD_ARGS[@]}"

echo "[2/3] Building ${TARGET_RECORDS} normalized seed rows at ${SEED_JSONL}"
SEED_ARGS=()
if [[ "${OVERWRITE_SEEDS}" == "1" || ! -s "${SEED_JSONL}" ]]; then
  SEED_ARGS+=(--overwrite)
else
  echo "Seed file exists; reusing ${SEED_JSONL}"
fi

if [[ "${#SEED_ARGS[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" -m predictive_memory_localization.data.build_pml_fresh_seed_pool \
    --raw-root "${RAW_ROOT}" \
    --output-jsonl "${SEED_JSONL}" \
    --target-records "${TARGET_RECORDS}" \
    --seed "${SEED}" \
    --backfill \
    "${SEED_ARGS[@]}"
fi

echo "[3/3] Generating normalized PML bidirectional records/eval bank with OpenRouter"
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
  --top-p "${TOP_P}" \
  --max-tokens "${MAX_TOKENS}" \
  --sleep "${SLEEP_SECONDS}" \
  "${GEN_ARGS[@]}"

echo
echo "Done."
echo "Raw HF root: ${RAW_ROOT}"
echo "Seeds: ${SEED_JSONL}"
echo "Records: ${OUT_PREFIX}.records.jsonl"
echo "Eval bank: ${OUT_PREFIX}.eval_bank.jsonl"
echo "Assignments: ${OUT_PREFIX}.eval_assignments.jsonl"
echo "Build bundles checkpoint: ${OUT_PREFIX}.build_bundles.jsonl"
echo "Failures: ${OUT_PREFIX}.failures.jsonl"
echo "Report: ${OUT_PREFIX}.report.json"
