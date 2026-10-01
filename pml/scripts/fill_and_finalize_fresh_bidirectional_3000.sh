#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"

BASE_PREFIX="${BASE_PREFIX:-/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000}"
FREEZE_PREFIX="${FREEZE_PREFIX:-/data/neural_controllers/pml/data/main_datas/pml_fresh_multidomain_3000_v1}"
SEED_JSONL="${SEED_JSONL:-${BASE_PREFIX}.seeds.jsonl}"
BUNDLES_JSONL="${BUNDLES_JSONL:-${BASE_PREFIX}.build_bundles.jsonl}"
RAW_ROOT="${RAW_ROOT:-/dev/shm/pml_datas}"

TARGET_RECORDS="${TARGET_RECORDS:-3000}"
SEED_TARGET_RECORDS="${SEED_TARGET_RECORDS:-3300}"
SEED="${SEED:-113}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
WORKERS="${WORKERS:-4}"
TEMPERATURE="${TEMPERATURE:-0.12}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-3200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.2}"
RETRIES="${RETRIES:-6}"

# Set RETRY_OPENROUTER=0 to only re-materialize/finalize existing bundles.
RETRY_OPENROUTER="${RETRY_OPENROUTER:-1}"
EXPAND_SEEDS="${EXPAND_SEEDS:-1}"
OVERWRITE_FREEZE="${OVERWRITE_FREEZE:-1}"

# Optional alpha=0 baseline margin QA. Leave empty to skip because it loads a model.
MARGIN_SANITY_MODEL_PATH="${MARGIN_SANITY_MODEL_PATH:-}"
MARGIN_SANITY_SAMPLE="${MARGIN_SANITY_SAMPLE:-300}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"

if [[ "${RETRY_OPENROUTER}" == "1" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set, but RETRY_OPENROUTER=1." >&2
  echo "Run: export OPENROUTER_API_KEY=your_key" >&2
  exit 1
fi

if [[ ! -s "${SEED_JSONL}" ]]; then
  echo "ERROR: missing seed file: ${SEED_JSONL}" >&2
  exit 1
fi

CURRENT_SEED_ROWS="$(wc -l < "${SEED_JSONL}")"
if [[ "${EXPAND_SEEDS}" == "1" && "${CURRENT_SEED_ROWS}" -lt "${SEED_TARGET_RECORDS}" ]]; then
  if [[ -d "${RAW_ROOT}/raw" ]]; then
    echo "[0/2] Expanding seed candidates from ${CURRENT_SEED_ROWS} to ${SEED_TARGET_RECORDS}"
    "${PYTHON_BIN}" -m predictive_memory_localization.data.build_pml_fresh_seed_pool \
      --raw-root "${RAW_ROOT}" \
      --output-jsonl "${SEED_JSONL}" \
      --target-records "${SEED_TARGET_RECORDS}" \
      --seed "${SEED}" \
      --backfill \
      --overwrite
  else
    echo "[0/2] WARNING: ${RAW_ROOT}/raw is missing; cannot expand seeds. Using ${CURRENT_SEED_ROWS} seeds." >&2
    SEED_TARGET_RECORDS="${CURRENT_SEED_ROWS}"
  fi
else
  echo "[0/2] Seed candidates: ${CURRENT_SEED_ROWS}"
fi

if [[ "${RETRY_OPENROUTER}" == "1" ]]; then
  echo "[1/2] Retrying missing/failed OpenRouter generations into ${BUNDLES_JSONL}"
  "${PYTHON_BIN}" -m predictive_memory_localization.data.generate_pml_bidirectional_data_openrouter \
    --seed-jsonl "${SEED_JSONL}" \
    --output-prefix "${BASE_PREFIX}" \
    --max-records "${SEED_TARGET_RECORDS}" \
    --seed "${SEED}" \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-tokens "${MAX_TOKENS}" \
    --sleep "${SLEEP_SECONDS}" \
    --retries "${RETRIES}" \
    --resume
else
  echo "[1/2] Skipping OpenRouter retry; using existing bundles ${BUNDLES_JSONL}"
fi

echo "[2/2] Finalizing frozen v1 data at ${FREEZE_PREFIX}"
FINALIZE_ARGS=()
if [[ "${OVERWRITE_FREEZE}" == "1" ]]; then
  FINALIZE_ARGS+=(--overwrite)
fi
if [[ -n "${MARGIN_SANITY_MODEL_PATH}" ]]; then
  FINALIZE_ARGS+=(
    --margin-sanity-model-path "${MARGIN_SANITY_MODEL_PATH}"
    --margin-sanity-sample "${MARGIN_SANITY_SAMPLE}"
    --torch-dtype "${TORCH_DTYPE}"
    --device-map "${DEVICE_MAP}"
  )
fi

"${PYTHON_BIN}" -m predictive_memory_localization.data.finalize_pml_bidirectional_data \
  --bundles-jsonl "${BUNDLES_JSONL}" \
  --output-prefix "${FREEZE_PREFIX}" \
  --target-records "${TARGET_RECORDS}" \
  --seed "${SEED}" \
  "${FINALIZE_ARGS[@]}"

echo
echo "Frozen records: ${FREEZE_PREFIX}.records.jsonl"
echo "Frozen eval bank: ${FREEZE_PREFIX}.eval_bank.jsonl"
echo "Frozen assignments: ${FREEZE_PREFIX}.eval_assignments.jsonl"
echo "Validation: ${FREEZE_PREFIX}.validation.json"
echo "Human QA sample: ${FREEZE_PREFIX}.sample_report.jsonl"
echo "Rejected bundles: ${FREEZE_PREFIX}.rejected_bundles.jsonl"
if [[ -n "${MARGIN_SANITY_MODEL_PATH}" ]]; then
  echo "Margin sanity: ${FREEZE_PREFIX}.margin_sanity.json"
fi
