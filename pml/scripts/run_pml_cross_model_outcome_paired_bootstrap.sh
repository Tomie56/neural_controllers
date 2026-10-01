#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
REPLICATION_ROOT="${REPLICATION_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/replication_500_residual_rms_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPLICATION_ROOT}/cross_model_outcome_paired_bootstrap}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-10000}"
SEED="${SEED:-113}"
OVERWRITE="${OVERWRITE:-0}"

ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[cross-model outcome CI] replication_root=${REPLICATION_ROOT}"
echo "[cross-model outcome CI] output_dir=${OUTPUT_DIR}"
echo "[cross-model outcome CI] bootstrap_reps=${BOOTSTRAP_REPS} seed=${SEED}"

"${PYTHON_BIN}" -m predictive_memory_localization.analysis.analyze_cross_model_outcome_paired_bootstrap \
  --replication-root "${REPLICATION_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-reps "${BOOTSTRAP_REPS}" \
  --seed "${SEED}" \
  "${ARGS[@]}"

echo "[cross-model outcome CI] complete"
echo "Report: ${OUTPUT_DIR}/CROSS_MODEL_OUTCOME_PAIRED_BOOTSTRAP_REPORT.md"
echo "Results: ${OUTPUT_DIR}/cross_model_outcome_paired_bootstrap.csv"
echo "Manifest: ${OUTPUT_DIR}/cross_model_outcome_paired_bootstrap_manifest.json"
