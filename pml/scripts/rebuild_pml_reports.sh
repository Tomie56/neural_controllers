#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"

PATH_ROWS_CSV="${PATH_ROWS_CSV:-pml/results/path_geometry_audit/strength_path_rows.csv}"
EDITING_ROWS_CSV="${EDITING_ROWS_CSV:-pml/results/commonsenseqa_balanced_followup_144/rome/superficial_editing_analysis/superficial_editing_rows.csv}"
PATH_EDITING_TRANSFER_DIR="${PATH_EDITING_TRANSFER_DIR:-pml/results/commonsenseqa_balanced_followup_144/rome/path_editing_transfer}"
DATASET_LABEL="${DATASET_LABEL:-heldout_144_agop_direct}"
METHODS="${METHODS:-agop_top1 agop_topk_project}"

CAPABILITY_RESULT_ROOT="${CAPABILITY_RESULT_ROOT:-pml/results/path_representative_50_capability_onset_probe}"
CAPABILITY_METHODS="${CAPABILITY_METHODS:-mean_difference logistic}"

DENSE_ALPHA_PATH_ROWS_CSV="${DENSE_ALPHA_PATH_ROWS_CSV:-pml/results/heldout_100_dense_alpha_baselines/path_geometry_audit/strength_path_rows.csv}"
DENSE_ALPHA_DATASET_LABEL="${DENSE_ALPHA_DATASET_LABEL:-heldout_100_dense_alpha_baselines}"
DENSE_ALPHA_MIN_STRENGTH_DIR="${DENSE_ALPHA_MIN_STRENGTH_DIR:-pml/results/heldout_100_dense_alpha_baselines/minimum_effective_strength}"

REBUILD_TRANSFER="${REBUILD_TRANSFER:-1}"
REBUILD_STABILITY="${REBUILD_STABILITY:-0}"
STABILITY_BOOTSTRAP="${STABILITY_BOOTSTRAP:-300}"
STABILITY_PERMUTATIONS="${STABILITY_PERMUTATIONS:-300}"

echo "PML report rebuild started."
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "REBUILD_TRANSFER=${REBUILD_TRANSFER}"
echo "REBUILD_STABILITY=${REBUILD_STABILITY}"

if [[ "${REBUILD_TRANSFER}" == "1" ]]; then
  if [[ ! -s "${PATH_ROWS_CSV}" ]]; then
    echo "ERROR: missing path rows: ${PATH_ROWS_CSV}" >&2
    exit 1
  fi
  if [[ ! -s "${EDITING_ROWS_CSV}" ]]; then
    echo "ERROR: missing editing rows: ${EDITING_ROWS_CSV}" >&2
    exit 1
  fi
  read -r -a METHOD_ARGS <<< "${METHODS}"
  echo "Rebuilding path-to-editing transfer..."
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_path_editing_transfer.py \
    --path-rows-csv "${PATH_ROWS_CSV}" \
    --editing-rows-csv "${EDITING_ROWS_CSV}" \
    --output-dir "${PATH_EDITING_TRANSFER_DIR}" \
    --dataset-label "${DATASET_LABEL}" \
    --methods "${METHOD_ARGS[@]}"
fi

if [[ "${REBUILD_STABILITY}" == "1" ]]; then
  echo "Rebuilding path-to-editing stability checks..."
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_path_editing_transfer_stability.py \
    --joined-csv "${PATH_EDITING_TRANSFER_DIR}/path_editing_joined_rows.csv" \
    --output-dir "${PATH_EDITING_TRANSFER_DIR}/stability" \
    --n-bootstrap "${STABILITY_BOOTSTRAP}" \
    --n-permutations "${STABILITY_PERMUTATIONS}"
else
  echo "Skipping stability bootstrap. Set REBUILD_STABILITY=1 to refresh it."
fi

if [[ -s "${PATH_EDITING_TRANSFER_DIR}/path_editing_joined_rows.csv" ]]; then
  echo "Rebuilding path-to-editing case audit..."
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_path_editing_case_audit.py \
    --joined-csv "${PATH_EDITING_TRANSFER_DIR}/path_editing_joined_rows.csv" \
    --output-dir "${PATH_EDITING_TRANSFER_DIR}/case_audit"
else
  echo "WARNING: missing joined rows; skipping case audit: ${PATH_EDITING_TRANSFER_DIR}/path_editing_joined_rows.csv" >&2
fi

CAPABILITY_READY=1
read -r -a CAPABILITY_METHOD_ARGS <<< "${CAPABILITY_METHODS}"
for METHOD in "${CAPABILITY_METHOD_ARGS[@]}"; do
  if [[ ! -s "${CAPABILITY_RESULT_ROOT}/${METHOD}/suppression_results.jsonl" ]]; then
    CAPABILITY_READY=0
  fi
done
if [[ "${CAPABILITY_READY}" == "1" ]]; then
  echo "Capability-onset results found; rebuilding representative capability analysis..."
  bash pml/scripts/analyze_path_representative_capability_probe.sh
else
  echo "Capability-onset scoring results are incomplete; skipping representative capability analysis."
fi

echo "Refreshing representative capability status..."
"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/check_path_representative_probe_status.py

if [[ -s "${DENSE_ALPHA_PATH_ROWS_CSV}" ]]; then
  echo "Rebuilding dense-alpha minimum effective strength..."
  "${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_minimum_effective_strength.py \
    --path-rows-csv "${DENSE_ALPHA_PATH_ROWS_CSV}" \
    --dataset-label "${DENSE_ALPHA_DATASET_LABEL}" \
    --output-dir "${DENSE_ALPHA_MIN_STRENGTH_DIR}"
else
  echo "WARNING: missing dense-alpha path rows; skipping minimum-strength report: ${DENSE_ALPHA_PATH_ROWS_CSV}" >&2
fi

echo "Rebuilding path-prediction feature audit..."
"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_path_prediction_feature_audit.py

echo "Rebuilding paper tables..."
"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/build_paper_tables.py

echo "Rebuilding final results summary..."
"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/build_final_results_summary.py

echo "PML report rebuild complete."
echo "Paper tables: pml/results/paper_tables/PML_PAPER_TABLES.md"
echo "Final summary: pml/results/final_summary/FINAL_RESULTS_SUMMARY.md"
