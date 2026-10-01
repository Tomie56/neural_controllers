#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/data/neural_controllers/pml/results/path_representative_50_capability_onset_probe}"
DATASET_LABEL="${DATASET_LABEL:-path_representative_50_capability_onset_probe}"
METHODS="${METHODS:-mean_difference logistic}"
DIAGNOSTICS_CSV="${DIAGNOSTICS_CSV:-/data/neural_controllers/pml/results/path_representative_subset/representative_50_diagnostics.csv}"

read -r -a METHOD_ARGS <<< "${METHODS}"

RESULT_FILES=()
for METHOD in "${METHOD_ARGS[@]}"; do
  RESULT_PATH="${RESULT_ROOT}/${METHOD}/suppression_results.jsonl"
  if [[ ! -s "${RESULT_PATH}" ]]; then
    echo "ERROR: missing result file for ${METHOD}: ${RESULT_PATH}" >&2
    exit 1
  fi
  RESULT_FILES+=("${RESULT_PATH}")
done

SUMMARY_DIR="${RESULT_ROOT}/summary"
"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/summarize_suppression_results.py \
  --results-jsonl "${RESULT_FILES[@]}" \
  --output-dir "${SUMMARY_DIR}"

"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_strength_curves.py \
  --per-record-csv "${SUMMARY_DIR}/per_record.csv" \
  --label "${DATASET_LABEL}" \
  --output-dir "${RESULT_ROOT}/path_geometry_audit" \
  --plot

"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_strength_prediction.py \
  --path-rows-csv "${RESULT_ROOT}/path_geometry_audit/strength_path_rows.csv" \
  --output-dir "${RESULT_ROOT}/path_geometry_audit/strength_prediction" \
  --groups pooled dataset="${DATASET_LABEL}" \
  --label-keys clean_window_label usable_path_label damage_first_label capability_first_label no_effect_label \
  --numeric-keys target_onset_strength damage_onset_strength capability_onset_strength clean_window_width pareto_score_with_capability \
  --feature-groups baseline all

"${PYTHON_BIN}" pml/src/predictive_memory_localization/analysis/analyze_representative_subset_outcomes.py \
  --path-rows-csv "${RESULT_ROOT}/path_geometry_audit/strength_path_rows.csv" \
  --diagnostics-csv "${DIAGNOSTICS_CSV}" \
  --output-dir "${RESULT_ROOT}/path_geometry_audit/representative_outcomes" \
  --dataset-label "${DATASET_LABEL}"

echo "Analysis complete: ${RESULT_ROOT}"
