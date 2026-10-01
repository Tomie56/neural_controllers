#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MAIN_ROOT="${MAIN_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage_activation_pml/main_3000/qwen3_1_7b}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-${MAIN_ROOT}/prediction_dataset_strict_later}"
PILOT_ROOT="${PILOT_ROOT:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b}"
EXCLUSION_DATASET_DIR="${EXCLUSION_DATASET_DIR:-${MAIN_ROOT}/prediction_dataset_strict_later_excluding_pilot500}"
EXCLUSION_ANALYSIS_DIR="${EXCLUSION_ANALYSIS_DIR:-${MAIN_ROOT}/prediction_analysis_strict_later_excluding_pilot500}"
R_ONLY_ANALYSIS_DIR="${R_ONLY_ANALYSIS_DIR:-${MAIN_ROOT}/prediction_analysis_strict_later_weak_response_baselines}"
SIMPLE_BASELINE_DIR="${SIMPLE_BASELINE_DIR:-${MAIN_ROOT}/weak_response_simple_baselines}"
PATH_TOPOLOGY_DIR="${PATH_TOPOLOGY_DIR:-${MAIN_ROOT}/path_topology_analysis}"
COMBINED_REPORT_DIR="${COMBINED_REPORT_DIR:-${MAIN_ROOT}/cpu_supplemental_evidence}"
DENSE_TRAIN_DATASET_DIR="${DENSE_TRAIN_DATASET_DIR:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2_bidirectional_pilot_500/qwen3_1_7b/selector_dataset_tau_main_q95}"
SELECTOR_VALIDATION_DATASET_DIR="${SELECTOR_VALIDATION_DATASET_DIR:-/data/neural_controllers/pml/results/fresh_multidomain_3000/stage2b_selector_validation_100/qwen3_1_7b/selector_dataset_tau_main_q95}"
SELECTOR_OUTPUT_DIR="${SELECTOR_OUTPUT_DIR:-${MAIN_ROOT}/strength_predictor_multifidelity_dense}"

TARGETS="${TARGETS:-later_suppression_path later_enhancement_path later_target_any_path later_neighbor_damage_any_path later_capability_damage_any_path later_clean_suppression_path later_clean_enhancement_path later_clean_any_path}"
RF_N_JOBS="${RF_N_JOBS:-4}"
N_SPLITS="${N_SPLITS:-5}"
SEED="${SEED:-113}"
OVERWRITE="${OVERWRITE:-0}"
STAGE="${STAGE:-all}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

run_exclusion() {
  echo "[CPU supplemental] build 2,500-record layer-pilot exclusion dataset"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.build_layer_pilot_exclusion_dataset \
    --source-dataset-dir "${SOURCE_DATASET_DIR}" \
    --pilot-root "${PILOT_ROOT}" \
    --output-dir "${EXCLUSION_DATASET_DIR}" \
    --seed "${SEED}" \
    "${OVERWRITE_ARGS[@]}"

  echo "[CPU supplemental] record-held-out predictor on remaining 2,500 records"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_activation_space_pml_main_prediction \
    --dataset-dir "${EXCLUSION_DATASET_DIR}" \
    --output-dir "${EXCLUSION_ANALYSIS_DIR}" \
    --targets ${TARGETS} \
    --feature-sets B B+M B+M+L B+M+L+R \
    --splits record \
    --cohorts all learned rfm \
    --models logistic random_forest \
    --n-splits "${N_SPLITS}" \
    --rf-n-jobs "${RF_N_JOBS}" \
    --seed "${SEED}" \
    "${OVERWRITE_ARGS[@]}"
}

run_weak_response() {
  echo "[CPU supplemental] learned R-only and cumulative weak-response predictors"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.run_activation_space_pml_main_prediction \
    --dataset-dir "${SOURCE_DATASET_DIR}" \
    --output-dir "${R_ONLY_ANALYSIS_DIR}" \
    --targets ${TARGETS} \
    --feature-sets R M+R B+M+R B+M+L+R \
    --splits record dataset domain \
    --cohorts all learned rfm \
    --models logistic random_forest \
    --n-splits "${N_SPLITS}" \
    --rf-n-jobs "${RF_N_JOBS}" \
    --seed "${SEED}" \
    "${OVERWRITE_ARGS[@]}"

  echo "[CPU supplemental] training-free weak-response baselines"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.analyze_weak_response_baselines \
    --dataset-dir "${SOURCE_DATASET_DIR}" \
    --output-dir "${SIMPLE_BASELINE_DIR}" \
    --n-splits "${N_SPLITS}" \
    --seed "${SEED}" \
    "${OVERWRITE_ARGS[@]}"
}

run_selector_zero() {
  echo "[CPU supplemental] selector versus no-intervention and fixed-policy baselines"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.evaluate_strength_selector_policy \
    --dense-train-dataset-dir "${DENSE_TRAIN_DATASET_DIR}" \
    --validation-dataset-dir "${SELECTOR_VALIDATION_DATASET_DIR}" \
    --selector-output-dir "${SELECTOR_OUTPUT_DIR}" \
    --output-dir "${SELECTOR_OUTPUT_DIR}" \
    --bootstrap-reps 5000 \
    --seed "${SEED}"
}

run_path_topology() {
  echo "[CPU supplemental] onset ordering, clean-window span, damage-first, and instability"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.analyze_intervention_path_topology \
    --stage-root "${MAIN_ROOT}/outcomes" \
    --dataset-manifest "${SOURCE_DATASET_DIR}/prediction_dataset_manifest.json" \
    --output-dir "${PATH_TOPOLOGY_DIR}" \
    --seed "${SEED}" \
    "${OVERWRITE_ARGS[@]}"
}

run_report() {
  echo "[CPU supplemental] consolidate prevalence, AUROC, AP, and 95% confidence intervals"
  "${PYTHON_BIN}" -m predictive_memory_localization.analysis.summarize_cpu_supplemental_evidence \
    --main-root "${MAIN_ROOT}" \
    --output-dir "${COMBINED_REPORT_DIR}"
}

case "${STAGE}" in
  all)
    run_exclusion
    run_weak_response
    run_selector_zero
    run_path_topology
    run_report
    ;;
  exclusion) run_exclusion ;;
  weak-response) run_weak_response ;;
  selector-zero) run_selector_zero ;;
  path-topology) run_path_topology ;;
  report) run_report ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected all, exclusion, weak-response, selector-zero, path-topology, or report" >&2
    exit 2
    ;;
esac

echo "[CPU supplemental] complete"
echo "Exclusion report: ${EXCLUSION_DATASET_DIR}/PILOT500_EXCLUSION_REPORT.md"
echo "Exclusion prediction: ${EXCLUSION_ANALYSIS_DIR}/PML_MAIN_PREDICTION_REPORT.md"
echo "Weak learned prediction: ${R_ONLY_ANALYSIS_DIR}/PML_MAIN_PREDICTION_REPORT.md"
echo "Weak simple baselines: ${SIMPLE_BASELINE_DIR}/WEAK_RESPONSE_BASELINE_REPORT.md"
echo "Selector paired comparisons: ${SELECTOR_OUTPUT_DIR}/strength_policy_paired_comparisons.csv"
echo "Path topology: ${PATH_TOPOLOGY_DIR}/PATH_TOPOLOGY_REPORT.md"
echo "Combined metrics: ${COMBINED_REPORT_DIR}/CPU_SUPPLEMENTAL_EVIDENCE_REPORT.md"
