#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

export INPUT_JSONL="${INPUT_JSONL:-/data/neural_controllers/pml/data/suppression_commonsenseqa_path_representative_50_with_capability.jsonl}"
export RESULT_ROOT="${RESULT_ROOT:-/data/neural_controllers/pml/results/path_representative_50_capability_onset_probe}"
export MAX_RECORDS="${MAX_RECORDS:-50}"
export METHODS="${METHODS:-mean_difference logistic}"
export DATASET_LABEL="${DATASET_LABEL:-path_representative_50_capability_onset_probe}"
export DIAGNOSTICS_CSV="${DIAGNOSTICS_CSV:-/data/neural_controllers/pml/results/path_representative_subset/representative_50_diagnostics.csv}"

echo "Running path-representative capability-onset probe."
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "MAX_RECORDS=${MAX_RECORDS}"
echo "METHODS=${METHODS}"
echo "DATASET_LABEL=${DATASET_LABEL}"
echo "DIAGNOSTICS_CSV=${DIAGNOSTICS_CSV}"

exec bash pml/scripts/run_heldout_capability_onset_probe.sh "$@"
