#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers
export PYTHONPATH="/data/neural_controllers/pml/src:/data/neural_controllers:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base}"
TASKS_JSONL="${TASKS_JSONL:-/data/neural_controllers/pml/data/editing_tasks_commonsenseqa_balanced_followup_144.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-/data/neural_controllers/pml/results/commonsenseqa_balanced_followup_144/rome_attack}"
BASELINE_JOINED_CSV="${BASELINE_JOINED_CSV:-/data/neural_controllers/pml/results/commonsenseqa_balanced_followup_144/rome/prediction_analysis_baseline_features/editing_joined_rows.csv}"
AGOP_JOINED_CSV="${AGOP_JOINED_CSV:-/data/neural_controllers/pml/results/commonsenseqa_balanced_followup_144/rome/prediction_analysis_agop_features/editing_joined_rows.csv}"
MAX_RECORDS="${MAX_RECORDS:-144}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
ATTACK_PROMPT_LIMIT="${ATTACK_PROMPT_LIMIT:-6}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-300}"

ROME_DIR="${RESULT_ROOT}/rome"

"$PYTHON_BIN" pml/src/predictive_memory_localization/methods/run_easyedit_prediction.py \
  --tasks-jsonl "$TASKS_JSONL" \
  --output-dir "$ROME_DIR" \
  --method ROME \
  --model-local-path "$MODEL_PATH" \
  --max-records "$MAX_RECORDS" \
  --torch-dtype "$TORCH_DTYPE" \
  --include-attack-prompts \
  --attack-prompt-limit "$ATTACK_PROMPT_LIMIT" \
  --overwrite

"$PYTHON_BIN" pml/src/predictive_memory_localization/analysis/analyze_rome500_error_ablation.py \
  --editing-results-jsonl "${ROME_DIR}/rome_editing_results.jsonl" \
  --baseline-joined-csv "$BASELINE_JOINED_CSV" \
  --agop-joined-csv "$AGOP_JOINED_CSV" \
  --output-dir "${ROME_DIR}/error_ablation_analysis" \
  --bootstrap-iters "$BOOTSTRAP_ITERS"

"$PYTHON_BIN" pml/src/predictive_memory_localization/analysis/analyze_attack_editing_prediction.py \
  --error-rows-csv "${ROME_DIR}/error_ablation_analysis/rome500_error_rows.csv" \
  --output-dir "${ROME_DIR}/attack_editing_analysis" \
  --bootstrap-iters "$BOOTSTRAP_ITERS"

echo "Attack ROME 144 results: $RESULT_ROOT"
