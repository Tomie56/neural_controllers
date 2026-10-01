#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"

RUN_BUILD_TRAIN="${RUN_BUILD_TRAIN:-1}"
RUN_VALIDATION_DENSE="${RUN_VALIDATION_DENSE:-1}"
RUN_TRAIN_EVAL="${RUN_TRAIN_EVAL:-1}"

if [[ "${RUN_BUILD_TRAIN}" == "1" ]]; then
  echo "[Stage 2b] Build selector dataset from existing 500 dense curves"
  PYTHON_BIN="${PYTHON_BIN}" OVERWRITE=1 ./pml/scripts/build_stage2_strength_selector_dataset_qwen3_1_7b.sh
fi

if [[ "${RUN_VALIDATION_DENSE}" == "1" ]]; then
  echo "[Stage 2b] Run 100-record dense validation grid"
  PYTHON_BIN="${PYTHON_BIN}" OVERWRITE=0 ./pml/scripts/run_stage2b_selector_validation_100_qwen3_1_7b.sh
fi

if [[ "${RUN_TRAIN_EVAL}" == "1" ]]; then
  echo "[Stage 2b] Train and evaluate lightweight selectors"
  PYTHON_BIN="${PYTHON_BIN}" OVERWRITE=1 ./pml/scripts/train_and_eval_strength_selector_qwen3_1_7b.sh
fi

echo "Stage 2b strength selector experiment complete."
