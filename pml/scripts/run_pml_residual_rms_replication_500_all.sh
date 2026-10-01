#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers

RUN_CALIBRATION="${RUN_CALIBRATION:-1}"
RUN_QWEN3_1_7B_EXTRACT="${RUN_QWEN3_1_7B_EXTRACT:-1}"
RUN_QWEN3_1_7B="${RUN_QWEN3_1_7B:-0}"
RUN_QWEN3_5_2B="${RUN_QWEN3_5_2B:-1}"
RUN_MINISTRAL="${RUN_MINISTRAL:-1}"

if [[ "${RUN_CALIBRATION}" == "1" ]]; then
  /data/neural_controllers/pml/scripts/run_pml_residual_rms_calibration.sh
fi
if [[ "${RUN_QWEN3_1_7B_EXTRACT}" == "1" ]]; then
  /data/neural_controllers/pml/scripts/run_pml_extract_qwen3_1_7b_residual_rms_500.sh
fi
if [[ "${RUN_QWEN3_1_7B}" == "1" ]]; then
  /data/neural_controllers/pml/scripts/run_pml_residual_rms_replication_500_qwen3_1_7b.sh
fi
if [[ "${RUN_QWEN3_5_2B}" == "1" ]]; then
  /data/neural_controllers/pml/scripts/run_pml_residual_rms_replication_500_qwen3_5_2b.sh
fi
if [[ "${RUN_MINISTRAL}" == "1" ]]; then
  /data/neural_controllers/pml/scripts/run_pml_residual_rms_replication_500_ministral_3_3b.sh
fi

echo "[RMS replication queue] complete"
