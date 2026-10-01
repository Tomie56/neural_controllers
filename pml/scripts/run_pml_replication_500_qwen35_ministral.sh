#!/usr/bin/env bash
set -euo pipefail

cd /data/neural_controllers

RUN_QWEN35="${RUN_QWEN35:-1}"
RUN_MINISTRAL="${RUN_MINISTRAL:-1}"
RUN_QWEN35_MAIN="${RUN_QWEN35_MAIN:-${RUN_QWEN35}}"
RUN_QWEN35_ENDPOINT="${RUN_QWEN35_ENDPOINT:-${RUN_QWEN35}}"
RUN_MINISTRAL_MAIN="${RUN_MINISTRAL_MAIN:-${RUN_MINISTRAL}}"
RUN_MINISTRAL_ENDPOINT="${RUN_MINISTRAL_ENDPOINT:-${RUN_MINISTRAL}}"

if [[ "${RUN_QWEN35_MAIN}" == "1" ]]; then
  echo "[replication queue] starting Qwen3.5-2B-Base main replication"
  /data/neural_controllers/pml/scripts/run_pml_replication_500_qwen3_5_2b.sh
fi

if [[ "${RUN_QWEN35_ENDPOINT}" == "1" ]]; then
  echo "[replication queue] starting Qwen3.5-2B-Base endpoint validation"
  /data/neural_controllers/pml/scripts/run_pml_endpoint_formal_qwen3_5_2b.sh
fi

if [[ "${RUN_MINISTRAL_MAIN}" == "1" ]]; then
  echo "[replication queue] starting Ministral-3-3B-Base-2512 main replication"
  /data/neural_controllers/pml/scripts/run_pml_replication_500_ministral_3_3b.sh
fi

if [[ "${RUN_MINISTRAL_ENDPOINT}" == "1" ]]; then
  echo "[replication queue] starting Ministral-3-3B-Base-2512 endpoint validation"
  /data/neural_controllers/pml/scripts/run_pml_endpoint_formal_ministral_3_3b.sh
fi

echo "[replication queue] all requested main and endpoint stages complete"
