#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/rfm/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pml-matplotlib}"
cd "${PAPER_DIR}"

if [[ "${1:-}" == "--require-cross-model" ]]; then
  "${PYTHON_BIN}" -c 'import pandas as pd; d = pd.read_csv("data/cross_model_replication_summary.csv"); assert len(d) == 24 and d.complete.all(), "Incomplete frozen cross-model outcomes"'
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--require-cross-model]" >&2
  exit 2
fi

# Refresh presentation from frozen inputs. Experimental reruns and result/prose
# replacement are separate operations, never implicit in a formatting refresh.
"${PYTHON_BIN}" scripts/prepare_iclr_assets.py
"${PYTHON_BIN}" scripts/audit_frozen_results_assets.py
bash scripts/compile_pdfs.sh
"${PYTHON_BIN}" scripts/check_iclr_pdf.py
"${PYTHON_BIN}" scripts/build_overleaf_bundle.py
echo "[ICLR paper] frozen-asset refresh, audit, build and packaging complete"
