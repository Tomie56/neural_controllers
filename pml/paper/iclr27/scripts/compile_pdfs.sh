#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PAPER_DIR}/../../.." && pwd)"
TEXBIN="${TEXBIN:-${REPO_DIR}/.TinyTeX/bin/x86_64-linux}"
LATEXMK="${TEXBIN}/latexmk"
DELIVERABLES_DIR="${PAPER_DIR}/deliverables"

if [[ ! -x "${LATEXMK}" ]]; then
  echo "error: latexmk not found at ${LATEXMK}" >&2
  echo "set TEXBIN to the TinyTeX/TeX Live binary directory" >&2
  exit 1
fi

export PATH="${TEXBIN}:${PATH}"
mkdir -p "${PAPER_DIR}/build_pdf_main" \
  "${PAPER_DIR}/build_pdf_supp" \
  "${DELIVERABLES_DIR}"

cd "${PAPER_DIR}"
"${LATEXMK}" -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build_pdf_main main.tex
"${LATEXMK}" -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build_pdf_supp supplement.tex

for entry in \
  "build_pdf_main/main:PML_ICLR27_main" \
  "build_pdf_supp/supplement:PML_ICLR27_supplement"; do
  source_stem="${entry%%:*}"
  output_stem="${entry##*:}"
  log_file="${PAPER_DIR}/${source_stem}.log"

  if rg -q 'undefined citations|undefined references|Citation .* undefined|Reference .* undefined|Overfull \\[hv]box' "${log_file}"; then
    echo "error: unresolved reference or overfull box in ${log_file}" >&2
    rg -n 'undefined citations|undefined references|Citation .* undefined|Reference .* undefined|Overfull \\[hv]box' "${log_file}" >&2
    exit 1
  fi

  cp "${PAPER_DIR}/${source_stem}.pdf" "${DELIVERABLES_DIR}/${output_stem}.pdf"
done

echo "[pdf] main: ${DELIVERABLES_DIR}/PML_ICLR27_main.pdf"
pdfinfo "${DELIVERABLES_DIR}/PML_ICLR27_main.pdf" | rg 'Pages|Page size|File size'
echo "[pdf] supplement: ${DELIVERABLES_DIR}/PML_ICLR27_supplement.pdf"
pdfinfo "${DELIVERABLES_DIR}/PML_ICLR27_supplement.pdf" | rg 'Pages|Page size|File size'
