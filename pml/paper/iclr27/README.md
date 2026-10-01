# PML: ICLR 2027 submission

This is the independent ICLR workspace. The AAAI workspace is unchanged.
The official downloaded template is retained in `iclr2027/`; the root style
and bibliography files are byte-identical to those originals.

## Build and refresh

Run from this directory:

```bash
bash scripts/refresh_paper_assets.sh --require-cross-model
```

The script formats frozen assets, audits numerical consistency, builds both
PDFs, checks page count/fonts/anonymity, and creates the Overleaf source ZIP.
It does not rerun experiments or regenerate manuscript prose. Override
`PYTHON_BIN` or `TEXBIN` if using a different Python/TeX installation.

To compile edited text without redrawing figures:

```bash
bash scripts/compile_pdfs.sh
/data/miniconda3/envs/rfm/bin/python scripts/check_iclr_pdf.py
/data/miniconda3/envs/rfm/bin/python scripts/build_overleaf_bundle.py
```

## Outputs

- `deliverables/PML_ICLR27_main.pdf`: 13 Letter pages; main text ends on page 9;
  AI/reproducibility statements and references start on page 10.
- `deliverables/PML_ICLR27_supplement.pdf`: 10 Letter pages.
- `deliverables/PML_ICLR27_citation_bibtex.zip`: all 37 original source
  BibTeX files, download URLs/checksums, normalization records and corrected bibliography.
  The unpacked sources are also kept in `data/citation_bibtex/`.
- `dist/pml_iclr27_overleaf.zip`: self-contained manuscript source for editing;
  select `main.tex` or `supplement.tex` as the root file.
- `data/iclr_pdf_checks.json`: latest PDF checks and hashes.

Both PDFs use the official anonymous review mode and embedded Type 1/TrueType
fonts. CID TrueType fonts with Identity-H encoding are not Type 3 fonts and
are retained. Separate supplementary text is permitted by the supplied
ICLR author guidelines. The Overleaf ZIP is a manuscript source archive,
not an experimental-code release or a required submission upload.

The original result-generation scripts remain available for provenance.
They can replace tables/prose and are not part of the default refresh.
After a deliberate result update, run the ICLR refresh and review the changes.

## Remaining author checks

All 37 original source BibTeX files were downloaded on 2026-09-23. See
`data/citation_bibtex/README.md` for the entry-by-entry mapping and explicit
source exceptions: 25 conference exports, 2 publisher-deposited Crossref
exports, 2 author-provided exports, 7 arXiv exports and 1 model-card citation.
Google Scholar returned HTTP 429; the alternative sources are not labeled as
Scholar or conference downloads. The earlier metadata audit is retained in
`data/CITATION_AUDIT_20260923.md` as a historical record.

`references.bib` is now derived from these raw exports. Final publication
versions replace the earlier entries for MMLU-Pro, MMLU-Redux and RFM.
`data/citation_bibtex/normalization.json` records every field change and
formatting exception. Original exports preserve complete author lists;
the two long model reports render the first ten names plus et al.
The asset audit checks both manuscript-entry hashes and raw-source hashes
against `data/citation_verification.json`. These checks detect local changes;
they do not substitute for online verification or citation-context review.

To refresh sources deliberately, run `scripts/download_citation_bibtex.py`
(requires requests), then `scripts/rebuild_references_from_exports.py`
(requires bibtexparser 2.x and pylatexenc), inspect the source exceptions and
normalization record, and run the normal paper refresh. Source downloads
are not part of the default offline build.

See `data/ICLR27_REPAIR_NOTES.md`. The AI-use inventory and human-validation
history need author confirmation. Primary all-method prediction estimates
still come from the 36,000 logged-row cohort with a duplicate random entry;
the supplement now discloses its weighting, and the old aggregate
0.1774/0.1821 claims have been removed. No deduplicated rerun is claimed.
