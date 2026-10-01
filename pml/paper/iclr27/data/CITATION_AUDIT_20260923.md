# ICLR citation audit and repair

Mode: citation-audit. Date: 2026-09-23.

This file records the earlier metadata/context audit. A subsequent same-day
pass downloaded an original BibTeX for every entry and rebuilt the bibliography.
The current source-of-record is [the 37-entry download ledger](citation_bibtex/README.md)
and its normalization.json. In particular, MMLU-Redux now cites NAACL 2025,
MMLU-Pro cites NeurIPS 2024, and RFM cites Science 2024. The RFM title discrepancy
below was a version mismatch, not a nonexistent title. Current arXiv exports
use 2024 for ActAdd and 2025 for RepE. LiveBench now follows its actual official
BibTeX export, with the author-name spelling discrepancy documented in the ledger.
After that rebuild, the main PDF remains 13 pages with 9 pages of main text;
the supplement is 10 pages. Both pass reference, layout, font and anonymity checks.

## Scope and result

Checked all 37 bibliography entries and their uses in the main manuscript and
supplement. Fifteen entries needed changes: thirteen had author, title, venue,
page or identifier issues; two additional entries had incorrect primary arXiv
categories. All 37 intended works were identifiable. No unrelated replacement
papers or new literature were introduced. Experimental results were unchanged.

The copied AAAI bibliography had the same errors. The earlier AAAI audits
checked key existence, uniqueness and mandatory fields; those checks did not
establish that an author list or DOI belonged to the named paper. The old
literature search record also contained the wrong Singh venue and Xu URLs.
Only the active ICLR workspace is changed by this repair.

## Corrected entries

| Entry | Finding | Correction and primary source |
| --- | --- | --- |
| pmlr-v238-singh24d → pmlr-v235-singh24d | Unrelated authors; incorrect AISTATS venue, volume, pages and URL | Shashwat Singh et al., ICML 2024, PMLR 235:45663–45680. [PMLR](https://proceedings.mlr.press/v235/singh24d.html) |
| xu-etal-2026-controllable | Wrong authors/pages; old DOI/URL identify MentalSeek-Dx, an unrelated psychiatry paper | Ziwen Xu et al.; ACL ID 2026.acl-long.1443; 31269–31299. [ACL](https://aclanthology.org/2026.acl-long.1443/) |
| xu-etal-2026-steering | Wrong authors/pages; old DOI/URL identify an unrelated hallucination-detection paper | Ziwen Xu et al.; ACL ID 2026.acl-long.1463; 31719–31736. [ACL](https://aclanthology.org/2026.acl-long.1463/) |
| fan2026steerable | Entire author list mismatched | Chenrui Fan, Yize Cheng, Ming Li, Soheil Feizi, Tianyi Zhou. [arXiv](https://arxiv.org/abs/2606.11599) |
| wu25a | Missing/unrelated authors and wrong pages | Official eight-author list; 67035–67080. [PMLR](https://proceedings.mlr.press/v267/wu25a.html) |
| lee-etal-2025-localization | Wrong/missing coauthors and pages | Hwiyeong Lee, Uiji Hwang, Hyelim Lim, Taeuk Kim; 21857–21869. [ACL](https://aclanthology.org/2025.emnlp-main.1109/) |
| liu2026ministral3 | Incorrect coauthors after the first author | Verified first ten authors followed by et al., matching the existing abbreviated-report convention. [arXiv](https://arxiv.org/abs/2601.08584) |
| gema2024mmluredux | Second half of the author list mismatched | Use the full author list of the cited arXiv record. [arXiv](https://arxiv.org/abs/2406.04127) |
| white2025livebench | Mixed preprint/final author list; no source URL | Use the final 18-author list and official proceedings URL. The [final PDF, p. 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/e4a46394ba5378b3f9a186a5b4c650d1-Paper-Conference.pdf) resolves omissions/abbreviations in HTML metadata. |
| mitchell22a | Last two authors reversed | Christopher D. Manning precedes Chelsea Finn. [PMLR](https://proceedings.mlr.press/v162/mitchell22a.html) |
| turner2023steering | Wrong given name for Leech | Gavin Leech. [arXiv](https://arxiv.org/abs/2308.10248) |
| tan2024steering | First-author name and author order differ from publication | Daniel Tan; Brooks Paige before Dimitrios Kanoulas; added volume 37 and pages 139179–139212. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b-Abstract-Conference.html) |
| radhakrishnan2022mechanism | Title differs from current cited arXiv record | Align to the official title concerning deep fully connected networks and recursive kernel machines. This is a real work, not a nonexistent citation. [arXiv](https://arxiv.org/abs/2212.13881) |
| pres2024reliable | Wrong primary arXiv category | cs.AI. [arXiv](https://arxiv.org/abs/2410.17245) |
| zou2023representation | Wrong primary arXiv category | cs.LG. [arXiv](https://arxiv.org/abs/2310.01405) |

## Citation-context matrix

| Manuscript use | Support assessment | Action |
| --- | --- | --- |
| Feed-forward memory, knowledge neurons and causal localization | Supported at the stated background level by Geva, Dai and ROME | Retained |
| Learned editors, external memory, parameter updates and evaluation toolkits | Supported by KnowledgeEditor, MEND, SERAC, MEMIT, Yao and EasyEdit | Retained; corrected SERAC author order |
| Localization need not identify the best editing/unlearning parameters | Supported by Hase and Lee | Retained; corrected Lee metadata |
| Activation engineering, CAA, RepE and affine representation surgery | Supported; affine surgery is a transformation, not necessarily a single direction | Use “activation interventions” for the grouped description |
| Truthfulness, activation scaling and preference–utility tradeoffs | Supported by ITI, Stoehr and the correctly identified Xu paper | Corrected Xu bibliographic identity |
| Detection/steering comparisons and robustness across prompts/models | Supported by AxBench, Tan, Da Silva and Goyal | Retained; corrected metadata |
| Pres and hierarchical controllability evaluation | Previous joint attribution was broader than the stated Pres evaluation criteria | Separate task-matched/likelihood-aware/baseline evaluation from behavioral-granularity evaluation |
| Closest-work comparison with SteerBoost | Full text, Sections 4–5, uses early decoded token/layer states and probability-ranked strength search | Replace “first-token dynamics” with “early decoding dynamics”; preserve the actual comparison |
| RFM/AGOP as task-adapted geometry | Supported by the RFM paper's stated feature-learning mechanism | Correct title, retain method attribution |
| Benchmark and model sources | Source identities match the named benchmarks/models; they do not independently validate PML's generated probes or measured results | Correct MMLU-Redux, LiveBench and Ministral metadata |

Metadata checking used publisher/ACL/PMLR/NeurIPS pages, arXiv records and the
official model card. Context checking used official abstracts and the relevant
SteerBoost full-text sections. OpenReview pages were blocked by browser
verification: MEND was cross-checked using its arXiv record (including ICLR
2022 in the comments); MEMIT using arXiv and the authors' project page (ICLR
2023); LiveBench using the official ICLR proceedings and final PDF.

## Validation and limits

- citation_verification.json records primary-source URLs for every entry and
  hashes of the reviewed BibTeX entries.
- The normal asset audit rejects changed/new bibliography entries until their
  source verification is updated. This prevents silent reuse of the old
  bibliography; it is not an automated proof of citation truth.
- Main/supplement PDFs and the Overleaf bundle rebuilt successfully. Main text
  still ends on page 9 (13 total pages); supplement is 9 pages. No unresolved
  citations/references or overfull boxes; all fonts remain embedded.
- A regression check that reinstates the incorrect Fan given name in memory
  is rejected by the citation audit. No manuscript file is changed by the test.
- This citation audit does not independently replicate experiments or certify
  author-confirmation items from ICLR27_REPAIR_NOTES.md.

Severity: the wrong-paper identifiers and unrelated author lists were
submission-blocking bibliographic errors and have been corrected. Minor name,
order, title-version and category issues were corrected separately.

Next owner: local submission checks for the rebuilt PDFs. No-invention status:
all corrections trace to identified primary sources; no new results or claims
were fabricated.
