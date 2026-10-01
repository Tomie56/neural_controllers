#!/usr/bin/env python3
"""Build the manuscript bibliography from archived source BibTeX.

Requires bibtexparser 2.x and pylatexenc. Raw exports are never rewritten.
Every normalization and source exception is recorded for inspection.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import bibtexparser
from pylatexenc.latexencode import unicode_to_latex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW = DATA / 'citation_bibtex'
FIELDS = ('title', 'author', 'booktitle', 'journal', 'volume', 'number', 'pages',
          'year', 'series', 'publisher', 'howpublished', 'eprint', 'archiveprefix',
          'primaryclass', 'doi', 'url')
EXCEPTIONS = {
    'mitchell2022fast': 'OpenReview API returned HTTP 403; its guessed /bibtex route returned 404. Google Scholar returned HTTP 429. Downloaded the ICLR BibTeX supplied by the authors in their official MEND repository. Conference corroboration: https://iclr.cc/virtual/2022/poster/6846 . This is an author export, not a conference/Scholar download.',
    'meng2023mass': 'OpenReview API returned HTTP 403; its guessed /bibtex route returned 404. Downloaded the authors\' own BibTeX from the MEMIT project. Conference corroboration: https://iclr.cc/virtual/2023/poster/11880 . This is an author export, not a conference/Scholar download.',
    'khot2020qasc': 'AAAI OJS download was inaccessible through the shell proxy. Downloaded publisher-deposited DOI metadata as BibTeX via Crossref. Conference record: https://ojs.aaai.org/index.php/AAAI/article/view/6319 . This is a Crossref export, not an OJS download.',
    'radhakrishnan2022mechanism': 'Use the final Science 2024 article and its publisher-deposited Crossref BibTeX. The former arXiv 2022 entry referred to an earlier title/version. Citation key is retained for compatibility.',
    'qwen2026qwen35': 'This is a model release, not a conference paper. Downloaded the exact citation supplied in the official Qwen3.5-2B-Base model card; it cites the Qwen3.5 release blog.',
    'white2025livebench': 'The official ICLR BibTeX contains all 18 authors, including Shubh-Agrawal; it uses Sandeep Sandha, whereas the paper PDF spells out Sandeep Singh Sandha. Follow the exported name in the manuscript and retain this discrepancy explicitly.',
}


def parse_one(text):
    library = bibtexparser.parse_string(text)
    assert not library.failed_blocks, library.failed_blocks
    assert len(library.entries) == 1
    entry = library.entries[0]
    return entry, {f.key.lower(): ' '.join(f.value.split()) for f in entry.fields}


def digest(block):
    return hashlib.sha256(' '.join(block.split()).encode()).hexdigest()


def main():
    manifest = json.loads((RAW / 'download_manifest.json').read_text())
    records = manifest['records']
    assert len(records) == 37 and all(r['status'] == 'downloaded' for r in records)
    previous = (ROOT / 'references.bib').read_text()
    original = RAW / 'references.before-source-rebuild.txt'
    if not original.exists():
        original.write_text(previous)
    before = {e.key: {f.key.lower(): ' '.join(f.value.split()) for f in e.fields}
              for e in bibtexparser.parse_string(original.read_text()).entries}
    rendered, raw_combined, verification, changes = [], [], {}, {}
    rows = ['# Downloaded citation sources (2026-09-23)', '',
            'All 37 entries have a real source BibTeX file. Files named *.bib are unmodified source exports (HTML/Markdown blocks are extracted verbatim apart from code indentation).', '',
            'This is NOT 37 conference/Google Scholar downloads. Google Scholar returned HTTP 429. The table distinguishes conference exports from Crossref, author, arXiv, and model-card sources. Access failures are not evidence that a paper does not exist.', '',
            'The manuscript bibliography is derived from these files. Original citation keys are retained, even when the final publication year changes. Formatting, omitted optional fields, and explicit exceptions are recorded in normalization.json. Raw files retain complete author lists.', '',
            '| Manuscript key | Raw BibTeX | Source kind | Download source |',
            '|---|---|---|---|']
    for record in records:
        key = record['key']
        raw = (RAW / record['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['sha256'], key
        entry, fields = parse_one(raw.decode('utf-8'))
        assert {'title', 'author', 'year'} <= fields.keys(), key
        notes = ['Retain the manuscript citation key; normalize whitespace, Unicode accents and page dashes; protect title capitalization; omit optional abstract/editor/month/address/ISBN fields.']
        entry_type = entry.entry_type
        authors = fields['author'].split(' and ')
        if len(authors) > 30:
            fields['author'] = ' and '.join(authors[:10] + ['others'])
            notes.append(f'Render the first 10 of {len(authors)} exported author entries plus et al.; the raw file preserves every exported name.')
        if entry_type == 'misc' and fields.get('eprint'):
            fields['howpublished'] = 'arXiv preprint arXiv:' + fields['eprint']
            notes.append('Identify the arXiv source explicitly using its exported eprint identifier; follow the year in the downloaded current-version export.')
        if key == 'qwen2026qwen35':
            fields['howpublished'] = 'Qwen Team model release'
            notes.append('Add source-type description; title, author, year and URL follow the official model-card citation.')
        if key == 'meng2023mass':
            entry_type = 'inproceedings'
            fields['booktitle'] = fields.pop('journal')
            fields['title'] = 'Mass-Editing Memory in a Transformer'
            fields['url'] = 'https://openreview.net/forum?id=MkbcAHIYgyS'
            notes.append('The author export labels ICLR as a journal. Normalize to inproceedings/booktitle and use the hyphenated title and paper link confirmed on the official ICLR 2023 program page.')
        normalized = {}
        for field in FIELDS:
            if field not in fields:
                continue
            value = fields[field]
            if field == 'pages':
                value = value.replace('\u2013', '--')
            value = unicode_to_latex(value, non_ascii_only=True)
            if field == 'title':
                value = '{' + value + '}'
            normalized[field] = value
        block = '@' + entry_type + '{' + key + ',\n' + ''.join(
            '    ' + field + ' = {' + value + '},\n' for field, value in normalized.items()) + '}\n'
        rendered.append(block)
        raw_combined.append(raw.decode('utf-8').strip())
        changes[key] = dict(source_key=entry.key, notes=notes, exception=EXCEPTIONS.get(key, ''),
                            changed_fields={f: {'before': before[key].get(f), 'after': fields.get(f)}
                                            for f in FIELDS if before[key].get(f) != fields.get(f)})
        verification[key] = dict(source_urls=[record['source_url'], record['resolved_url']],
            bibliography_entry_sha256=digest(block), source_bibtex_file='citation_bibtex/' + record['file'],
            source_bibtex_sha256=record['sha256'], source_kind=record['source_kind'],
            action='Rebuilt from archived original BibTeX with documented normalization.',
            verification_note=EXCEPTIONS.get(key, 'Source export title, author list, year and identifiers checked against the intended cited work.'))
        rows.append(f"| {key} | [{record['file']}]({record['file']}) | {record['source_kind']} | [source]({record['resolved_url']}) |")
    counts = dict(Counter(r['source_kind'] for r in records))
    rows += ['', '## Source counts', '', json.dumps(counts, ensure_ascii=False), '', '## Exceptions and publication-version changes', '']
    rows += ['- ' + key + ': ' + note for key, note in EXCEPTIONS.items()]
    rows += ['', '- MMLU-Pro now uses NeurIPS 2024, volume 37, pages 95266–95290.',
             '- MMLU-Redux now uses NAACL 2025, pages 5069–5096; the existing 2024 citation key remains an internal identifier.',
             '- ActAdd and RepE current arXiv BibTeX exports use 2024 and 2025 respectively; these are version years, not claims about first publication dates.',
             '', 'Scope: citation metadata and downloaded-source provenance. This does not independently validate experiments or establish that every contextual claim is supported.',
             'No-invention status: no missing BibTeX was synthesized and no author/venue was guessed. Non-conference sources remain explicitly labeled.', '']
    (ROOT / 'references.bib').write_text('\n'.join(rendered))
    (RAW / 'normalization.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n')
    (RAW / 'all_original_exports.bib.txt').write_text('\n\n'.join(raw_combined) + '\n')
    (RAW / 'README.md').write_text('\n'.join(rows))
    (DATA / 'citation_verification.json').write_text(json.dumps(dict(audit_date='2026-09-23',
        mode='citation-audit', scope='All 37 existing citations: archived source BibTeX and manuscript normalization.',
        automation_limit='Hashes check local integrity, not online truth or citation-context support.',
        source_counts=counts, entries=verification), ensure_ascii=False, indent=2) + '\n')
    archive = ROOT / 'deliverables/PML_ICLR27_citation_bibtex.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
        for path in sorted(RAW.rglob('*')):
            if path.is_file():
                bundle.write(path, 'citation_bibtex/' + str(path.relative_to(RAW)))
        bundle.write(ROOT / 'references.bib', 'references.bib')
        bundle.write(DATA / 'citation_verification.json', 'citation_verification.json')
    print(json.dumps(dict(entries=len(records), source_counts=counts, archive=str(archive)), indent=2))


if __name__ == '__main__':
    main()
