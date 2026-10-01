#!/usr/bin/env python3
"""Archive real source exports; never synthesize a failed download."""
import concurrent.futures
import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/citation_bibtex'
OVERRIDES = {
    'mihaylov-etal-2018-suit': ('conference_export', 'https://aclanthology.org/D18-1260.bib'),
    'welbl-etal-2017-crowdsourcing': ('conference_export', 'https://aclanthology.org/W17-4413.bib'),
    'zellers-etal-2019-hellaswag': ('conference_export', 'https://aclanthology.org/P19-1472.bib'),
    'yang2025qwen3': ('arxiv_export', 'https://arxiv.org/bibtex/2505.09388'),
    'liu2026ministral3': ('arxiv_export', 'https://arxiv.org/bibtex/2601.08584'),
    'wang2024mmlupro': ('conference_page', 'https://proceedings.neurips.cc/paper_files/paper/2024/hash/ad236edc564f3e3156e1b2feafb99a24-Abstract.html'),
    'gema2024mmluredux': ('conference_export', 'https://aclanthology.org/2025.naacl-long.262.bib'),
    'radhakrishnan2022mechanism': ('publisher_crossref_export', 'https://api.crossref.org/works/10.1126/science.adi5639/transform/application/x-bibtex'),
    'mitchell2022fast': ('author_markdown', 'https://raw.githubusercontent.com/eric-mitchell/mend/main/README.md'),
    'meng2023mass': ('author_page', 'https://memit.baulab.info/'),
    'khot2020qasc': ('publisher_crossref_export', 'https://api.crossref.org/works/10.1609/aaai.v34i05.6319/transform/application/x-bibtex'),
    'qwen2026qwen35': ('model_card', 'https://huggingface.co/Qwen/Qwen3.5-2B-Base/raw/main/README.md'),
}


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.blocks, self.block = [], [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])
        if tag == 'pre' or (tag == 'code' and attrs.get('id') == 'bibtex'):
            self.block = []

    def handle_endtag(self, tag):
        if tag in ('pre', 'code') and self.block is not None:
            self.blocks.append(''.join(self.block))
            self.block = None

    def handle_data(self, data):
        if self.block is not None:
            self.block.append(data)


def entries():
    for block in re.split(r'(?m)(?=^@)', (ROOT / 'references.bib').read_text()):
        match = re.match(r'@(\w+)\{([^,]+),', block)
        if match:
            fields = dict(re.findall(r'^\s*(\w+)\s*=\s*[{"](.*)[}"]\s*,?\s*$', block, re.M))
            yield match[2], fields


def source(key, fields):
    if key in OVERRIDES:
        return OVERRIDES[key]
    url = fields.get('url', '')
    if 'aclanthology.org/' in url:
        return 'conference_export', url.rstrip('/') + '.bib'
    if any(domain in url for domain in ('proceedings.mlr.press', 'proceedings.neurips.cc', 'proceedings.iclr.cc')):
        return 'conference_page', url
    if fields.get('eprint'):
        return 'arxiv_export', 'https://arxiv.org/bibtex/' + fields['eprint']
    return 'unknown', url


def get(url):
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    response.encoding = 'utf-8'
    return response


def download(item):
    key, fields = item
    kind, url = source(key, fields)
    record = dict(key=key, source_kind=kind, source_url=url)
    try:
        response = get(url)
        raw = response.content
        if kind in ('conference_page', 'author_page'):
            (DEST / 'pages' / (key + '.html')).write_bytes(raw)
            page = Page()
            page.feed(response.text)
            links = [link for link in page.links if '/bibtex' in link or link.endswith('.bib')]
            if links:
                response = get(urljoin(response.url, links[0]))
                raw = response.content
                record['source_kind'] = 'conference_export'
            else:
                blocks = [block for block in page.blocks if re.search(r'@\w+\s*\{', block)]
                if len(blocks) != 1:
                    raise ValueError('No unique official BibTeX export block')
                raw = blocks[0].encode('utf-8')
                record['source_kind'] = 'conference_inline_export' if kind == 'conference_page' else 'author_inline_export'
        elif kind in ('model_card', 'author_markdown'):
            (DEST / 'pages' / (key + '.md')).write_bytes(raw)
            fence = chr(96) * 3
            blocks = re.findall(fence + r'(?:bibtex|bib)\s*\n(.*?)' + fence, response.text, re.S | re.I)
            if kind == 'author_markdown' and not blocks:
                blocks = re.findall(r'(?m)^    (@inproceedings\{.*?\n    \})', response.text, re.S)
                blocks = [re.sub(r'(?m)^    ', '', block) for block in blocks]
            if len(blocks) != 1:
                raise ValueError('No unique model-card BibTeX block')
            raw = blocks[0].encode('utf-8')
            record['source_kind'] = 'model_card_inline_export' if kind == 'model_card' else 'author_inline_export'
        if not re.match(rb'\s*@(?:article|inproceedings|misc|incollection|techreport)\s*\{', raw, re.I):
            raise ValueError('Response is not a BibTeX entry')
        filename = key + '.bib'
        (DEST / filename).write_bytes(raw)
        record.update(status='downloaded', file=filename, resolved_url=response.url,
                      sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    except Exception as error:
        record.update(status='failed', error=str(error))
    print(key + ': ' + record['status'], flush=True)
    return record


def main():
    (DEST / 'pages').mkdir(parents=True, exist_ok=True)
    old_path = DEST / 'download_manifest.json'
    old = json.loads(old_path.read_text())['records'] if old_path.exists() else []
    completed = {r['key']: r for r in old if r['status'] == 'downloaded'
                 and (DEST / r['file']).exists()
                 and hashlib.sha256((DEST / r['file']).read_bytes()).hexdigest() == r['sha256']}
    items = list(entries())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        updated = list(pool.map(download, [item for item in items if item[0] not in completed]))
    completed.update({r['key']: r for r in updated})
    records = [completed[key] for key, _ in items]
    manifest = dict(retrieved_at=datetime.now(timezone.utc).isoformat(), records=records)
    (DEST / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Downloaded', sum(r['status'] == 'downloaded' for r in records), '/', len(records), flush=True)


if __name__ == '__main__':
    main()
