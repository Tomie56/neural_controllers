#!/usr/bin/env python3
"""Build a minimal, auditable Overleaf upload bundle for the PML paper."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = PAPER_ROOT / "dist"
BUNDLE_ROOT = DIST_ROOT / "overleaf_preview"
ZIP_PATH = DIST_ROOT / "pml_iclr27_overleaf.zip"

ROOT_FILES = [
    "main.tex",
    "supplement.tex",
    "references.bib",
    "iclr2027_conference.sty",
    "iclr2027_conference.bst",
    "math_commands.tex",
    "natbib.sty",
    "fancyhdr.sty",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_sources() -> None:
    if BUNDLE_ROOT.exists():
        shutil.rmtree(BUNDLE_ROOT)
    BUNDLE_ROOT.mkdir(parents=True)
    for name in ROOT_FILES:
        shutil.copy2(PAPER_ROOT / name, BUNDLE_ROOT / name)

    # Include only the active LaTeX dependency graph, including raster panels.
    # Stale, unused tables and internal result ledgers stay out of the archive.
    pending = [PAPER_ROOT / name for name in ("main.tex", "supplement.tex")]
    visited = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        text = source.read_text()
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        for name in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
            child = PAPER_ROOT / name
            if not child.suffix:
                child = child.with_suffix(".tex")
            pending.append(child)
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            child = PAPER_ROOT / name
            assert child.is_file(), f"Missing figure: {child}"
            target = BUNDLE_ROOT / child.relative_to(PAPER_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
        target = BUNDLE_ROOT / source.relative_to(PAPER_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build_manifest() -> dict[str, object]:
    cross_path = PAPER_ROOT / "data/cross_model_replication_summary.csv"
    complete_layers = 0
    total_layers = 4
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        replication = cross.loc[cross["records"].eq(500)]
        layers = replication.groupby(["model_key", "layer"])["complete"].all()
        complete_layers = int(layers.sum())
        total_layers = int(len(layers))

    files = []
    for path in sorted(BUNDLE_ROOT.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(BUNDLE_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "main.tex",
        "entrypoints": ["main.tex", "supplement.tex"],
        "cross_model_layers_complete": complete_layers,
        "cross_model_layers_total": total_layers,
        "cross_model_completion_gate_passed": complete_layers == total_layers,
        "rendered_pdf_gate_passed": pdf_checks_current(),
        "files": files,
    }


def pdf_checks_current() -> bool:
    checks = PAPER_ROOT / "data/iclr_pdf_checks.json"
    if not checks.exists():
        return False
    report = json.loads(checks.read_text())
    return report["main_text_end_page"] <= 9 and all(
        sha256(PAPER_ROOT / f"deliverables/PML_ICLR27_{stem}.pdf") == info["sha256"]
        for stem, info in report["pdfs"].items()
    )


def build_zip() -> None:
    ZIP_PATH.unlink(missing_ok=True)
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(BUNDLE_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUNDLE_ROOT))


def main() -> None:
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    copy_sources()
    manifest = build_manifest()
    manifest_path = BUNDLE_ROOT / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_zip()
    print(f"Wrote {BUNDLE_ROOT}")
    print(f"Wrote {ZIP_PATH}")
    print(
        "Cross-model gate: "
        f"{manifest['cross_model_layers_complete']}/{manifest['cross_model_layers_total']}"
    )


if __name__ == "__main__":
    main()
