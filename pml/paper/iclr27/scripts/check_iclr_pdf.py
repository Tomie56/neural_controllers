"""Check generated ICLR PDFs without modifying them or their fonts."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

PAPER = Path(__file__).resolve().parents[1]


def output(*args):
    return subprocess.check_output(args, text=True)


def main():
    aux = (PAPER / "build_pdf_main/main.aux").read_text()
    match = re.search(r"\\newlabel\{pml:main-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    assert match, "Missing main-text end marker; compile main.tex first"
    end_page = int(match[1])
    assert end_page <= 9, f"Main text ends on page {end_page}, exceeding nine pages"
    for key, page in re.findall(r"\\newlabel\{((?:fig|tab):[^}]+)\}\{\{[^}]*\}\{(\d+)\}", aux):
        assert int(page) <= 9, f"Main-text float {key} appears after page nine"
    report = {"main_text_end_page": end_page, "pdfs": {}}
    for stem, build in [("main", "build_pdf_main/main"),
                        ("supplement", "build_pdf_supp/supplement")]:
        path = PAPER / f"deliverables/PML_ICLR27_{stem}.pdf"
        assert path.read_bytes() == (PAPER / f"{build}.pdf").read_bytes(), "Stale deliverable"
        info = output("pdfinfo", str(path))
        assert re.search(r"Page size:\s+612 x 792 pts", info), "Not Letter size"
        author = re.search(r"^Author:[ \t]*(.*)$", info, re.M)
        assert not author or not author[1].strip(), "Author metadata present"
        fonts = output("pdffonts", str(path)).splitlines()[2:]
        assert fonts and all("Type 3" not in line for line in fonts), "Type 3 font"
        for line in fonts:
            # Columns at the right: emb, sub, uni, object ID (two integers).
            assert line.split()[-5] == "yes", f"Unembedded font: {line}"
            assert "Type 1" in line or "TrueType" in line, f"Unexpected font: {line}"
        text = output("pdftotext", "-layout", str(path), "-")
        assert "Anonymous authors" in text and "Under review as a conference paper at ICLR 2027" in text
        log = (PAPER / f"{build}.log").read_text()
        assert not re.search(r"undefined citations|undefined references|(?:Citation|Reference) .* undefined|Overfull \\[hv]box", log)
        report["pdfs"][stem] = {
            "pages": int(re.search(r"Pages:\s+(\d+)", info)[1]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "letter": True, "fonts_embedded_type1_or_truetype": True,
            "anonymous_header_and_metadata": True,
            "no_unresolved_references_or_overfull_boxes": True,
        }
    (PAPER / "data/iclr_pdf_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
