from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def source_key(dataset: Any, source_id: Any) -> str:
    return f"{dataset}::{source_id}"


def seed_key(row: Dict[str, Any]) -> str:
    return source_key(row.get("dataset_name"), row.get("source_id"))


def suppression_key(row: Dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return source_key(source.get("dataset"), source.get("source_id"))


def nonempty(value: Any) -> bool:
    return bool(str(value or "").strip())


def audit(rows: List[Dict[str, Any]], *, kind: str) -> Dict[str, Any]:
    keys = Counter()
    source_keys = Counter()
    fact_missing = 0
    fact_empty = 0
    source_fact_missing = 0
    source_fact_empty = 0
    for row in rows:
        keys.update(row.keys())
        if "fact_or_concept" not in row:
            fact_missing += 1
        elif not nonempty(row.get("fact_or_concept")):
            fact_empty += 1
        source = row.get("source")
        if isinstance(source, dict):
            source_keys.update(source.keys())
            if "fact_or_concept" not in source:
                source_fact_missing += 1
            elif not nonempty(source.get("fact_or_concept")):
                source_fact_empty += 1

    return {
        "kind": kind,
        "rows": len(rows),
        "keys": sorted(keys),
        "source_keys": sorted(source_keys),
        "fact_or_concept_missing": fact_missing,
        "fact_or_concept_empty": fact_empty,
        "source_fact_or_concept_missing": source_fact_missing,
        "source_fact_or_concept_empty": source_fact_empty,
    }


def build_concept_maps(
    seed_rows: List[Dict[str, Any]],
    suppression_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    source_to_concept: Dict[str, str] = {}
    source_to_subject: Dict[str, str] = {}

    for row in suppression_rows:
        key = suppression_key(row)
        concept = str(row.get("fact_or_concept") or row.get("concept") or "").strip()
        if concept:
            source_to_concept.setdefault(key, concept)
        subject = str(row.get("subject") or "").strip()
        if subject:
            source_to_subject.setdefault(key, subject)

    for row in seed_rows:
        key = seed_key(row)
        concept = str(row.get("fact_or_concept") or "").strip()
        if concept:
            source_to_concept.setdefault(key, concept)

    return source_to_concept, source_to_subject


def backup(path: Path, suffix: str) -> Path:
    backup_path = path.with_suffix(path.suffix + suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Align commonsense seed/suppression schema and fill fact_or_concept.")
    parser.add_argument(
        "--seed-jsonl",
        type=Path,
        default=Path("/data/neural_controllers/pml/data/commonsense_seed_mix_3000.jsonl"),
    )
    parser.add_argument(
        "--suppression-jsonl",
        type=Path,
        default=Path("/data/neural_controllers/pml/data/suppression_commonsense_3000.jsonl"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("/data/neural_controllers/pml/data/commonsense_schema_alignment_report.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    seed_rows = load_jsonl(args.seed_jsonl)
    suppression_rows = load_jsonl(args.suppression_jsonl)
    before = {
        "seed": audit(seed_rows, kind="seed"),
        "suppression": audit(suppression_rows, kind="suppression"),
    }

    source_to_concept, source_to_subject = build_concept_maps(seed_rows, suppression_rows)

    seed_filled = 0
    seed_unfilled: List[str] = []
    for row in seed_rows:
        key = seed_key(row)
        if not nonempty(row.get("fact_or_concept")):
            concept = source_to_concept.get(key, "")
            if concept:
                row["fact_or_concept"] = concept
                seed_filled += 1
            else:
                seed_unfilled.append(key)

    suppression_top_added = 0
    suppression_source_added = 0
    suppression_unfilled: List[str] = []
    for row in suppression_rows:
        key = suppression_key(row)
        concept = str(row.get("fact_or_concept") or row.get("concept") or source_to_concept.get(key, "")).strip()
        if not concept:
            suppression_unfilled.append(key)
            continue
        if not nonempty(row.get("fact_or_concept")):
            row["fact_or_concept"] = concept
            suppression_top_added += 1
        source = row.setdefault("source", {})
        if isinstance(source, dict) and not nonempty(source.get("fact_or_concept")):
            source["fact_or_concept"] = concept
            suppression_source_added += 1
        if isinstance(source, dict) and key in source_to_subject and not nonempty(source.get("subject")):
            source["subject"] = source_to_subject[key]

    after = {
        "seed": audit(seed_rows, kind="seed"),
        "suppression": audit(suppression_rows, kind="suppression"),
    }
    report = {
        "seed_jsonl": str(args.seed_jsonl),
        "suppression_jsonl": str(args.suppression_jsonl),
        "dry_run": args.dry_run,
        "before": before,
        "after": after,
        "changes": {
            "seed_fact_or_concept_filled": seed_filled,
            "suppression_top_fact_or_concept_added": suppression_top_added,
            "suppression_source_fact_or_concept_added": suppression_source_added,
            "seed_unfilled_count": len(seed_unfilled),
            "suppression_unfilled_count": len(suppression_unfilled),
            "seed_unfilled_examples": seed_unfilled[:20],
            "suppression_unfilled_examples": suppression_unfilled[:20],
        },
    }

    if not args.dry_run:
        backups = {}
        if not args.no_backup:
            backups["seed"] = str(backup(args.seed_jsonl, ".pre_schema_align.bak"))
            backups["suppression"] = str(backup(args.suppression_jsonl, ".pre_schema_align.bak"))
        write_jsonl(args.seed_jsonl, seed_rows)
        write_jsonl(args.suppression_jsonl, suppression_rows)
        report["backups"] = backups

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
