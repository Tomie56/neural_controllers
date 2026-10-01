from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a per-record CSV to ids present in a JSONL dataset.")
    parser.add_argument("--ids-jsonl", required=True, help="JSONL file with an `id` field.")
    parser.add_argument("--input-csv", required=True, help="Input per-record CSV.")
    parser.add_argument("--output-csv", required=True, help="Filtered output CSV.")
    return parser.parse_args()


def load_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" in row:
                ids.add(str(row["id"]))
    return ids


def read_rows(path: Path) -> tuple[List[str], List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ids = load_ids(Path(args.ids_jsonl))
    fieldnames, rows = read_rows(Path(args.input_csv))
    filtered = [row for row in rows if str(row.get("id", "")) in ids]
    write_rows(Path(args.output_csv), fieldnames, filtered)
    print(f"Wrote {len(filtered)} filtered rows for {len(ids)} ids to {args.output_csv}")


if __name__ == "__main__":
    main()
