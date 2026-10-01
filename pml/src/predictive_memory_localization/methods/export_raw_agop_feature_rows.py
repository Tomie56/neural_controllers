from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw AGOP feature CSV as editing-analysis feature rows."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def should_keep(key: str) -> bool:
    return key.startswith("agop_")


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    exported: List[Dict[str, Any]] = []
    for row in rows:
        out: Dict[str, Any] = {
            "id": row.get("id", ""),
            "concept": row.get("concept", ""),
            "source_dataset": row.get("source_dataset", ""),
        }
        for key, value in row.items():
            if should_keep(key):
                out[f"feature_{key}"] = value
        exported.append(out)

    fieldnames = sorted({key for row in exported for key in row})
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exported)
    print(f"Wrote raw AGOP editing feature rows to {args.output_csv}")


if __name__ == "__main__":
    main()
