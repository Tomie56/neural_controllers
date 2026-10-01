from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import load_jsonl, validate_suppression_record, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter suppression JSONL to records passing strict validation.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, row in enumerate(load_jsonl(args.input_jsonl)):
        try:
            validate_suppression_record(row)
        except Exception as exc:
            errors.append({"index": idx, "id": row.get("id"), "error": repr(exc)})
            continue
        valid_rows.append(row)

    write_jsonl(args.output_jsonl, valid_rows)
    report = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "input_rows": len(valid_rows) + len(errors),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(errors),
        "errors": errors[:100],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
