from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_TARGETS = [
    "later_suppression_path",
    "later_enhancement_path",
    "later_target_any_path",
    "later_neighbor_damage_any_path",
    "later_capability_damage_any_path",
    "later_clean_suppression_path",
    "later_clean_enhancement_path",
    "later_clean_any_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize strict later-strength outcome rates without training predictors."
    )
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_float(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"true", "yes"}:
        return 1.0
    if normalized in {"false", "no", ""}:
        return 0.0
    return float(value)


def summarize(rows: Iterable[Dict[str, str]], targets: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"n_paths": 0, **{target: 0.0 for target in targets}}
    )
    overall = {"n_paths": 0, **{target: 0.0 for target in targets}}
    for row in rows:
        key = (row["method"], row["layer"])
        grouped[key]["n_paths"] += 1
        overall["n_paths"] += 1
        for target in targets:
            value = as_float(row[target])
            grouped[key][target] += value
            overall[target] += value

    output: List[Dict[str, Any]] = []
    for (method, layer), values in sorted(grouped.items()):
        n_paths = int(values["n_paths"])
        output.append(
            {
                "scope": "method_layer",
                "method": method,
                "layer": int(layer),
                "n_paths": n_paths,
                **{f"{target}_rate": values[target] / n_paths for target in targets},
            }
        )
    n_overall = int(overall["n_paths"])
    output.append(
        {
            "scope": "overall",
            "method": "all",
            "layer": "all",
            "n_paths": n_overall,
            **{f"{target}_rate": overall[target] / n_overall for target in targets},
        }
    )
    return output


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_csv)
    output_dir = Path(args.output_dir)
    output_csv = output_dir / "strict_later_outcome_summary.csv"
    manifest_path = output_dir / "strict_later_outcome_summary_manifest.json"
    if (output_csv.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Summary exists under {output_dir}. Use --overwrite to replace it.")

    with dataset_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [target for target in args.targets if target not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Dataset is missing requested targets: {missing}")
        rows = summarize(reader, args.targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "dataset_csv": str(dataset_path),
        "output_csv": str(output_csv),
        "targets": args.targets,
        "n_summary_rows": len(rows),
        "n_paths": rows[-1]["n_paths"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
