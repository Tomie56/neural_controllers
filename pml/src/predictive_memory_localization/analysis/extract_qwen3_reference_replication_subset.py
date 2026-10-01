from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


DEFAULT_METHODS = ["random", "mean_difference", "logistic", "rfm_agop_top1"]
DEFAULT_LAYERS = [-21, -17]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an exact Qwen3 reference-model subset from completed outcome files."
    )
    parser.add_argument("--source-outcome-root", required=True)
    parser.add_argument("--output-outcome-root", required=True)
    parser.add_argument("--record-ids-file", required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def filter_csv(source: Path, output: Path, record_ids: set[str]) -> Dict[str, int]:
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    wrote_header = False
    selected_rows = 0
    selected_ids: set[str] = set()
    for chunk in pd.read_csv(source, dtype={"record_id": str}, chunksize=100_000):
        selected = chunk[chunk["record_id"].astype(str).isin(record_ids)].copy()
        if selected.empty:
            continue
        if "control_scale" not in selected.columns:
            selected["control_scale"] = 1.0
        if "alpha" in selected.columns and "effective_control_coef" not in selected.columns:
            selected["effective_control_coef"] = pd.to_numeric(
                selected["alpha"], errors="coerce"
            )
        selected.to_csv(tmp, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        selected_rows += len(selected)
        selected_ids.update(selected["record_id"].astype(str))
    if not wrote_header:
        raise ValueError(f"No selected records found in {source}")
    missing = sorted(record_ids - selected_ids)
    if missing:
        raise ValueError(f"CSV {source} is missing selected record IDs: {missing[:10]}")
    os.replace(tmp, output)
    return {"n_rows": selected_rows, "n_records": len(selected_ids)}


def filter_jsonl(source: Path, output: Path, record_ids: set[str]) -> Dict[str, int]:
    tmp = output.with_suffix(output.suffix + ".tmp")
    selected_rows = 0
    selected_ids: set[str] = set()
    with source.open(encoding="utf-8") as input_handle, tmp.open("w", encoding="utf-8") as out:
        for line in input_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("record_id", ""))
            if record_id not in record_ids:
                continue
            row["control_scale"] = 1.0
            alpha_rows = row.get("alphas", [])
            if alpha_rows and isinstance(alpha_rows[0], dict):
                for alpha_row in alpha_rows:
                    alpha_row["control_scale"] = 1.0
                    alpha_row["effective_control_coef"] = float(alpha_row.get("alpha", 0.0))
            else:
                row["effective_control_coefficients"] = [float(alpha) for alpha in alpha_rows]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            selected_rows += 1
            selected_ids.add(record_id)
    missing = sorted(record_ids - selected_ids)
    if missing:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"JSONL {source} is missing selected record IDs: {missing[:10]}")
    os.replace(tmp, output)
    return {"n_rows": selected_rows, "n_records": len(selected_ids)}


def layer_dir_name(layer: int) -> str:
    return f"layer_{str(layer).replace('-', 'neg')}"


def complete_manifest_matches(path: Path, expected: Dict[str, Any]) -> bool:
    if not path.exists():
        return False
    previous = json.loads(path.read_text(encoding="utf-8"))
    return all(previous.get(key) == value for key, value in expected.items())


def extract_path(
    source_dir: Path,
    output_dir: Path,
    record_ids: set[str],
    ids_path: Path,
    method: str,
    layer: int,
    overwrite: bool,
) -> Dict[str, Any]:
    required = ["per_eval.csv", "per_record.csv", "path_rows.csv", "raw_scores.jsonl"]
    missing_files = [name for name in required if not (source_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing source files under {source_dir}: {missing_files}")
    expected = {
        "source_dir": str(source_dir),
        "source_run_config_sha256": file_sha256(source_dir / "run_config.json"),
        "record_ids_file": str(ids_path),
        "record_ids_sha256": file_sha256(ids_path),
        "n_records": len(record_ids),
        "method": method,
        "layer": layer,
        "control_scale": 1.0,
        "reuse_rule": "exact_score_subset_reference_scale_equals_one",
    }
    manifest_path = output_dir / "extraction_manifest.json"
    if not overwrite and complete_manifest_matches(manifest_path, expected):
        if all((output_dir / name).is_file() for name in required):
            print(f"[reference extract] skip complete method={method} layer={layer}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        name: filter_jsonl(source_dir / name, output_dir / name, record_ids)
        if name.endswith(".jsonl")
        else filter_csv(source_dir / name, output_dir / name, record_ids)
        for name in required
    }

    source_summary = json.loads((source_dir / "method_summary.json").read_text(encoding="utf-8"))
    summary = {
        **source_summary,
        "n_records": len(record_ids),
        "n_per_record_rows": stats["per_record.csv"]["n_rows"],
        "control_scale": 1.0,
        "elapsed_seconds": 0.0,
        "source_n_records": int(source_summary.get("n_records", 0)),
        "reused_exact_scores": True,
        "source_outcome_dir": str(source_dir),
        "n_failures": 0,
    }
    atomic_write_json(output_dir / "method_summary.json", summary)

    source_config = json.loads((source_dir / "run_config.json").read_text(encoding="utf-8"))
    run_config = {
        **source_config,
        "output_dir": str(output_dir),
        "max_records": len(record_ids),
        "record_ids_file": str(ids_path),
        "control_scale": 1.0,
        "source_outcome_dir": str(source_dir),
        "reused_exact_scores": True,
    }
    atomic_write_json(output_dir / "run_config.json", run_config)
    for name in ["data_manifest.json", "model_config.json"]:
        source_path = source_dir / name
        if source_path.exists():
            (output_dir / name).write_bytes(source_path.read_bytes())
    (output_dir / "failures.jsonl").write_text("", encoding="utf-8")
    report = [
        "# Qwen3 Reference Subset Extraction",
        "",
        f"- Source: `{source_dir}`",
        f"- Records: `{len(record_ids)}`",
        f"- Method: `{method}`",
        f"- Layer: `{layer}`",
        "- Residual-RMS control scale: `1.0`",
        "- Scores recomputed: `no`",
        "- Reuse rule: exact subset of the completed reference-model run",
        "",
    ]
    (output_dir / "STAGE_RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {**expected, "files": stats, "output_dir": str(output_dir)}
    atomic_write_json(manifest_path, manifest)
    print(f"[reference extract] complete method={method} layer={layer} root={output_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_outcome_root)
    output_root = Path(args.output_outcome_root)
    ids_path = Path(args.record_ids_file)
    record_ids = {
        line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if not record_ids:
        raise ValueError(f"Record IDs file is empty: {ids_path}")
    manifests: List[Dict[str, Any]] = []
    for method in args.methods:
        for layer in args.layers:
            layer_name = layer_dir_name(layer)
            manifests.append(
                extract_path(
                    source_dir=source_root / method / layer_name,
                    output_dir=output_root / method / layer_name,
                    record_ids=record_ids,
                    ids_path=ids_path,
                    method=method,
                    layer=layer,
                    overwrite=args.overwrite,
                )
            )
    top_manifest = {
        "source_outcome_root": str(source_root),
        "output_outcome_root": str(output_root),
        "record_ids_file": str(ids_path),
        "record_ids_sha256": file_sha256(ids_path),
        "n_records": len(record_ids),
        "methods": args.methods,
        "layers": args.layers,
        "n_paths": len(manifests),
        "control_scale": 1.0,
        "reused_exact_scores": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "reference_subset_manifest.json", top_manifest)
    print(json.dumps(top_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
