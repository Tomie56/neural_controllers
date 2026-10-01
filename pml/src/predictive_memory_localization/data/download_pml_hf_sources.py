from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


DEFAULT_SOURCES: List[Dict[str, Any]] = [
    {
        "dataset_name": "TIGER-Lab/MMLU-Pro",
        "dataset_config": "",
        "split": "test",
        "domain": "recent_multidomain_academic",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 650,
    },
    {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "dataset_config": "global_facts",
        "split": "test",
        "domain": "recent_corrected_global_facts",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 100,
    },
    {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "dataset_config": "high_school_biology",
        "split": "test",
        "domain": "recent_corrected_biology",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 100,
    },
    {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "dataset_config": "high_school_physics",
        "split": "test",
        "domain": "recent_corrected_physics",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 100,
    },
    {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "dataset_config": "college_chemistry",
        "split": "test",
        "domain": "recent_corrected_chemistry",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 100,
    },
    {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "dataset_config": "computer_security",
        "split": "test",
        "domain": "recent_corrected_computer_security",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 100,
    },
    {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "dataset_config": "machine_learning",
        "split": "test",
        "domain": "recent_corrected_machine_learning",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 100,
    },
    {
        "dataset_name": "livebench/reasoning",
        "dataset_config": "",
        "split": "test",
        "domain": "recent_objective_reasoning",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 200,
    },
    {
        "dataset_name": "livebench/math",
        "dataset_config": "",
        "split": "test",
        "domain": "recent_objective_math",
        "release_year": 2024,
        "freshness_group": "recent_2024_2026",
        "target_records": 150,
    },
    {
        "dataset_name": "allenai/openbookqa",
        "dataset_config": "additional",
        "split": "train",
        "domain": "elementary_science",
        "release_year": 2018,
        "freshness_group": "classic_pre_2024",
        "target_records": 200,
    },
    {
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Challenge",
        "split": "train",
        "domain": "science_exam",
        "release_year": 2018,
        "freshness_group": "classic_pre_2024",
        "target_records": 200,
    },
    {
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Easy",
        "split": "train",
        "domain": "science_exam",
        "release_year": 2018,
        "freshness_group": "classic_pre_2024",
        "target_records": 150,
    },
    {
        "dataset_name": "allenai/sciq",
        "dataset_config": "",
        "split": "train",
        "domain": "science_qa",
        "release_year": 2017,
        "freshness_group": "classic_pre_2024",
        "target_records": 200,
    },
    {
        "dataset_name": "allenai/qasc",
        "dataset_config": "",
        "split": "train",
        "domain": "science_composition",
        "release_year": 2019,
        "freshness_group": "classic_pre_2024",
        "target_records": 150,
    },
    {
        "dataset_name": "tau/commonsense_qa",
        "dataset_config": "",
        "split": "train",
        "domain": "commonsense",
        "release_year": 2018,
        "freshness_group": "classic_pre_2024",
        "target_records": 200,
    },
    {
        "dataset_name": "ybisk/piqa",
        "dataset_config": "",
        "split": "train",
        "domain": "physical_interaction",
        "release_year": 2019,
        "freshness_group": "classic_pre_2024",
        "target_records": 150,
    },
    {
        "dataset_name": "Rowan/hellaswag",
        "dataset_config": "",
        "split": "train",
        "domain": "situated_commonsense",
        "release_year": 2019,
        "freshness_group": "classic_pre_2024",
        "target_records": 150,
    },
    {
        "dataset_name": "google/boolq",
        "dataset_config": "",
        "split": "train",
        "domain": "wiki_boolean_facts",
        "release_year": 2019,
        "freshness_group": "classic_pre_2024",
        "target_records": 100,
    },
    {
        "dataset_name": "allenai/social_i_qa",
        "dataset_config": "",
        "split": "train",
        "domain": "social_commonsense",
        "release_year": 2019,
        "freshness_group": "classic_pre_2024",
        "target_records": 100,
        "trust_remote_code": True,
    },
    {
        "dataset_name": "EleutherAI/truthful_qa_mc",
        "dataset_config": "multiple_choice",
        "split": "validation",
        "domain": "truthfulness_facts",
        "release_year": 2021,
        "freshness_group": "classic_pre_2024",
        "target_records": 50,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw Hugging Face datasets for the fresh PML bidirectional data build."
    )
    parser.add_argument("--raw-root", default="/dev/shm/pml_datas")
    parser.add_argument("--source-spec-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue if one source fails. The seed builder will use available sources.",
    )
    return parser.parse_args()


def source_specs(path: Optional[str]) -> List[Dict[str, Any]]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return DEFAULT_SOURCES


def safe_source_name(spec: Dict[str, Any]) -> str:
    dataset = spec["dataset_name"].replace("/", "__")
    config = spec.get("dataset_config") or "default"
    split = spec.get("split", "train")
    name = f"{dataset}__{config}__{split}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_source(spec: Dict[str, Any], cache_dir: Path):
    kwargs: Dict[str, Any] = {
        "path": spec["dataset_name"],
        "split": spec.get("split", "train"),
        "cache_dir": str(cache_dir),
    }
    config = spec.get("dataset_config") or None
    if config:
        kwargs["name"] = config
    if spec.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    return load_dataset(**kwargs)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    raw_dir = raw_root / "raw"
    cache_dir = raw_root / "hf_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for spec in source_specs(args.source_spec_json):
        source_name = safe_source_name(spec)
        out_dir = raw_dir / source_name
        if out_dir.exists() and not args.overwrite:
            status = "exists"
            n_rows = None
        else:
            if out_dir.exists():
                # save_to_disk cannot overwrite in-place.
                import shutil

                shutil.rmtree(out_dir)
            try:
                dataset = load_source(spec, cache_dir)
                dataset.save_to_disk(str(out_dir))
                status = "downloaded"
                n_rows = len(dataset)
            except BaseException as exc:  # noqa: BLE001
                if not args.skip_errors:
                    raise
                status = f"failed: {exc!r}"
                n_rows = 0
        manifest_rows.append(
            {
                **spec,
                "source_name": source_name,
                "raw_path": str(out_dir),
                "status": status,
                "n_rows": n_rows,
            }
        )
        print(json.dumps(manifest_rows[-1], ensure_ascii=False))

    manifest = {
        "raw_root": str(raw_root),
        "raw_dir": str(raw_dir),
        "cache_dir": str(cache_dir),
        "sources": manifest_rows,
    }
    manifest_path = raw_root / "pml_hf_sources_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
