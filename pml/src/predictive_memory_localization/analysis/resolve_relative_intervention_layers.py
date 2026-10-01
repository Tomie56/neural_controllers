from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoConfig

from predictive_memory_localization.common import text_num_hidden_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map reference intervention layers to normalized depths in another model."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reference-num-layers", type=int, default=28)
    parser.add_argument("--reference-layers", type=int, nargs="+", default=[-21, -17])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--format", choices=["shell", "json"], default="shell")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def map_layers(reference_count: int, reference_layers: List[int], target_count: int) -> List[Dict[str, Any]]:
    if reference_count < 2 or target_count < 2:
        raise ValueError("Both reference and target models need at least two layers")
    rows: List[Dict[str, Any]] = []
    for reference_layer in reference_layers:
        reference_index = reference_layer if reference_layer >= 0 else reference_count + reference_layer
        if reference_index < 0 or reference_index >= reference_count:
            raise ValueError(
                f"Reference layer {reference_layer} is invalid for {reference_count} layers"
            )
        normalized_depth = reference_index / (reference_count - 1)
        target_index = int(round(normalized_depth * (target_count - 1)))
        rows.append(
            {
                "reference_layer": reference_layer,
                "reference_index": reference_index,
                "normalized_depth": normalized_depth,
                "target_index": target_index,
                "target_layer": target_index - target_count,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    target_count = text_num_hidden_layers(config)
    rows = map_layers(args.reference_num_layers, args.reference_layers, target_count)
    text_config = getattr(config, "text_config", config)
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    for row in rows:
        row["target_layer_type"] = (
            layer_types[row["target_index"]] if len(layer_types) == target_count else None
        )
    payload = {
        "model_path": str(Path(args.model_path)),
        "model_type": getattr(config, "model_type", None),
        "reference_num_layers": args.reference_num_layers,
        "target_num_layers": target_count,
        "mappings": rows,
        "layers": [row["target_layer"] for row in rows],
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(" ".join(str(layer) for layer in payload["layers"]))


if __name__ == "__main__":
    main()
