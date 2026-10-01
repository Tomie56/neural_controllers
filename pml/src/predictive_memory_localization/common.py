from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype | str]:
    if dtype_name is None:
        return None
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model_and_tokenizer(
    model_path: str,
    device_map: str = "cuda",
    torch_dtype: Optional[str] = None,
    model_loader: str = "auto",
    trust_remote_code: bool = False,
):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    architectures = list(getattr(config, "architectures", []) or [])
    if model_loader == "auto":
        is_image_text = getattr(config, "model_type", "") in {"qwen3_5", "mistral3"} or any(
            architecture.endswith("ForConditionalGeneration") for architecture in architectures
        )
        resolved_loader = "image_text_to_text" if is_image_text else "causal_lm"
    else:
        resolved_loader = model_loader
    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    dtype = resolve_torch_dtype(torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    if resolved_loader == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    elif resolved_loader == "image_text_to_text":
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model loader: {model_loader}")
    use_fast = "LlamaForCausalLM" not in getattr(model.config, "architectures", [])
    tokenizer_kwargs = {
        "use_fast": use_fast,
        "padding_side": "left",
        "legacy": False,
        "trust_remote_code": trust_remote_code,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    except TypeError:
        tokenizer_kwargs.pop("legacy", None)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    setattr(model, "_pml_model_loader", resolved_loader)
    return model.eval(), tokenizer


def resolve_model_layers(model):
    candidates = [
        ("model.layers", lambda value: value.model.layers),
        ("model.language_model.layers", lambda value: value.model.language_model.layers),
        ("language_model.layers", lambda value: value.language_model.layers),
        ("language_model.model.layers", lambda value: value.language_model.model.layers),
    ]
    for path, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers, path
    raise AttributeError(
        f"Unable to locate transformer/text layers for model class {type(model).__name__}"
    )


def text_num_hidden_layers(config) -> int:
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "num_hidden_layers", None)
    if value is None:
        value = getattr(config, "num_hidden_layers", None)
    if value is None:
        raise ValueError(f"Unable to determine text layer count from {type(config).__name__}")
    return int(value)


def maybe_format_prompt(tokenizer, prompt: str, enabled: bool) -> str:
    if not enabled:
        return prompt
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    ).strip()


@torch.no_grad()
def continuation_logprob(model, tokenizer, prompt: str, continuation: str) -> Dict[str, Any]:
    """Teacher-force the continuation and return total and per-token log-probability."""
    full_text = prompt + continuation
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Continuation produced no additional tokens. Check whitespace/tokenization.")

    input_ids = full_ids[:-1].unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    target_ids = full_ids[1:].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(outputs.logits[0].float(), dim=-1)

    start = max(len(prompt_ids) - 1, 0)
    end = len(full_ids) - 1
    continuation_log_probs = log_probs[start:end]
    continuation_targets = target_ids[start:end]

    token_log_probs = continuation_log_probs.gather(
        dim=-1,
        index=continuation_targets.unsqueeze(-1),
    ).squeeze(-1)

    return {
        "total_logprob": float(token_log_probs.sum().detach().cpu()),
        "mean_logprob": float(token_log_probs.mean().detach().cpu()),
        "num_tokens": int(token_log_probs.numel()),
    }


def validate_suppression_record(record: Dict[str, Any]) -> None:
    required = ["id", "train_positive", "train_negative", "eval"]
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f"Record {record.get('id', '<unknown>')} missing fields: {missing}")
    if not record["train_positive"] or not record["train_negative"]:
        raise ValueError(f"Record {record['id']} needs nonempty train_positive and train_negative")
    for item in record["eval"]:
        if "prompt" not in item or "target" not in item:
            raise ValueError(f"Record {record['id']} eval items need prompt and target")
        if not str(item["prompt"]).strip() or not str(item["target"]).strip():
            raise ValueError(f"Record {record['id']} eval items need nonempty prompt and target")
    for item in record.get("neighbors", []):
        if "prompt" not in item or "target" not in item:
            raise ValueError(f"Record {record['id']} neighbor items need prompt and target")
        if not str(item["prompt"]).strip() or not str(item["target"]).strip():
            raise ValueError(f"Record {record['id']} neighbor items need nonempty prompt and target")
