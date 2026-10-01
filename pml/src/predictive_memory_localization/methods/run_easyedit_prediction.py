from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_memory_localization.common import append_jsonl, continuation_logprob, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ROME/MEMIT editing tasks and save prediction-ready results. "
            "This runner calls EasyEdit's algorithm modules directly to avoid "
            "the optional EasyEdit trainer/multimodal dependency chain."
        )
    )
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["ROME", "MEMIT"], required=True)
    parser.add_argument("--model-name", default="Qwen3-1.7B-Base")
    parser.add_argument("--model-local-path", default="/data/Rollout-Steering-GRPO/Qwen3-1.7B-Base")
    parser.add_argument("--hparams-dir", default=None)
    parser.add_argument("--hparams-file", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="ROME/MEMIT update math is most robust in float32 for this EasyEdit version.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-missing-subject",
        action="store_true",
        help="Skip tasks whose subject phrase is not in the primary prompt.",
    )
    parser.add_argument(
        "--memit-cov-source",
        choices=["task_prompts", "easyedit_default"],
        default="task_prompts",
        help=(
            "For MEMIT, estimate covariance from local editing-task prompts or "
            "let EasyEdit load its default Wikipedia/WikiText statistics."
        ),
    )
    parser.add_argument(
        "--memit-cov-max-texts",
        type=int,
        default=512,
        help="Maximum local task texts used to estimate MEMIT covariance.",
    )
    parser.add_argument(
        "--include-attack-prompts",
        action="store_true",
        help="Evaluate reversal/attack prompts for new and old answers before and after editing.",
    )
    parser.add_argument(
        "--attack-prompt-limit",
        type=int,
        default=6,
        help="Maximum attack prompt templates per task when --include-attack-prompts is set.",
    )
    return parser.parse_args()


def install_transformers_compat_shims() -> None:
    import transformers.modeling_utils as modeling_utils
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - already_pruned_heads
            mask = torch.ones(n_heads, head_size)
            heads = sorted(heads)
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def install_easyeditor_package_stubs() -> Path:
    root = Path("/data/miniconda3/envs/rfm/lib/python3.10/site-packages/easyeditor")
    if not root.exists():
        raise RuntimeError(f"EasyEdit package directory not found: {root}")

    package_paths = {
        "easyeditor": root,
        "easyeditor.models": root / "models",
        "easyeditor.models.rome": root / "models" / "rome",
        "easyeditor.models.memit": root / "models" / "memit",
    }
    for name, path in package_paths.items():
        module = sys.modules.get(name)
        if module is None:
            module = ModuleType(name)
            sys.modules[name] = module
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        module.__package__ = name

    util_name = "easyeditor.util"
    if util_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            util_name,
            root / "util" / "__init__.py",
            submodule_search_locations=[str(root / "util")],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not build EasyEdit util import spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[util_name] = module
        spec.loader.exec_module(module)
    return root


def load_algorithm(method: str):
    install_transformers_compat_shims()
    install_easyeditor_package_stubs()
    if method == "ROME":
        from easyeditor.models.rome.rome_hparams import ROMEHyperParams
        from easyeditor.models.rome.rome_main import apply_rome_to_model

        return ROMEHyperParams, apply_rome_to_model
    from easyeditor.models.memit.memit_hparams import MEMITHyperParams
    from easyeditor.models.memit.memit_main import apply_memit_to_model

    return MEMITHyperParams, apply_memit_to_model


def resolve_hparams_path(args: argparse.Namespace) -> str:
    if args.hparams_file:
        return args.hparams_file
    if args.hparams_dir:
        candidate = Path(args.hparams_dir)
        if candidate.suffix == ".yaml":
            return str(candidate)
        return str(candidate / f"{args.method.lower()}_qwen3_1_7b.yaml")
    default = Path(__file__).resolve().parent / "hparams" / f"{args.method.lower()}_qwen3_1_7b.yaml"
    return str(default)


def load_hparams(args: argparse.Namespace):
    hparams_cls, apply_fn = load_algorithm(args.method)
    hparams_path = resolve_hparams_path(args)
    hparams = hparams_cls.from_hparams(hparams_path)
    hparams.model_name = args.model_local_path
    hparams.device = args.device
    return hparams, apply_fn, hparams_path


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_local_path,
        device_map=f"cuda:{args.device}" if torch.cuda.is_available() else "cpu",
        torch_dtype=dtype_map[args.torch_dtype],
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_local_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def collect_covariance_texts(tasks: List[Dict[str, Any]], limit: int) -> List[str]:
    texts: List[str] = []
    for task in tasks:
        texts.append(str(task.get("prompt", "")))
        texts.extend(str(x) for x in task.get("paraphrase_prompts", []))
        for item in task.get("locality_prompts", []):
            texts.append(str(item.get("prompt", "")))
        if len(texts) >= limit:
            break
    return [text for text in texts[:limit] if text.strip()]


@torch.no_grad()
def estimate_second_moment_from_texts(
    model,
    tokenizer,
    layer_name: str,
    texts: List[str],
    device: str,
) -> torch.Tensor:
    install_easyeditor_package_stubs()
    from easyeditor.util.nethook import Trace

    module = dict(model.named_modules())[layer_name]
    in_features = module.weight.shape[1]
    total = torch.zeros((in_features, in_features), dtype=torch.float64)
    count = 0

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            add_special_tokens=False,
        )
        if encoded.input_ids.numel() == 0:
            continue
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with Trace(model, layer_name, retain_input=True, retain_output=False, stop=True) as trace:
            model(**encoded)
        feats = trace.input
        if isinstance(feats, tuple):
            feats = feats[0]
        feats = feats.detach().float().reshape(-1, feats.shape[-1]).cpu()
        total += feats.double().T @ feats.double()
        count += feats.shape[0]

    if count == 0:
        raise RuntimeError("Could not estimate MEMIT covariance: no tokens collected")
    cov = total / count
    cov += torch.eye(cov.shape[0], dtype=cov.dtype) * 1e-4
    return cov.float()


def install_local_memit_covariance(model, tokenizer, hparams, tasks: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if args.method != "MEMIT" or args.memit_cov_source != "task_prompts":
        return
    texts = collect_covariance_texts(tasks, args.memit_cov_max_texts)
    if not texts:
        raise RuntimeError("No local texts available for MEMIT covariance estimation")

    import importlib

    memit_main = importlib.import_module("easyeditor.models.memit.memit_main")

    model_name = model.config._name_or_path.replace("/", "_")
    for layer in hparams.layers:
        layer_name = hparams.rewrite_module_tmp.format(layer)
        key = (model_name, layer_name)
        if key in memit_main.COV_CACHE:
            continue
        cov = estimate_second_moment_from_texts(
            model,
            tokenizer,
            layer_name,
            texts,
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu",
        )
        memit_main.COV_CACHE[key] = cov
        print(
            f"Installed local MEMIT covariance for {layer_name} from "
            f"{len(texts)} task texts: shape={tuple(cov.shape)}"
        )


def _hidden_tensor(value):
    return value[0] if isinstance(value, tuple) else value


def patched_memit_compute_z(
    model,
    tok,
    request: Dict[str, Any],
    hparams,
    layer: int,
    context_templates: List[List[str]],
) -> torch.Tensor:
    """MEMIT compute_z with support for decoder layers that return a tensor."""
    install_easyeditor_package_stubs()
    from easyeditor.models.memit.compute_z import find_fact_lookup_idx
    from easyeditor.util import nethook

    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")
    target_ids = tok(request["target_new"], return_tensors="pt").to(f"cuda:{hparams.device}")[
        "input_ids"
    ][0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt,
            request["subject"],
            tok,
            hparams.fact_token,
            verbose=(i == 0),
        )
        for i, prompt in enumerate(all_prompts)
    ]

    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    if hasattr(model.config, "n_embd"):
        hidden_size = model.config.n_embd
    elif hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
    else:
        raise NotImplementedError
    delta = torch.zeros((hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.layer_module_tmp.format(layer):
            layer_out = _hidden_tensor(cur_out)
            if target_init is None:
                print("Recording initial value of v*")
                target_init = layer_out[0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                layer_out[i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        full_repr = _hidden_tensor(tr[hparams.layer_module_tmp.format(loss_layer)].output)[
            : len(rewriting_prompts)
        ]
        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device),
            dim=2,
        )
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0)
            .unsqueeze(2)
            .to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init,
            kl_log_probs,
            log_target=True,
            reduction="batchmean",
        )
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + "
            f"{np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] {torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2 or it == hparams.v_num_grad_steps - 1:
            break
        loss.backward()
        opt.step()
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}")
    return target


def patched_rome_compute_v(
    model,
    tok,
    request: Dict[str, Any],
    hparams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """ROME compute_v with CPU support for environments without CUDA."""
    install_easyeditor_package_stubs()
    from easyeditor.models.rome.compute_v import find_fact_lookup_idx, get_module_input_output_at_word
    from easyeditor.util import nethook

    device = next(model.parameters()).device
    print("Computing right vector (v)")

    target_ids = tok(request["target_new"], return_tensors="pt").to(device)["input_ids"][0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ]
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)

    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt,
            request["subject"],
            tok,
            hparams.fact_token,
            verbose=(i == 0),
        )
        for i, prompt in enumerate(all_prompts)
    ]

    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    if hasattr(model.config, "n_embd"):
        hidden_size = model.config.n_embd
    elif hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
    else:
        raise NotImplementedError
    delta = torch.zeros((hidden_size,), requires_grad=True, device=device)
    target_init, kl_distr_init = None, None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            layer_out = _hidden_tensor(cur_out)
            if target_init is None:
                print("Recording initial value of v*")
                target_init = layer_out[0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                layer_out[i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        log_probs = torch.log_softmax(logits, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init,
            kl_log_probs,
            log_target=True,
            reduction="batchmean",
        )
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + "
            f"{np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] {torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2 or it == hparams.v_num_grad_steps - 1:
            break
        loss.backward()
        opt.step()
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )
    cur_input = cur_input.to(left_vector.device)
    cur_output = cur_output.to(target.device)
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} "
        f"=> {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")
    return right_vector


def install_rome_qwen3_compat_patch(args: argparse.Namespace) -> None:
    if args.method != "ROME":
        return
    import importlib

    rome_main = importlib.import_module("easyeditor.models.rome.rome_main")
    rome_compute_v = importlib.import_module("easyeditor.models.rome.compute_v")
    rome_main.compute_v = patched_rome_compute_v
    rome_compute_v.compute_v = patched_rome_compute_v


def install_memit_qwen3_compat_patch(args: argparse.Namespace) -> None:
    if args.method != "MEMIT":
        return
    import importlib

    memit_main = importlib.import_module("easyeditor.models.memit.memit_main")
    memit_main.compute_z = patched_memit_compute_z


def task_to_request(task: Dict[str, Any], case_id: int) -> Dict[str, Any]:
    subject = task.get("subject") or task.get("concept", task["id"])
    prompt = task["prompt"]
    lower_prompt = prompt.lower()
    lower_subject = str(subject).lower()
    start = lower_prompt.find(lower_subject)
    if start >= 0:
        subject = prompt[start : start + len(str(subject))]
        prefix = prompt[:start].replace("{", "{{").replace("}", "}}")
        suffix = prompt[start + len(str(subject)) :].replace("{", "{{").replace("}", "}}")
        prompt = f"{prefix}{{}}{suffix}"
    else:
        prompt = prompt.replace("{", "{{").replace("}", "}}")
    return {
        "case_id": case_id,
        "prompt": prompt,
        "subject": subject,
        "target_new": task["target_new"],
        "ground_truth": task["ground_truth"],
    }


def load_completed_ids(*paths: Path) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(str(path)):
            row_id = row.get("id")
            if row_id:
                completed.add(str(row_id))
    return completed


def task_eval_record(task: Dict[str, Any]) -> Dict[str, Any]:
    locality = {}
    for idx, item in enumerate(task.get("locality_prompts", [])):
        locality[f"neighbor_{idx}"] = {
            "prompt": item["prompt"],
            "ground_truth": item["ground_truth"],
        }
    return {
        "rewrite": [{"prompt": task["prompt"], "target": task["target_new"]}],
        "old_answer": [{"prompt": task["prompt"], "target": task["ground_truth"]}],
        "rephrase": [
            {"prompt": prompt, "target": task["target_new"]}
            for prompt in task.get("paraphrase_prompts", [])
        ],
        "locality": [
            {"prompt": item["prompt"], "target": item["ground_truth"]}
            for item in task.get("locality_prompts", [])
        ],
    }


def filled_prompt(task: Dict[str, Any]) -> str:
    prompt = str(task["prompt"])
    subject = str(task.get("subject") or task.get("concept") or task.get("id") or "")
    if "{}" in prompt:
        return prompt.format(subject)
    return prompt


def attack_prompts_for_task(task: Dict[str, Any], limit: int) -> List[str]:
    base_prompt = filled_prompt(task).strip()
    subject = str(task.get("subject") or task.get("concept") or task.get("id") or "").strip()
    concept = str(task.get("concept") or subject).strip()
    target_new = str(task.get("target_new", "")).strip()
    ground_truth = str(task.get("ground_truth", "")).strip()
    templates = [
        "{base} Answer according to the model's original knowledge, not the edit:",
        "{base} Ignore any recent update and answer with the original fact:",
        "{base} Before any correction, the answer was:",
        "For the question about {subject}, give the original answer:",
        "A common prior answer for {concept}, before the update, is:",
        "If the edit is reversed, {base} The answer is:",
        "{base} Some sources still say the answer is:",
        "{base} In the unedited model, the answer would be:",
    ]
    prompts: List[str] = []
    for template in templates:
        text = template.format(
            base=base_prompt,
            subject=subject,
            concept=concept,
            target_new=target_new,
            ground_truth=ground_truth,
        )
        text = " ".join(text.split())
        if text and text not in prompts:
            prompts.append(text)
        if len(prompts) >= limit:
            break
    return prompts


def attack_eval_record(task: Dict[str, Any], limit: int) -> Dict[str, Any]:
    prompts = attack_prompts_for_task(task, limit)
    return {
        "attack_new": [{"prompt": prompt, "target": task["target_new"]} for prompt in prompts],
        "attack_old": [{"prompt": prompt, "target": task["ground_truth"]} for prompt in prompts],
    }


def score_items(model, tokenizer, items: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    scores = []
    for item in items:
        metric = continuation_logprob(model, tokenizer, item["prompt"], " " + item["target"].strip())
        scores.append({"prompt": item["prompt"], "target": item["target"], **metric})
    return scores


def evaluate_task(model, tokenizer, task: Dict[str, Any], *, include_attack_prompts: bool = False, attack_prompt_limit: int = 6) -> Dict[str, Any]:
    record = task_eval_record(task)
    if include_attack_prompts:
        record.update(attack_eval_record(task, attack_prompt_limit))
    out: Dict[str, Any] = {}
    for key, items in record.items():
        out[key] = score_items(model, tokenizer, items)
    if out["rewrite"]:
        rewrite_lp = out["rewrite"][0]["total_logprob"]
        old_lp = out["old_answer"][0]["total_logprob"] if out["old_answer"] else None
        out["rewrite_beats_old"] = bool(old_lp is not None and rewrite_lp > old_lp)
        out["rewrite_minus_old_logprob"] = None if old_lp is None else rewrite_lp - old_lp
    if include_attack_prompts and out.get("attack_new") and out.get("attack_old"):
        attack_margins = []
        for new_item, old_item in zip(out["attack_new"], out["attack_old"]):
            attack_margins.append(new_item["total_logprob"] - old_item["total_logprob"])
        out["attack_new_beats_old_rate"] = sum(1.0 for margin in attack_margins if margin > 0) / len(attack_margins)
        out["mean_attack_new_minus_old_logprob"] = sum(attack_margins) / len(attack_margins)
    return out


def restore_weights(model, weights_copy: Dict[str, torch.Tensor]) -> None:
    if not weights_copy:
        return
    install_transformers_compat_shims()
    from easyeditor.util import nethook

    with torch.no_grad():
        for name, value in weights_copy.items():
            nethook.get_parameter(model, name)[...] = value.to(nethook.get_parameter(model, name).device)


def summarize_metrics(pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    pre_margin = pre.get("rewrite_minus_old_logprob")
    post_margin = post.get("rewrite_minus_old_logprob")
    locality_deltas = []
    for before, after in zip(pre.get("locality", []), post.get("locality", [])):
        locality_deltas.append(after["total_logprob"] - before["total_logprob"])
    rephrase_deltas = []
    for before, after in zip(pre.get("rephrase", []), post.get("rephrase", [])):
        rephrase_deltas.append(after["total_logprob"] - before["total_logprob"])
    locality_preserved = [1.0 if delta >= -1.0 else 0.0 for delta in locality_deltas]
    attack_margin_delta = None
    attack_success_rate = post.get("attack_new_beats_old_rate")
    old_resurgence_rate = None
    if post.get("mean_attack_new_minus_old_logprob") is not None:
        pre_attack_margin = pre.get("mean_attack_new_minus_old_logprob")
        post_attack_margin = post.get("mean_attack_new_minus_old_logprob")
        if pre_attack_margin is not None:
            attack_margin_delta = post_attack_margin - pre_attack_margin
        old_resurgence_rate = 1.0 - float(attack_success_rate or 0.0)
    return {
        "pre_rewrite_beats_old": pre.get("rewrite_beats_old"),
        "post_rewrite_beats_old": post.get("rewrite_beats_old"),
        "rewrite_margin_delta": None
        if pre_margin is None or post_margin is None
        else post_margin - pre_margin,
        "mean_rephrase_margin_delta": None
        if not rephrase_deltas
        else sum(rephrase_deltas) / len(rephrase_deltas),
        "post_rephrase_beats_old_rate": None,
        "mean_locality_logprob_delta": None
        if not locality_deltas
        else sum(locality_deltas) / len(locality_deltas),
        "mean_locality_preserved": None
        if not locality_preserved
        else sum(locality_preserved) / len(locality_preserved),
        "post_attack_new_beats_old_rate": attack_success_rate,
        "mean_attack_margin_delta": attack_margin_delta,
        "post_attack_old_resurgence_rate": old_resurgence_rate,
        "num_rephrase_prompts": len(rephrase_deltas),
        "num_locality_prompts": len(locality_deltas),
        "num_attack_prompts": len(post.get("attack_new", [])) if isinstance(post.get("attack_new"), list) else 0,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.method.lower()}_editing_results.jsonl"
    failures_path = out_dir / f"{args.method.lower()}_editing_failures.jsonl"
    config_path = out_dir / f"{args.method.lower()}_config.json"

    if args.overwrite:
        results_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)
    completed_ids = set() if args.overwrite else load_completed_ids(results_path)

    hparams, apply_fn, hparams_path = load_hparams(args)
    install_rome_qwen3_compat_patch(args)
    install_memit_qwen3_compat_patch(args)
    config = vars(args) | {"resolved_hparams_path": hparams_path}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    tasks = list(load_jsonl(args.tasks_jsonl))
    if args.max_records is not None:
        tasks = tasks[: args.max_records]

    model, tokenizer = load_model_and_tokenizer(args)
    install_local_memit_covariance(model, tokenizer, hparams, tasks, args)

    for case_id, task in enumerate(tasks):
        if str(task.get("id", "")) in completed_ids:
            continue
        try:
            subject = task.get("subject") or task.get("concept", task.get("id", ""))
            if args.skip_missing_subject and subject.lower() not in task["prompt"].lower():
                append_jsonl(
                    failures_path,
                    {
                        "id": task.get("id"),
                        "method": args.method,
                        "error": f"Skipped: subject {subject!r} not in prompt",
                    },
                )
                continue

            request = task_to_request(task, case_id)
            pre_metrics = evaluate_task(
                model,
                tokenizer,
                task,
                include_attack_prompts=args.include_attack_prompts,
                attack_prompt_limit=args.attack_prompt_limit,
            )
            if args.method == "ROME":
                _, weights_copy = apply_fn(
                    model,
                    tokenizer,
                    [request],
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=True,
                )
            else:
                _, weights_copy = apply_fn(
                    model,
                    tokenizer,
                    [request],
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=True,
                )
            post_metrics = evaluate_task(
                model,
                tokenizer,
                task,
                include_attack_prompts=args.include_attack_prompts,
                attack_prompt_limit=args.attack_prompt_limit,
            )
            restore_weights(model, weights_copy)
            append_jsonl(
                results_path,
                {
                    "id": task["id"],
                    "concept": task.get("concept"),
                    "subject": subject,
                    "method": args.method,
                    "task": task,
                    "request": request,
                    "pre_metrics": pre_metrics,
                    "post_metrics": post_metrics,
                    "metrics": summarize_metrics(pre_metrics, post_metrics),
                },
            )
        except Exception as exc:
            append_jsonl(
                failures_path,
                {
                    "id": task.get("id"),
                    "method": args.method,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"Wrote editing results to {results_path}")
    if failures_path.exists():
        print(f"Wrote editing failures to {failures_path}")


if __name__ == "__main__":
    main()
