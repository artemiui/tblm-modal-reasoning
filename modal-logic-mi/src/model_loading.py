from __future__ import annotations

import argparse
import os
from typing import Any, List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None


def add_model_source_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_source",
        choices=["huggingface", "modelscope", "local"],
        default=os.environ.get("MODEL_SOURCE", "huggingface"),
        help="Backend source for downloading/loading model checkpoints.",
    )


def resolve_true_false_token_ids(model: Any) -> Tuple[int, int]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return 1, 0

    candidates_true = [" True", "True", "true", "TRUE", "T"]
    candidates_false = [" False", "False", "false", "FALSE", "F"]

    true_id = None
    for cand in candidates_true:
        tokens = tokenizer.encode(cand, add_special_tokens=False)
        if len(tokens) == 1:
            true_id = tokens[0]
            break
    if true_id is None:
        true_id = tokenizer.encode("True", add_special_tokens=False)[-1]

    false_id = None
    for cand in candidates_false:
        tokens = tokenizer.encode(cand, add_special_tokens=False)
        if len(tokens) == 1:
            false_id = tokens[0]
            break
    if false_id is None:
        false_id = tokenizer.encode("False", add_special_tokens=False)[-1]

    return int(true_id), int(false_id)


def resolve_model_prompt(model: Any, prompt: str, enable_thinking: bool = False) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(rendered)
    except Exception:
        return prompt


def to_tokens(model: Any, text: str, prepend_bos: bool = True) -> Any:
    if hasattr(model, "to_tokens"):
        return model.to_tokens(text, prepend_bos=prepend_bos)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Model has no tokenizer attached")
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=prepend_bos)
    return enc["input_ids"]


def load_hooked_transformer(
    model_id: str,
    device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
    source: str = "huggingface",
    torch_dtype: Any = None,
    error_context: str = "",
    device_map: Optional[str] = None,
) -> Any:
    if HookedTransformer is None:
        raise ImportError("transformer_lens is required to load HookedTransformer models.")

    hf_model_id = model_id
    if source == "modelscope":
        try:
            from modelscope import snapshot_download
            hf_model_id = snapshot_download(model_id)
        except ImportError:
            pass

    dtype_val = torch_dtype if torch_dtype is not None else (torch.float16 if (torch is not None and device != "cpu") else torch.float32)

    load_kwargs: dict[str, Any] = {
        "device": device,
        "dtype": dtype_val,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    try:
        model = HookedTransformer.from_pretrained(hf_model_id, **load_kwargs)
    except Exception as e:
        if device == "cpu" or (torch is not None and not torch.cuda.is_available()):
            model = HookedTransformer.from_pretrained(hf_model_id, device="cpu", dtype=torch.float32)
        else:
            raise RuntimeError(f"Failed loading model {model_id} ({error_context}): {e}") from e

    return model


def get_gqa_group_mapping(n_heads: int, n_kv_heads: int) -> dict[int, int]:
    if n_kv_heads <= 0 or n_kv_heads >= n_heads:
        return {h: h for h in range(n_heads)}
    group_size = n_heads // n_kv_heads
    return {h: h // group_size for h in range(n_heads)}
