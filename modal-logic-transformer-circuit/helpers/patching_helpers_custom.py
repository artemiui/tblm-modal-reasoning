from __future__ import annotations

import itertools
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None
    Tensor = Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def logits_diff(logits: Any, answer_tokens: Any, per_prompt: bool = False) -> Any:
    final_logits = logits[:, -1, :]
    if hasattr(final_logits, "cpu"):
        final_logits = final_logits.cpu()
    if hasattr(answer_tokens, "cpu"):
        answer_tokens = answer_tokens.cpu()

    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    diff = correct_logits - incorrect_logits
    return diff if per_prompt else diff.mean()


def basic_metric(
    logits: Any,
    corrupted_logit_diff: Any,
    clean_logit_diff: Any,
    answer_tokens: Any,
    normalize: bool = True,
    average: bool = True,
) -> Any:
    if average:
        patched_logit_diff = logits_diff(logits, answer_tokens)
        if normalize:
            denom = corrupted_logit_diff - clean_logit_diff
            if abs(float(denom)) < 1e-8:
                return 0.0
            return (patched_logit_diff - clean_logit_diff) / denom
        else:
            return patched_logit_diff - clean_logit_diff
    else:
        patched_logit_diff = logits_diff(logits, answer_tokens, per_prompt=True)
        return patched_logit_diff - clean_logit_diff


def patch_head(
    clean_head_vector: Any,
    hook: Any,
    head_index: int,
    corrupted_cache: Any,
    positions_l: int,
    positions_u: int,
) -> Any:
    if positions_u == 0:
        clean_head_vector[:, positions_l:, head_index] = corrupted_cache[hook.name][:, positions_l:, head_index]
    else:
        clean_head_vector[:, positions_l:positions_u, head_index] = corrupted_cache[hook.name][:, positions_l:positions_u, head_index]
    return clean_head_vector


def basic_patching(
    model: Any,
    clean_tokens: Any,
    clean_logit_diff: Any,
    corrupted_logit_diff: Any,
    corrupted_cache: Any,
    metric: Callable,
    component: Union[str, Sequence[str]],
    answer_tokens_batch: Any,
    GQA_constant: int = 1,
    positions_l: int = 0,
    positions_u: int = 0,
    batch_size: int = 1,
) -> Any:
    if isinstance(component, str):
        components = [component]
    else:
        components = list(component)

    device = getattr(model.cfg, "device", "cpu") if hasattr(model, "cfg") else "cpu"
    patched_history = torch.zeros(len(components), model.cfg.n_layers, model.cfg.n_heads, batch_size, device=device, dtype=torch.float32)

    for comp_idx, comp in enumerate(components):
        if comp in {"q", "z"}:
            upper_limit = model.cfg.n_heads
        else:
            upper_limit = max(1, int(model.cfg.n_heads / max(1, GQA_constant)))

        for (layer, head) in tqdm(list(itertools.product(range(model.cfg.n_layers), range(upper_limit))), desc=f"Sweeping component {comp}"):
            hook_fn = partial(
                patch_head,
                head_index=head,
                corrupted_cache=corrupted_cache,
                positions_l=positions_l,
                positions_u=positions_u,
            )
            act_name = f"blocks.{layer}.attn.hook_{comp}" if hasattr(model, "blocks") else f"blocks.{layer}.hook_{comp}"
            patched_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(act_name, hook_fn)],
                return_type="logits",
            )
            score = metric(patched_logits, corrupted_logit_diff, clean_logit_diff, answer_tokens=answer_tokens_batch)
            patched_history[comp_idx, layer, head] = score

    return patched_history
