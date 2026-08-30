from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
try:
    import torch
except ImportError:
    torch = None

try:
    from transformer_lens import utils
except ImportError:
    utils = None


def patch_head_output(
    model: Any,
    corrupted_tokens: torch.Tensor,
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
    head: int,
    patch_pos: Optional[Union[int, Sequence[int]]] = None,
) -> torch.Tensor:
    """Patch attention head output (z) from clean_cache into corrupted run."""
    act_name = utils.get_act_name("z", layer)

    def hook_fn(z: torch.Tensor, hook: Any) -> torch.Tensor:
        clean_z = clean_cache[act_name]
        if patch_pos is None:
            z[:, :, head, :] = clean_z[:, :, head, :]
        elif isinstance(patch_pos, int):
            z[:, patch_pos, head, :] = clean_z[:, patch_pos, head, :]
        else:
            z[:, patch_pos, head, :] = clean_z[:, patch_pos, head, :]
        return z

    return model.run_with_hooks(corrupted_tokens, fwd_hooks=[(act_name, hook_fn)])


def patch_subcomponent(
    model: Any,
    corrupted_tokens: torch.Tensor,
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
    head: int,
    component: str = "q",  # 'q', 'k', or 'v'
    patch_pos: Optional[Union[int, Sequence[int]]] = None,
) -> torch.Tensor:
    """
    Sub-component (Q/K/V) activation patching per Hong et al. Figure 6/7.
    Splices clean Q, K, or V activations for a specific head into corrupted run.
    """
    act_name = utils.get_act_name(component, layer)

    def hook_fn(act: torch.Tensor, hook: Any) -> torch.Tensor:
        clean_act = clean_cache[act_name]
        if patch_pos is None:
            act[:, :, head, :] = clean_act[:, :, head, :]
        elif isinstance(patch_pos, int):
            act[:, patch_pos, head, :] = clean_act[:, patch_pos, head, :]
        else:
            act[:, patch_pos, head, :] = clean_act[:, patch_pos, head, :]
        return act

    return model.run_with_hooks(corrupted_tokens, fwd_hooks=[(act_name, hook_fn)])


def patch_residual_stream(
    model: Any,
    corrupted_tokens: torch.Tensor,
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
    pos: int,
) -> torch.Tensor:
    """Token-wise residual stream patching (Chen et al. ?4.2)."""
    act_name = utils.get_act_name("resid_pre", layer)

    def hook_fn(resid: torch.Tensor, hook: Any) -> torch.Tensor:
        clean_resid = clean_cache[act_name]
        resid[:, pos, :] = clean_resid[:, pos, :]
        return resid

    return model.run_with_hooks(corrupted_tokens, fwd_hooks=[(act_name, hook_fn)])


def patch_mlp_region(
    model: Any,
    tokens: torch.Tensor,
    layer: int,
    region_indices: Sequence[int],
    ablation_type: str = "mean",  # 'mean' or 'zero'
) -> torch.Tensor:
    """MLP region ablation (Chen et al. ?4.1)."""
    act_name = utils.get_act_name("mlp_out", layer)

    def hook_fn(mlp_out: torch.Tensor, hook: Any) -> torch.Tensor:
        if ablation_type == "zero":
            mlp_out[:, region_indices, :] = 0.0
        else:
            mean_vec = mlp_out.mean(dim=1, keepdim=True)
            mlp_out[:, region_indices, :] = mean_vec
        return mlp_out

    return model.run_with_hooks(tokens, fwd_hooks=[(act_name, hook_fn)])


def complement_patch_heads(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupted_cache: Dict[str, torch.Tensor],
    circuit_heads: Sequence[Tuple[int, int]],
    n_layers: int,
    n_heads: int,
) -> torch.Tensor:
    """
    Complement patching (Hong et al. ?3.2, ?B.5):
    Corrupt all heads EXCEPT those in `circuit_heads`.
    """
    circuit_set = set(circuit_heads)
    hooks = []

    for layer in range(n_layers):
        heads_to_corrupt = [h for h in range(n_heads) if (layer, h) not in circuit_set]
        if not heads_to_corrupt:
            continue

        act_name = utils.get_act_name("z", layer)

        def make_hook(layer_name: str, h_list: List[int]) -> Callable:
            def hook_fn(z: torch.Tensor, hook: Any) -> torch.Tensor:
                corrupted_z = corrupted_cache[layer_name]
                z[:, :, h_list, :] = corrupted_z[:, :, h_list, :]
                return z
            return hook_fn

        hooks.append((act_name, make_hook(act_name, heads_to_corrupt)))

    return model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
