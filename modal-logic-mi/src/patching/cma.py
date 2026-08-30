from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
try:
    import torch
except ImportError:
    torch = None
from .activation_patch import patch_head_output, complement_patch_heads
from .metrics import compute_logit_diff, compute_calibrated_ld


def cma_necessity_sweep(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    clean_cache: Dict[str, torch.Tensor],
    corrupted_cache: Dict[str, torch.Tensor],
    correct_id: int,
    incorrect_id: int,
    n_layers: int,
    n_heads: int,
    pos: int = -1,
) -> torch.Tensor:
    """
    CMA Necessity Test (Hong et al. ?3.1):
    For each (layer, head), patch corrupted head output into clean run.
    Returns (n_layers, n_heads) tensor of Indirect Effect (calibrated logit drop).
    """
    clean_logits = model(clean_tokens)
    corrupted_logits = model(corrupted_tokens)
    clean_ld = float(compute_logit_diff(clean_logits, correct_id, incorrect_id, pos=pos).item())
    corrupted_ld = float(compute_logit_diff(corrupted_logits, correct_id, incorrect_id, pos=pos).item())

    effects = torch.zeros((n_layers, n_heads), dtype=torch.float32)

    for layer in range(n_layers):
        for head in range(n_heads):
            # Patch corrupted z into clean run
            act_name = f"blocks.{layer}.attn.hook_z" if hasattr(model, "blocks") else f"blocks.{layer}.hook_z"

            def hook_fn(z, hook, l=layer, h=head):
                del hook
                corrupted_z = corrupted_cache[act_name]
                z[:, :, h, :] = corrupted_z[:, :, h, :]
                return z

            patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=[(act_name, hook_fn)])
            patched_ld = float(compute_logit_diff(patched_logits, correct_id, incorrect_id, pos=pos).item())

            # IE = 1 - cLD(patched) = (clean_ld - patched_ld) / (clean_ld - corrupted_ld)
            ie = (clean_ld - patched_ld) / max(abs(clean_ld - corrupted_ld), 1e-8)
            effects[layer, head] = float(ie)

    return effects


def cma_sufficiency_test(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupted_cache: Dict[str, torch.Tensor],
    circuit_heads: Sequence[Tuple[int, int]],
    correct_id: int,
    incorrect_id: int,
    clean_ld: float,
    corrupted_ld: float,
    n_layers: int,
    n_heads: int,
    pos: int = -1,
) -> float:
    """
    CMA Sufficiency Test (Hong et al. ?3.2):
    Preserve only circuit heads; corrupt everything else.
    Returns calibrated LD retained by the circuit.
    """
    patched_logits = complement_patch_heads(
        model=model,
        clean_tokens=clean_tokens,
        corrupted_cache=corrupted_cache,
        circuit_heads=circuit_heads,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    patched_ld = float(compute_logit_diff(patched_logits, correct_id, incorrect_id, pos=pos).item())
    return compute_calibrated_ld(clean_ld, corrupted_ld, patched_ld)
