from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
try:
    import torch
except ImportError:
    torch = None
import numpy as np
from ..data_gen.circuit_pairs import CircuitSamplePair
from ..model_loading import to_tokens, resolve_true_false_token_ids
from ..patching.cma import cma_necessity_sweep
from ..progress import make_tqdm


def discover_circuit(
    model: Any,
    pairs: Sequence[CircuitSamplePair],
    threshold: float = 0.05,
    top_k: int = 20,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    """
    Sweep every (layer, head) over the clean/corrupted pairs and rank heads by Indirect Effect.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    true_id, false_id = resolve_true_false_token_ids(model)

    accumulated_effects = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    valid_pairs = 0

    for pair in make_tqdm(pairs, desc="circuit-discovery-sweep", leave=False):
        clean_tokens = to_tokens(model, pair.clean_prompt)
        corrupted_tokens = to_tokens(model, pair.counterfactual_prompt)

        # Match token lengths
        if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
            continue

        c_id = true_id if pair.clean_label else false_id
        i_id = false_id if pair.clean_label else true_id

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens)
            _, corrupted_cache = model.run_with_cache(corrupted_tokens)

            effects = cma_necessity_sweep(
                model=model,
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                clean_cache=clean_cache,
                corrupted_cache=corrupted_cache,
                correct_id=c_id,
                incorrect_id=i_id,
                n_layers=n_layers,
                n_heads=n_heads,
            )
            accumulated_effects += effects
            valid_pairs += 1

    if valid_pairs > 0:
        mean_effects = accumulated_effects / valid_pairs
    else:
        mean_effects = accumulated_effects

    ranked_heads: List[Dict[str, Any]] = []
    flat_indices = np.argsort(-mean_effects.cpu().numpy().flatten())

    for idx in flat_indices[:top_k]:
        l = int(idx // n_heads)
        h = int(idx % n_heads)
        score = float(mean_effects[l, h].item())
        if score >= threshold or len(ranked_heads) < 5:
            ranked_heads.append({
                "layer": l,
                "head": h,
                "indirect_effect": score,
                "head_label": f"L{l}H{h}",
            })

    return ranked_heads, mean_effects
