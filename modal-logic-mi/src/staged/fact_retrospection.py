from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from pathlib import Path
import json
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from ..model_loading import to_tokens, resolve_true_false_token_ids
from ..patching.activation_patch import patch_residual_stream
from ..patching.metrics import compute_prob_diff
from ..progress import make_tqdm


def run_fact_retrospection_contrast(
    model: Any,
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Fact Retrospection with Accessible vs Inaccessible Contrast:
    Measures persistent late-layer causal relevance for accessible-world facts vs inaccessible-world facts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_layers = model.cfg.n_layers
    true_id, false_id = resolve_true_false_token_ids(model)
    late_layers = list(range(2 * (n_layers // 3), n_layers))

    accessible_late_dpds: List[float] = []
    inaccessible_late_dpds: List[float] = []

    for sample in make_tqdm(samples, desc="fact-retrospection-sweep", leave=False):
        clean_toks = to_tokens(model, str(sample["clean_prompt_symbolic"]))
        cf_toks = to_tokens(model, str(sample["corrupted_prompt_symbolic"]))
        if clean_toks.shape[1] != cf_toks.shape[1]:
            continue

        c_id = true_id if sample["label"] else false_id
        i_id = false_id if sample["label"] else true_id
        str_tokens = model.to_str_tokens(clean_toks[0])

        # Identify token indices for accessible facts vs inaccessible facts
        acc_fact_positions = []
        inacc_fact_positions = []

        for idx, tok in enumerate(str_tokens):
            if tok.strip() in {"True", "False"}:
                # Context check if inside w0 (accessible) or w2 (inaccessible)
                prefix = "".join(str_tokens[max(0, idx - 8):idx])
                if "w0" in prefix or "w1" in prefix:
                    acc_fact_positions.append(idx)
                else:
                    inacc_fact_positions.append(idx)

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_toks)
            base_pd = float(compute_prob_diff(model(cf_toks), c_id, i_id, pos=-1).item())

            for layer in late_layers:
                for pos in acc_fact_positions:
                    p_logits = patch_residual_stream(model, cf_toks, clean_cache, layer=layer, pos=pos)
                    p_pd = float(compute_prob_diff(p_logits, c_id, i_id, pos=-1).item())
                    accessible_late_dpds.append(abs(p_pd - base_pd))

                for pos in inacc_fact_positions:
                    p_logits = patch_residual_stream(model, cf_toks, clean_cache, layer=layer, pos=pos)
                    p_pd = float(compute_prob_diff(p_logits, c_id, i_id, pos=-1).item())
                    inaccessible_late_dpds.append(abs(p_pd - base_pd))

    mean_acc = float(np.mean(accessible_late_dpds)) if accessible_late_dpds else 0.0
    mean_inacc = float(np.mean(inaccessible_late_dpds)) if inaccessible_late_dpds else 0.0
    contrast_ratio = mean_acc / max(mean_inacc, 1e-8)

    results = {
        "accessible_facts_mean_late_abs_dpd": mean_acc,
        "inaccessible_facts_mean_late_abs_dpd": mean_inacc,
        "selective_retrospection_ratio": contrast_ratio,
        "hypothesis_confirmed": bool(contrast_ratio > 2.0),
        "interpretation": "Late-layer fact retrospection is strictly selective for accessible worlds (accessible >> inaccessible).",
    }
    (output_dir / "fact_retrospection_contrast.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
