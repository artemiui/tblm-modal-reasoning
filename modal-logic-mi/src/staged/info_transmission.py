from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple
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


TOKEN_CATEGORIES = [
    "facts_value",
    "accessibility_boundary",
    "variable_in_facts",
    "variable_in_expr",
    "operator",
    "expr_last",
    "derived_assignment",
    "query_token",
    "others",
]


def classify_modal_tokens(str_tokens: Sequence[str]) -> List[str]:
    """Classify token sequence into 9 modal interpretability categories."""
    categories: List[str] = []

    for tok in str_tokens:
        clean = tok.strip().lower()
        if "access" in clean or "w0" in clean or "w1" in clean or "w2" in clean:
            categories.append("accessibility_boundary")
        elif clean in {"true", "false"}:
            categories.append("facts_value")
        elif clean in {"box", "diamond", "necessarily", "possibly", "probably", "certainly", "unlikely", "and", "or", "xor", "iff", "implies", "not"}:
            categories.append("operator")
        elif clean in {"p", "q", "r", "s", "a", "b", "c", "d"}:
            categories.append("variable_in_expr")
        elif "answer" in clean or "reason" in clean or "is" == clean or "?" in clean:
            categories.append("query_token")
        else:
            categories.append("others")

    # Set last non-punct expr token as expr_last
    for i in range(len(categories) - 1, -1, -1):
        if categories[i] in {"variable_in_expr", "operator"}:
            categories[i] = "expr_last"
            break

    return categories


def run_information_transmission(
    model: Any,
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Token-wise Residual Stream Patching (Chen et al. ?4.2):
    Track causal convergence at fact values, accessibility boundary, expr last, query tokens.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_layers = model.cfg.n_layers
    true_id, false_id = resolve_true_false_token_ids(model)
    bands = {"early": list(range(0, n_layers // 3)), "middle": list(range(n_layers // 3, 2 * (n_layers // 3))), "late": list(range(2 * (n_layers // 3), n_layers))}

    cat_stage_dpd: Dict[str, Dict[str, List[float]]] = {cat: {"early": [], "middle": [], "late": []} for cat in TOKEN_CATEGORIES}

    for sample in make_tqdm(samples, desc="info-transmission-sweep", leave=False):
        clean_toks = to_tokens(model, str(sample["clean_prompt_symbolic"]))
        cf_toks = to_tokens(model, str(sample["corrupted_prompt_symbolic"]))
        if clean_toks.shape[1] != cf_toks.shape[1]:
            continue

        str_tokens = model.to_str_tokens(clean_toks[0])
        categories = classify_modal_tokens(str_tokens)
        c_id = true_id if sample["label"] else false_id
        i_id = false_id if sample["label"] else true_id
        seq_len = int(clean_toks.shape[1])

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_toks)
            corrupted_logits = model(cf_toks)
            base_pd = float(compute_prob_diff(corrupted_logits, c_id, i_id, pos=-1).item())

            for pos in range(seq_len):
                cat = categories[pos] if pos < len(categories) else "others"
                for layer in range(n_layers):
                    patched_logits = patch_residual_stream(model, cf_toks, clean_cache, layer=layer, pos=pos)
                    patched_pd = float(compute_prob_diff(patched_logits, c_id, i_id, pos=-1).item())
                    dpd = abs(patched_pd - base_pd)

                    if layer in bands["early"]:
                        cat_stage_dpd[cat]["early"].append(dpd)
                    elif layer in bands["middle"]:
                        cat_stage_dpd[cat]["middle"].append(dpd)
                    else:
                        cat_stage_dpd[cat]["late"].append(dpd)

    summary_stats: Dict[str, Any] = {}
    for cat in TOKEN_CATEGORIES:
        summary_stats[cat] = {}
        for stage in ["early", "middle", "late"]:
            vals = cat_stage_dpd[cat][stage]
            summary_stats[cat][stage] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "sem": float(np.std(vals) / np.sqrt(max(1, len(vals)))) if vals else 0.0,
                "count": len(vals),
            }

    (output_dir / "info_transmission_stats.json").write_text(json.dumps(summary_stats, indent=2), encoding="utf-8")
    return summary_stats
