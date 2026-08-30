from __future__ import annotations

import math
from typing import Tuple, Union

try:
    import torch
except ImportError:
    torch = None


def compute_2way_probs(logits: Any, true_id: int, false_id: int, pos: int = -1) -> Tuple[Any, Any]:
    """Compute binary softmax probabilities over True and False tokens."""
    logit_t = logits[:, pos, true_id]
    logit_f = logits[:, pos, false_id]
    pair = torch.stack([logit_t, logit_f], dim=-1)
    probs = torch.softmax(pair, dim=-1)
    return probs[..., 0], probs[..., 1]


def compute_prob_diff(logits: Any, correct_id: int, incorrect_id: int, pos: int = -1) -> Any:
    """p(correct) - p(incorrect) over the 2-token binary subspace."""
    p_c, p_i = compute_2way_probs(logits, correct_id, incorrect_id, pos=pos)
    return p_c - p_i


def compute_logit_diff(logits: Any, correct_id: int, incorrect_id: int, pos: int = -1) -> Any:
    """Logit difference: logit(correct) - logit(incorrect)."""
    return logits[:, pos, correct_id] - logits[:, pos, incorrect_id]


def compute_calibrated_ld(clean_ld: float, corrupted_ld: float, patched_ld: float, eps: float = 1e-8) -> float:
    """
    Hong et al. Eq. 1: Calibrated Logit Difference (Normalized Indirect Effect).
    cLD = (patched_ld - corrupted_ld) / (clean_ld - corrupted_ld)
    """
    denom = clean_ld - corrupted_ld
    if abs(denom) < eps:
        return 0.0
    return (patched_ld - corrupted_ld) / denom


def compute_dpd_shift(
    base_logits: Any,
    patched_logits: Any,
    true_id: int,
    false_id: int,
    label: Any = None,
    pos: int = -1,
) -> Any:
    """dPD = patched_PD - base_PD."""
    if label is None:
        p_base = compute_prob_diff(base_logits, true_id, false_id, pos=pos)
        p_patched = compute_prob_diff(patched_logits, true_id, false_id, pos=pos)
    else:
        c_id = true_id if bool(label) else false_id
        i_id = false_id if bool(label) else true_id
        p_base = compute_prob_diff(base_logits, c_id, i_id, pos=pos)
        p_patched = compute_prob_diff(patched_logits, c_id, i_id, pos=pos)
    return p_patched - p_base


def compute_r_ld(dpd_shift: float, base_ld: float, eps: float = 1e-8) -> float:
    """Normalized relative logit/prob difference (Chen et al. App. B.1)."""
    return dpd_shift / max(abs(base_ld), eps)
