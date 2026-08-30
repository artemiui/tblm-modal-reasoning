from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
try:
    import torch
except ImportError:
    torch = None

Span = Tuple[int, int]  # [start, end) token indices


def _to_ids(model: Any, text: str) -> List[int]:
    if hasattr(model, "to_tokens"):
        toks = model.to_tokens(text, prepend_bos=False)
        return toks[0].tolist() if len(toks.shape) > 1 else toks.tolist()
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer.encode(text, add_special_tokens=False)
    return [hash(text) % 1000]


def _find_all_subseq(sntc: List[int], subseq: List[int]) -> List[int]:
    out = []
    if len(subseq) == 0 or len(subseq) > len(sntc):
        return out
    n = len(subseq)
    for i in range(len(sntc) - n + 1):
        if sntc[i:i + n] == subseq:
            out.append(i)
    return out


def _find_marker(
    tokens_row: List[int],
    model: Any,
    marker: str,
    start_at: int = 0,
    end_at: Optional[int] = None,
    want_last: bool = False,
) -> Optional[Span]:
    """Find marker token span (e.g. ACCESS_START, ACCESS_END, Rules:, Facts:, Question:, Answer:)."""
    if end_at is None:
        end_at = len(tokens_row)
    hay = tokens_row[start_at:end_at]

    variants = [_to_ids(model, s) for s in [marker, " " + marker, "\n" + marker, marker + ":", " " + marker + ":"]]

    best: Optional[Span] = None
    for v in variants:
        if not v:
            continue
        starts = _find_all_subseq(hay, v)
        if not starts:
            continue
        idx = (starts[-1] if want_last else starts[0]) + start_at
        cand = (idx, idx + len(v))
        if best is None:
            best = cand
        else:
            if want_last and cand[0] > best[0]:
                best = cand
            if (not want_last) and cand[0] < best[0]:
                best = cand
    return best


def _find_in_range(
    tokens_row: List[int],
    model: Any,
    clause_text: str,
    start_: int,
    end_: int,
) -> Optional[Span]:
    region = tokens_row[start_:end_]
    leads = ["", " ", "\n"]
    trails = ["", ".", ";", " .", " ;"]
    variants = []
    for L in leads:
        for T in trails:
            ids = _to_ids(model, L + clause_text + T)
            if ids:
                variants.append(ids)

    for v in variants:
        starts = _find_all_subseq(region, v)
        if starts:
            s = starts[0] + start_
            return (s, s + len(v))
    return None


def clause_token_spans_for_batch(
    tokens_batch: Any,
    model: Any,
    problem_info_batch: Sequence[Dict[str, Any]],
) -> Dict[str, List[Optional[Span]]]:
    spans: Dict[str, List[Optional[Span]]] = {
        "queried_rule": [],
        "correct_fact": [],
        "modal_proposition_span": [],
        "modal_operator_token": [],
        "accessibility_clause": [],
        "accessible_facts": [],
        "inaccessible_facts": [],
    }

    n_prompts = len(problem_info_batch)
    for b in range(n_prompts):
        tokens_row = tokens_batch[b].tolist() if hasattr(tokens_batch[b], "tolist") else list(tokens_batch[b])
        info = problem_info_batch[b]

        acc_span = _find_marker(tokens_row, model, "ACCESS_START", want_last=True)
        rules_span = _find_marker(tokens_row, model, "Rules", want_last=True)
        facts_span = _find_marker(tokens_row, model, "Facts", want_last=True)
        q_span = _find_marker(tokens_row, model, "Question", want_last=True)
        ans_span = _find_marker(tokens_row, model, "Answer", want_last=True)

        spans["accessibility_clause"].append(acc_span)

        op_text = "necessarily" if info.get("operator") == "BOX" else "possibly"
        op_span = _find_marker(tokens_row, model, op_text, want_last=True)
        spans["modal_operator_token"].append(op_span)

        rule_text = str(info.get("queried_rule", ""))
        r_start = rules_span[1] if rules_span else 0
        r_end = q_span[0] if q_span else len(tokens_row)
        q_rule_span = _find_in_range(tokens_row, model, rule_text, r_start, r_end) if rule_text else None
        spans["queried_rule"].append(q_rule_span)
        spans["modal_proposition_span"].append(q_rule_span if q_rule_span else rules_span)

        fact_text = str(info.get("correct_fact", ""))
        f_start = facts_span[1] if facts_span else 0
        f_end = rules_span[0] if rules_span else (q_span[0] if q_span else len(tokens_row))
        c_fact_span = _find_in_range(tokens_row, model, fact_text, f_start, f_end) if fact_text else None
        spans["correct_fact"].append(c_fact_span)

        acc_fact = _find_in_range(tokens_row, model, "w0", f_start, f_end)
        inacc_fact = _find_in_range(tokens_row, model, "w2", f_start, f_end)
        spans["accessible_facts"].append(acc_fact)
        spans["inaccessible_facts"].append(inacc_fact)

    return spans


def compute_modal_attention_statistics(
    model: Any,
    tokens: Any,
    heads: Sequence[Tuple[int, int]],
    target_spans: Sequence[Optional[Span]],
    source_pos: int = -1,
) -> List[float]:
    scores = []
    with torch.no_grad() if torch is not None else None:
        _, cache = model.run_with_cache(tokens)
        for layer, head in heads:
            attn_name = f"blocks.{layer}.attn.hook_pattern" if hasattr(model, "blocks") else f"blocks.{layer}.hook_pattern"
            pattern = cache[attn_name]
            head_masses = []
            for b in range(pattern.shape[0]):
                span = target_spans[b] if b < len(target_spans) else None
                if span is not None:
                    s, e = span
                    mass = pattern[b, head, source_pos, s:e].sum().item()
                    head_masses.append(float(mass))
                else:
                    head_masses.append(0.0)
            scores.append(float(sum(head_masses) / max(1, len(head_masses))))
    return scores
