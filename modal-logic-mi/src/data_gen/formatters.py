from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from .modal_grammar import KripkeFrame, to_symbolic, to_natural, Expr


def format_facts_block(valuation: Dict[str, Dict[str, bool]]) -> str:
    world_parts = []
    for w in sorted(valuation.keys()):
        var_parts = [f"{var} is {val}" for var, val in sorted(valuation[w].items())]
        world_parts.append(f"In {w}: {', '.join(var_parts)}")
    return "; ".join(world_parts)


def format_accessibility_block(frame: KripkeFrame, base_world: str = "w0") -> str:
    acc = sorted(frame.accessible_from(base_world))
    return f"Accessibility from {base_world}: [{', '.join(acc)}]"


def build_modal_mi_prompt(
    *,
    valuation: Dict[str, Dict[str, bool]],
    frame: KripkeFrame,
    expr_text: str,
    base_world: str = "w0",
    prompt_order: str = "facts_first",
    mode: str = "nocot",
) -> str:
    facts_text = format_facts_block(valuation)
    acc_text = format_accessibility_block(frame, base_world=base_world)
    suffix = " Answer with one word only: True or False." if mode == "nocot" else " Reason step by step, then end with one final word: True or False."

    if prompt_order == "facts_first":
        # Facts -> Accessibility -> Expression -> Query
        return f"Given facts: {facts_text}. {acc_text}. Evaluate proposition: {expr_text} is.{suffix}"
    elif prompt_order == "expr_first":
        # Expression -> Facts -> Accessibility -> Query
        return f"Evaluate proposition: {expr_text} is? Given facts: {facts_text}. {acc_text}.{suffix}"
    raise ValueError(f"Unknown prompt_order: {prompt_order}")


def build_4region_char_spans(
    prompt: str,
    valuation: Dict[str, Dict[str, bool]],
    frame: KripkeFrame,
    expr_text: str,
    base_world: str = "w0",
    prompt_order: str = "facts_first",
) -> Dict[str, List[Tuple[int, int]]]:
    facts_text = format_facts_block(valuation)
    acc_text = format_accessibility_block(frame, base_world=base_world)

    spans: Dict[str, List[Tuple[int, int]]] = {
        "facts_region": [],
        "accessibility_region": [],
        "expression_region": [],
        "query_region": [],
    }

    f_idx = prompt.find(facts_text)
    if f_idx >= 0:
        spans["facts_region"].append((f_idx, f_idx + len(facts_text)))

    a_idx = prompt.find(acc_text)
    if a_idx >= 0:
        spans["accessibility_region"].append((a_idx, a_idx + len(acc_text)))

    e_idx = prompt.find(expr_text)
    if e_idx >= 0:
        spans["expression_region"].append((e_idx, e_idx + len(expr_text)))

    # Query region covers prompt tail / answer instructions
    query_markers = [" Answer with one word only:", " Reason step by step", " is.", " is?"]
    for marker in query_markers:
        q_idx = prompt.find(marker)
        if q_idx >= 0:
            spans["query_region"].append((q_idx, len(prompt)))
            break
    if not spans["query_region"]:
        spans["query_region"].append((max(0, len(prompt) - 30), len(prompt)))

    return spans
