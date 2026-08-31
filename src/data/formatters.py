from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

PROMPT_ENDING_CHOICES = ('answer_suffix', 'terminal_is')

def normalize_prompt_ending(prompt_ending: str) -> str:
    """Normalize the prompt ending."""
    if prompt_ending not in PROMPT_ENDING_CHOICES:
        raise ValueError(f"Invalid prompt_ending: {prompt_ending}. Must be one of {PROMPT_ENDING_CHOICES}")
    return prompt_ending

def resolve_prompt_ending(row: Mapping, default: str = 'answer_suffix') -> str:
    """Resolve prompt ending from a row or default."""
    return normalize_prompt_ending(row.get('prompt_ending', default))

def format_facts(facts: Dict[str, bool], truth_style: str = 'full') -> str:
    """Format propositional facts."""
    parts = []
    for var, val in sorted(facts.items()):
        val_str = str(val) if truth_style == 'full' else str(val)[0]
        parts.append(f"{var} is {val_str}")
    return ", ".join(parts)

def format_derived_steps(steps: Iterable[Tuple[str, str]], truth_style: str = 'full') -> str:
    """Format derived steps."""
    parts = []
    for var, expr in steps:
        parts.append(f"{var} is {expr}")
    return ", ".join(parts)

def build_prop_prompt(
    facts: Dict[str, bool],
    expr_text: str,
    mode: str = 'nocot',
    template_style: str = 'compact',
    truth_style: str = 'full',
    derived_steps: Optional[Iterable[Tuple[str, str]]] = None,
    prompt_order: str = 'facts_first',
    prompt_ending: str = 'answer_suffix'
) -> str:
    """Build a propositional logic prompt."""
    prompt_ending = normalize_prompt_ending(prompt_ending)
    facts_str = format_facts(facts, truth_style)
    
    if derived_steps:
        derived_str = format_derived_steps(derived_steps, truth_style)
        if derived_str:
            facts_str += ", " + derived_str

    if template_style == 'compact':
        if prompt_order == 'facts_first':
            base_prompt = f"{facts_str}, {expr_text} is"
        else: # expr_first
            base_prompt = f"{expr_text} is? {facts_str}."
    else: # verbose
        base_prompt = f"Given the facts: {facts_str}. Evaluate the proposition: {expr_text}."

    if prompt_ending == 'answer_suffix':
        if base_prompt.endswith('is'):
            return f"{base_prompt}. Answer with one word only: True or False."
        else:
            return f"{base_prompt} Answer with one word only: True or False."
    else:
        return base_prompt

def build_depth_prompt(
    hop: int,
    facts: Dict[str, bool],
    query_expr_text: str,
    mode: str = 'nocot',
    derived_steps: Optional[Iterable[Tuple[str, str]]] = None,
    prompt_order: str = 'facts_first',
    prompt_ending: str = 'answer_suffix'
) -> str:
    """Build a depth prompt for propositional logic."""
    return build_prop_prompt(
        facts=facts,
        expr_text=query_expr_text,
        mode=mode,
        template_style='compact',
        derived_steps=derived_steps,
        prompt_order=prompt_order,
        prompt_ending=prompt_ending
    )

def format_frame(worlds: List[str], accessibility: Dict[str, List[str]]) -> str:
    """Format modal frame worlds and accessibility."""
    worlds_str = ", ".join(worlds)
    acc_parts = []
    for w in worlds:
        for succ in accessibility.get(w, []):
            acc_parts.append(f"{w} -> {succ}")
            
    if acc_parts:
        acc_str = ", ".join(acc_parts)
    else:
        acc_str = "None"
        
    return f"Worlds: {worlds_str}. Accessibility: {acc_str}."

def format_valuation(valuation: Dict[str, Dict[str, bool]], worlds: List[str]) -> str:
    """Format modal logic valuation across worlds."""
    parts = []
    for w in worlds:
        if w in valuation:
            world_facts = []
            for var in sorted(valuation[w].keys()):
                world_facts.append(f"{var} is {valuation[w][var]}")
            if world_facts:
                parts.append(f"At {w}: {', '.join(world_facts)}.")
    return " ".join(parts)

def build_modal_prompt(
    frame_worlds: List[str],
    frame_accessibility: Dict[str, List[str]],
    valuation: Dict[str, Dict[str, bool]],
    eval_world: str,
    expr_text: str,
    mode: str = 'nocot',
    prompt_style: str = 'symbolic',
    prompt_ending: str = 'answer_suffix'
) -> str:
    """Build a modal logic prompt."""
    prompt_ending = normalize_prompt_ending(prompt_ending)
    
    if prompt_style == 'symbolic':
        frame_str = format_frame(frame_worlds, frame_accessibility)
        val_str = format_valuation(valuation, frame_worlds)
        base_prompt = f"{frame_str} {val_str} Evaluate at {eval_world}: {expr_text} is"
        
    elif prompt_style == 'semi_natural':
        w_list = ", ".join(frame_worlds[:-1]) + ", and " + frame_worlds[-1] if len(frame_worlds) > 1 else frame_worlds[0]
        worlds_str = f"There are {len(frame_worlds)} possible worlds: {w_list}."
        
        acc_sentences = []
        for w in frame_worlds:
            succs = frame_accessibility.get(w, [])
            if succs:
                s_list = ", ".join(succs[:-1]) + " and " + succs[-1] if len(succs) > 1 else succs[0]
                acc_sentences.append(f"From {w}, worlds {s_list} are accessible.")
        acc_str = " ".join(acc_sentences)
        
        val_str = format_valuation(valuation, frame_worlds)
        
        # Replace operators with natural language
        nl_expr = expr_text.replace("□", "Necessarily ").replace("◇", "Possibly ").replace("->", "Implies")
        
        base_prompt = f"{worlds_str} {acc_str} {val_str} Evaluate the modal proposition at {eval_world}: {nl_expr}."
        
    elif prompt_style == 'verbose':
        worlds_set = "{" + ", ".join(frame_worlds) + "}"
        acc_pairs = []
        for w in frame_worlds:
            for succ in frame_accessibility.get(w, []):
                acc_pairs.append(f"({w},{succ})")
        acc_set = "{" + ", ".join(acc_pairs) + "}"
        
        val_parts = []
        for w in frame_worlds:
            if w in valuation:
                v_facts = ", ".join(f"{var}: {val}" for var, val in sorted(valuation[w].items()))
                val_parts.append(f"V({w}) = {{{v_facts}}}")
        val_str = ", ".join(val_parts)
        
        base_prompt = f"Given the Kripke model: Worlds: {worlds_set}. Accessibility relation R: {acc_set}. Valuation: {val_str}. Evaluate the following modal proposition at world {eval_world}: {expr_text}."
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")

    if prompt_ending == 'answer_suffix':
        if base_prompt.endswith('is'):
            return f"{base_prompt}. Answer with one word only: True or False."
        else:
            return f"{base_prompt} Answer with one word only: True or False."
    else:
        return base_prompt

def build_modal_depth_prompt(
    hop: int,
    frame_worlds: List[str],
    frame_accessibility: Dict[str, List[str]],
    valuation: Dict[str, Dict[str, bool]],
    eval_world: str,
    query_expr_text: str,
    mode: str = 'nocot',
    prompt_style: str = 'symbolic',
    derived_steps: Optional[Iterable[Tuple[str, str]]] = None,
    prompt_ending: str = 'answer_suffix'
) -> str:
    """Build a modal depth prompt, potentially with derived steps."""
    prompt_ending = normalize_prompt_ending(prompt_ending)
    
    if hop == 1 or not derived_steps:
        return build_modal_prompt(
            frame_worlds=frame_worlds,
            frame_accessibility=frame_accessibility,
            valuation=valuation,
            eval_world=eval_world,
            expr_text=query_expr_text,
            mode=mode,
            prompt_style=prompt_style,
            prompt_ending=prompt_ending
        )
        
    # For multi-hop, construct it manually based on symbolic format as requested
    frame_str = format_frame(frame_worlds, frame_accessibility)
    val_str = format_valuation(valuation, frame_worlds)
    derived_str = format_derived_steps(derived_steps, 'full')
    
    base_prompt = f"{frame_str} {val_str} {derived_str}. Evaluate at {eval_world}: {query_expr_text} is"
    
    if prompt_ending == 'answer_suffix':
        return f"{base_prompt}. Answer with one word only: True or False."
    return base_prompt
