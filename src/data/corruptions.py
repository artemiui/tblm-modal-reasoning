from __future__ import annotations

import copy
import itertools
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .modal_ast import Box, Diamond, Expr, Not, eval_expr, eval_modal

if TYPE_CHECKING:
    from .kripke import KripkeFrame, KripkeModel


def find_label_flipping_fact_corruption(
    expr: Expr,
    facts: Dict[str, bool],
) -> Tuple[Dict[str, bool], bool]:
    """Exhaustively tries flipping subsets of facts to find minimal Hamming distance corruption that flips the label."""
    clean_label = eval_expr(expr, facts)
    vars_to_flip = list(facts.keys())
    
    for num_flips in range(1, len(vars_to_flip) + 1):
        for combo in itertools.combinations(vars_to_flip, num_flips):
            corrupted_facts = dict(facts)
            for var in combo:
                corrupted_facts[var] = not corrupted_facts[var]
            
            corrupted_label = eval_expr(expr, corrupted_facts)
            if corrupted_label != clean_label:
                return corrupted_facts, corrupted_label
                
    raise ValueError("Could not find a fact corruption that flips the label.")


def _swap_operators(expr: Expr) -> Expr:
    """Walk the expression tree and swap all Box nodes to Diamond and vice versa."""
    if expr.op == 'box':
        return Diamond(_swap_operators(expr.children[0]))
    elif expr.op == 'diamond':
        return Box(_swap_operators(expr.children[0]))
    elif expr.op in ('var', 'const'):
        return expr
    elif len(expr.children) == 1:
        return Expr(op=expr.op, children=(_swap_operators(expr.children[0]),))
    else:
        return Expr(
            op=expr.op,
            children=(_swap_operators(expr.children[0]), _swap_operators(expr.children[1]))
        )


def find_modal_corruption(
    expr: Expr,
    model: KripkeModel,
    eval_world: str,
    rng: random.Random
) -> Tuple[KripkeModel, Expr, bool, str]:
    """Tries strategies in order of preference: valuation_flip, accessibility_flip, operator_swap.
    Returns (corrupted_model, corrupted_expr, corrupted_label, strategy)"""
    from .kripke import KripkeFrame, KripkeModel

    clean_label = eval_modal(expr, model, eval_world)
    worlds = model.frame.worlds
    
    # 1. Valuation Flip
    all_var_locations = []
    for w in worlds:
        for var in model.valuation[w]:
            all_var_locations.append((w, var))
            
    rng.shuffle(all_var_locations)
    for num_flips in range(1, min(3, len(all_var_locations)) + 1):
        for combo in itertools.combinations(all_var_locations, num_flips):
            new_valuation = {w: dict(vals) for w, vals in model.valuation.items()}
            for w, var in combo:
                new_valuation[w][var] = not new_valuation[w][var]
                
            corrupted_model = KripkeModel(frame=model.frame, valuation=new_valuation, eval_world=model.eval_world)
            if eval_modal(expr, corrupted_model, eval_world) != clean_label:
                return corrupted_model, expr, not clean_label, "valuation"

    # 2. Accessibility Flip
    possible_edges = [(w1, w2) for w1 in worlds for w2 in worlds]
    rng.shuffle(possible_edges)
    
    for w1, w2 in possible_edges:
        new_relation = {w: list(succ) for w, succ in model.frame.relation.items()}
        if w1 not in new_relation:
            new_relation[w1] = []
            
        if w2 in new_relation[w1]:
            new_relation[w1].remove(w2)
        else:
            new_relation[w1].append(w2)
            
        new_relation_tuple = {w: tuple(succ) for w, succ in new_relation.items()}
        new_frame = KripkeFrame(worlds=worlds, relation=new_relation_tuple)
        corrupted_model = KripkeModel(frame=new_frame, valuation=model.valuation, eval_world=model.eval_world)
        
        if eval_modal(expr, corrupted_model, eval_world) != clean_label:
            return corrupted_model, expr, not clean_label, "relation"

    # 3. Operator Swap
    corrupted_expr = _swap_operators(expr)
    if eval_modal(corrupted_expr, model, eval_world) != clean_label:
        return model, corrupted_expr, not clean_label, "operator"

    # 4. Eval World Swap
    other_worlds = [w for w in worlds if w != eval_world]
    rng.shuffle(other_worlds)
    for new_world in other_worlds:
        corrupted_model = KripkeModel(frame=model.frame, valuation=model.valuation, eval_world=new_world)
        if eval_modal(expr, corrupted_model, new_world) != clean_label:
            return corrupted_model, expr, not clean_label, "eval_world_swap"

    # 5. Expression negation (fallback for tautologies like duality)
    corrupted_expr_neg = Not(expr)
    if eval_modal(corrupted_expr_neg, model, eval_world) != clean_label:
        return model, corrupted_expr_neg, not clean_label, "negation"

    raise ValueError("Could not find any corruption that flips the label.")
