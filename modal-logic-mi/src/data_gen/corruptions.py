from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Tuple
from .modal_grammar import Expr, KripkeFrame, KripkeModel, eval_modal_expr


def find_label_flipping_fact_corruption(
    model: KripkeModel,
    expr: Expr,
    base_world: str = "w0",
    rng: Optional[random.Random] = None,
) -> Tuple[KripkeModel, Dict[str, object]]:
    """Flip exactly one fact variable in some world to flip the evaluation label."""
    if rng is None:
        rng = random.Random(42)

    clean_label = eval_modal_expr(expr, model, base_world)
    candidates: List[Tuple[str, str, bool, KripkeModel]] = []

    for world, facts in model.valuation.items():
        for var_name, val in facts.items():
            new_model = model.clone()
            new_model.valuation[world][var_name] = not val
            new_label = eval_modal_expr(expr, new_model, base_world)
            if new_label != clean_label:
                candidates.append((world, var_name, not val, new_model))

    if not candidates:
        raise ValueError("Could not find single-fact corruption that flips truth value.")

    chosen_world, chosen_var, new_val, corrupted_model = rng.choice(candidates)
    meta = {
        "corruption_type": "fact_corruption",
        "world": chosen_world,
        "variable": chosen_var,
        "old_value": not new_val,
        "new_value": new_val,
        "clean_label": clean_label,
        "corrupted_label": not clean_label,
    }
    return corrupted_model, meta


def find_label_flipping_accessibility_corruption(
    model: KripkeModel,
    expr: Expr,
    base_world: str = "w0",
    rng: Optional[random.Random] = None,
) -> Tuple[KripkeModel, Dict[str, object]]:
    """Add or remove an accessibility edge from base_world to flip the evaluation label."""
    if rng is None:
        rng = random.Random(42)

    clean_label = eval_modal_expr(expr, model, base_world)
    current_acc = set(model.frame.accessible_from(base_world))
    all_worlds = set(model.frame.worlds)

    candidates: List[Tuple[str, str, KripkeModel]] = []

    # Try removing an accessible world
    for w in sorted(current_acc):
        new_model = model.clone()
        new_acc = [x for x in current_acc if x != w]
        new_model.frame.accessibility[base_world] = new_acc
        new_label = eval_modal_expr(expr, new_model, base_world)
        if new_label != clean_label:
            candidates.append(("remove_edge", w, new_model))

    # Try adding an inaccessible world
    for w in sorted(all_worlds - current_acc):
        new_model = model.clone()
        new_acc = list(current_acc) + [w]
        new_model.frame.accessibility[base_world] = new_acc
        new_label = eval_modal_expr(expr, new_model, base_world)
        if new_label != clean_label:
            candidates.append(("add_edge", w, new_model))

    if not candidates:
        raise ValueError("Could not find single-accessibility corruption that flips truth value.")

    action, target_world, corrupted_model = rng.choice(candidates)
    meta = {
        "corruption_type": "accessibility_corruption",
        "action": action,
        "target_world": target_world,
        "clean_label": clean_label,
        "corrupted_label": not clean_label,
    }
    return corrupted_model, meta
