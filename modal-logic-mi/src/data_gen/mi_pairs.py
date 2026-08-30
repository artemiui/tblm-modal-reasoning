from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple
from .modal_grammar import (
    And, Box, Const, Diamond, Expr, Iff, Implies, KripkeFrame, KripkeModel,
    Not, Or, Var, Xor, eval_modal_expr, to_symbolic
)
from .formatters import build_modal_mi_prompt
from .corruptions import (
    find_label_flipping_fact_corruption,
    find_label_flipping_accessibility_corruption,
)


@dataclass(frozen=True)
class ModalRuleCategory:
    name: str
    formal_description: str
    build_fn: Callable[[random.Random], Tuple[Expr, KripkeModel, str]]


def _build_necessitation_implication(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = Box(Implies(Var("P"), Var("Q")))
    return expr, KripkeModel(frame, val), "w0"


def _build_possibility_implication(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = Diamond(Implies(Var("P"), Var("Q")))
    return expr, KripkeModel(frame, val), "w0"


def _build_duality(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1))},
    }
    expr = Box(Var("P"))
    return expr, KripkeModel(frame, val), "w0"


def _build_t_axiom(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    # T-axiom variation: box(P) and Q
    expr = And(Box(Var("P")), Var("Q"))
    return expr, KripkeModel(frame, val), "w0"


def _build_k_axiom(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = And(Box(Implies(Var("P"), Var("Q"))), Box(Var("P")))
    return expr, KripkeModel(frame, val), "w0"


def _build_b_axiom(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    # Axiom B: P -> box(diamond(P)) on symmetric frames
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"], "w1": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = Implies(Var("P"), Box(Diamond(Var("Q"))))
    return expr, KripkeModel(frame, val), "w0"


def _build_d_axiom(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    # Axiom D: box(P) -> diamond(P) on serial frames
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"], "w1": ["w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = Implies(Box(Var("P")), Diamond(Var("Q")))
    return expr, KripkeModel(frame, val), "w0"


def _build_four_axiom(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    # Axiom 4: box(P) -> box(box(P)) on transitive frames
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"], "w1": ["w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = Implies(Box(Var("P")), Box(Box(Var("Q"))))
    return expr, KripkeModel(frame, val), "w0"


def _build_five_axiom(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    # Axiom 5: diamond(P) -> box(diamond(P)) on euclidean frames
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"], "w1": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = Implies(Diamond(Var("P")), Box(Diamond(Var("Q"))))
    return expr, KripkeModel(frame, val), "w0"


def _build_modal_commutative_associative(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = Box(And(Var("P"), Var("Q")))
    return expr, KripkeModel(frame, val), "w0"


def _build_connective_disjunction(rng: random.Random) -> Tuple[Expr, KripkeModel, str]:
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    val = {
        "w0": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
        "w1": {"P": bool(rng.getrandbits(1)), "Q": bool(rng.getrandbits(1))},
    }
    expr = Box(Or(Var("P"), Var("Q")))
    return expr, KripkeModel(frame, val), "w0"


MODAL_RULE_CATEGORIES: List[ModalRuleCategory] = [
    ModalRuleCategory("necessitation_implication", "\u25a1(P -> Q)", _build_necessitation_implication),
    ModalRuleCategory("possibility_implication", "\u25c7(P -> Q)", _build_possibility_implication),
    ModalRuleCategory("duality", "\u25a1P <-> ~\u25c7~P", _build_duality),
    ModalRuleCategory("t_axiom", "\u25a1P -> P", _build_t_axiom),
    ModalRuleCategory("k_axiom", "\u25a1(P -> Q) -> (\u25a1P -> \u25a1Q)", _build_k_axiom),
    ModalRuleCategory("b_axiom", "P -> \u25a1\u25c7P", _build_b_axiom),
    ModalRuleCategory("d_axiom", "\u25a1P -> \u25c7P", _build_d_axiom),
    ModalRuleCategory("four_axiom", "\u25a1P -> \u25a1\u25a1P", _build_four_axiom),
    ModalRuleCategory("five_axiom", "\u25c7P -> \u25a1\u25c7P", _build_five_axiom),
    ModalRuleCategory("modal_commutative_associative", "\u25a1(P and Q) <-> \u25a1P and \u25a1Q", _build_modal_commutative_associative),
    ModalRuleCategory("connective_disjunction", "\u25a1(P or Q)", _build_connective_disjunction),
]


def generate_single_modal_mi_sample(
    sample_id: str,
    category_name: str,
    rng: random.Random,
    hop: str = "one_hop",
    prompt_order: str = "facts_first",
    corruption_mode: str = "random",
    max_attempts: int = 50,
) -> Dict[str, object]:
    # Alias normalization for 4_axiom / 5_axiom
    normalized_category = {
        "4_axiom": "four_axiom",
        "5_axiom": "five_axiom",
        "axiom_4": "four_axiom",
        "axiom_5": "five_axiom",
        "axiom_b": "b_axiom",
        "axiom_d": "d_axiom",
        "axiom_t": "t_axiom",
        "axiom_k": "k_axiom",
    }.get(category_name, category_name)
    cat = next((c for c in MODAL_RULE_CATEGORIES if c.name == normalized_category), None)
    if cat is None:
        raise KeyError(f"Unknown rule category {category_name}")

    for attempt in range(max_attempts):
        expr, model, base_world = cat.build_fn(rng)

        if hop == "two_hop":
            wrappers = [
                lambda e: Box(e),
                lambda e: Diamond(e),
                lambda e: Not(Not(e)),
                lambda e: And(e, Const(True)),
            ]
            expr = rng.choice(wrappers)(expr)

        clean_label = eval_modal_expr(expr, model, base_world)

        chosen_mode = corruption_mode
        if chosen_mode == "random":
            chosen_mode = "accessibility" if rng.random() < 0.5 else "fact"

        corrupted_model = None
        corruption_meta = None

        if chosen_mode == "accessibility":
            try:
                corrupted_model, corruption_meta = find_label_flipping_accessibility_corruption(model, expr, base_world, rng)
            except ValueError:
                try:
                    corrupted_model, corruption_meta = find_label_flipping_fact_corruption(model, expr, base_world, rng)
                except ValueError:
                    continue
        else:
            try:
                corrupted_model, corruption_meta = find_label_flipping_fact_corruption(model, expr, base_world, rng)
            except ValueError:
                try:
                    corrupted_model, corruption_meta = find_label_flipping_accessibility_corruption(model, expr, base_world, rng)
                except ValueError:
                    continue

        if corrupted_model is not None:
            corrupted_label = eval_modal_expr(expr, corrupted_model, base_world)
            if corrupted_label != clean_label:
                expr_text = to_symbolic(expr)
                clean_prompt = build_modal_mi_prompt(
                    valuation=model.valuation,
                    frame=model.frame,
                    expr_text=expr_text,
                    base_world=base_world,
                    prompt_order=prompt_order,
                )
                corrupted_prompt = build_modal_mi_prompt(
                    valuation=corrupted_model.valuation,
                    frame=corrupted_model.frame,
                    expr_text=expr_text,
                    base_world=base_world,
                    prompt_order=prompt_order,
                )

                return {
                    "id": sample_id,
                    "rule": category_name,
                    "category": category_name,
                    "hop": hop,
                    "prompt_order": prompt_order,
                    "clean_valuation": model.valuation,
                    "corrupted_valuation": corrupted_model.valuation,
                    "clean_accessibility": model.frame.accessibility,
                    "corrupted_accessibility": corrupted_model.frame.accessibility,
                    "expr_symbolic": expr_text,
                    "clean_label": clean_label,
                    "corrupted_label": corrupted_label,
                    "label": clean_label,
                    "label_corrupted": corrupted_label,
                    "clean_prompt_symbolic": clean_prompt,
                    "corrupted_prompt_symbolic": corrupted_prompt,
                    "corruption_meta": corruption_meta,
                }

    raise RuntimeError(f"Failed to generate valid flipping sample for category {category_name} after {max_attempts} attempts.")


def generate_modal_mi_dataset(
    n_samples: int = 500,
    seed: int = 42,
    prompt_order: str = "facts_first",
    one_hop_ratio: float = 0.5,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    categories = [c.name for c in MODAL_RULE_CATEGORIES]
    rows: List[Dict[str, object]] = []

    n_one_hop = int(n_samples * one_hop_ratio)
    n_two_hop = n_samples - n_one_hop

    for idx in range(n_samples):
        hop = "one_hop" if idx < n_one_hop else "two_hop"
        cat = categories[idx % len(categories)]
        sample_id = f"modal_mi_{cat}_{hop}_{idx}"
        sample = generate_single_modal_mi_sample(
            sample_id=sample_id,
            category_name=cat,
            rng=rng,
            hop=hop,
            prompt_order=prompt_order,
        )
        rows.append(sample)

    rng.shuffle(rows)
    return rows
