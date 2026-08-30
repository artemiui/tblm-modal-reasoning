from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .modal_grammar import (
    Box, Const, Diamond, Expr, Implies, KripkeFrame, KripkeModel,
    Not, Var, eval_modal_expr, format_accessibility_clause, format_world_facts,
    FEW_SHOT_4_SHOT_MISTRAL, FEW_SHOT_6_SHOT_GEMMA
)


@dataclass
class CircuitSamplePair:
    pair_type: str
    clean_prompt: str
    counterfactual_prompt: str
    clean_target: str
    counterfactual_target: str
    clean_label: bool
    counterfactual_label: bool
    metadata: Dict[str, object]


def _build_part_a_prompt(
    frame: KripkeFrame,
    valuation: Dict[str, Dict[str, bool]],
    rule1_text: str,
    rule2_text: str,
    query_text: str,
    few_shot: str = "",
) -> str:
    acc_clause = format_accessibility_clause(frame, base_world="w0", style="token_markers")
    facts_str = format_world_facts(valuation, style="compact")
    prompt = f"{acc_clause}\n{facts_str}\n[Rule 1] {rule1_text}. [Rule 2] {rule2_text}.\nQuery: {query_text}\nAnswer:"
    if few_shot:
        return f"{few_shot.strip()}\n\n{prompt}"
    return prompt


def query_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip QUERY between modal-chain conclusion and linear-chain conclusion."""
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    valuation = {
        "w0": {"P": True, "Q": True, "R": True, "S": False},
        "w1": {"P": True, "Q": True, "R": False, "S": False},
    }
    model = KripkeModel(frame, valuation)

    rule1 = "necessarily P implies Q"
    rule2 = "R implies S"

    clean_query = "Is Q necessarily true from w0?"
    cf_query = "Is S true in w0?"

    clean_label = eval_modal_expr(Box(Var("Q")), model, "w0")
    cf_label = valuation["w0"]["S"]

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(frame, valuation, rule1, rule2, clean_query, few_shot)
    cf_prompt = _build_part_a_prompt(frame, valuation, rule1, rule2, cf_query, few_shot)

    return CircuitSamplePair(
        pair_type="query_flip",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="True" if clean_label else "False",
        counterfactual_target="True" if cf_label else "False",
        clean_label=clean_label,
        counterfactual_label=cf_label,
        metadata={"query_clean": clean_query, "query_cf": cf_query},
    )


def modal_operator_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip \u25a1 <-> \u25c7 only, holding everything else identical."""
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    valuation = {
        "w0": {"P": False, "Q": False},
        "w1": {"P": True, "Q": True},
    }
    model = KripkeModel(frame, valuation)

    rule1_clean = "necessarily P implies Q"
    rule1_cf = "possibly P implies Q"
    rule2 = "R implies S"

    query = "Is Q true in w0?"
    clean_label = False
    cf_label = True

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(frame, valuation, rule1_clean, rule2, query, few_shot)
    cf_prompt = _build_part_a_prompt(frame, valuation, rule1_cf, rule2, query, few_shot)

    return CircuitSamplePair(
        pair_type="modal_operator_flip",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="False",
        counterfactual_target="True",
        clean_label=clean_label,
        counterfactual_label=cf_label,
        metadata={"operator_clean": "box", "operator_cf": "diamond"},
    )


def accessibility_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Change accessible-world set only."""
    frame_clean = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    frame_cf = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0"]})

    valuation = {
        "w0": {"P": False, "Q": False, "R": True, "S": False},
        "w1": {"P": True, "Q": True, "R": False, "S": False},
    }

    model_clean = KripkeModel(frame_clean, valuation)
    model_cf = KripkeModel(frame_cf, valuation)

    rule1 = "possibly P"
    rule2 = "R implies S"
    query = "Is P possibly true from w0?"

    clean_label = eval_modal_expr(Diamond(Var("P")), model_clean, "w0")
    cf_label = eval_modal_expr(Diamond(Var("P")), model_cf, "w0")

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(frame_clean, valuation, rule1, rule2, query, few_shot)
    cf_prompt = _build_part_a_prompt(frame_cf, valuation, rule1, rule2, query, few_shot)

    return CircuitSamplePair(
        pair_type="accessibility_flip",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="True",
        counterfactual_target="False",
        clean_label=clean_label,
        counterfactual_label=cf_label,
        metadata={"acc_clean": ["w0", "w1"], "acc_cf": ["w0"]},
    )


def fact_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip a fact's truth value within a fixed world."""
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    valuation_clean = {
        "w0": {"P": True, "Q": True, "R": True, "S": False},
        "w1": {"P": True, "Q": True, "R": False, "S": False},
    }
    valuation_cf = {
        "w0": {"P": True, "Q": True, "R": True, "S": False},
        "w1": {"P": False, "Q": True, "R": False, "S": False},
    }

    model_clean = KripkeModel(frame, valuation_clean)
    model_cf = KripkeModel(frame, valuation_cf)

    rule1 = "necessarily P implies Q"
    rule2 = "R implies S"
    query = "Is P necessarily true from w0?"

    clean_label = eval_modal_expr(Box(Var("P")), model_clean, "w0")
    cf_label = eval_modal_expr(Box(Var("P")), model_cf, "w0")

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(frame, valuation_clean, rule1, rule2, query, few_shot)
    cf_prompt = _build_part_a_prompt(frame, valuation_cf, rule1, rule2, query, few_shot)

    return CircuitSamplePair(
        pair_type="fact_flip",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="True",
        counterfactual_target="False",
        clean_label=clean_label,
        counterfactual_label=cf_label,
        metadata={"flipped_world": "w1", "flipped_var": "P"},
    )


def rule_location_swap_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Swap order of modal rule and linear rule."""
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    valuation = {
        "w0": {"P": True, "Q": True, "R": True, "S": False},
        "w1": {"P": True, "Q": True, "R": False, "S": False},
    }
    model = KripkeModel(frame, valuation)

    modal_rule = "necessarily P implies Q"
    linear_rule = "R implies S"
    query = "Is Q necessarily true from w0?"

    clean_label = eval_modal_expr(Box(Var("Q")), model, "w0")

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(frame, valuation, modal_rule, linear_rule, query, few_shot)
    cf_prompt = _build_part_a_prompt(frame, valuation, linear_rule, modal_rule, query, few_shot)

    return CircuitSamplePair(
        pair_type="rule_location_swap",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="True" if clean_label else "False",
        counterfactual_target="True" if clean_label else "False",
        clean_label=clean_label,
        counterfactual_label=clean_label,
        metadata={"swap": "rule1_rule2_order"},
    )


def graded_operator_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip graded probability operator 'probably' (majority >50%) <-> 'certainly' (100%)."""
    frame = KripkeFrame(worlds=["w0", "w1", "w2"], accessibility={"w0": ["w0", "w1", "w2"]})
    valuation = {
        "w0": {"P": True, "Q": False, "R": True, "S": False},
        "w1": {"P": True, "Q": True, "R": False, "S": False},
        "w2": {"P": False, "Q": False, "R": False, "S": False},
    }
    model = KripkeModel(frame, valuation)

    rule1_clean = "probably P"
    rule1_cf = "certainly P"
    rule2 = "R implies S"

    query_clean = "Is P probably true from w0?"
    query_cf = "Is P certainly true from w0?"

    clean_label = True   # 2 of 3 accessible worlds have P=True (>50%)
    cf_label = False     # not all 3 accessible worlds have P=True (<100%)

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(frame, valuation, rule1_clean, rule2, query_clean, few_shot)
    cf_prompt = _build_part_a_prompt(frame, valuation, rule1_cf, rule2, query_cf, few_shot)

    return CircuitSamplePair(
        pair_type="graded_operator_flip",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="True",
        counterfactual_target="False",
        clean_label=clean_label,
        counterfactual_label=cf_label,
        metadata={"operator_clean": "probably", "operator_cf": "certainly", "p_count": "2_of_3"},
    )


def connective_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip Boolean connective 'or' <-> 'and' under modal context."""
    frame = KripkeFrame(worlds=["w0", "w1"], accessibility={"w0": ["w0", "w1"]})
    valuation = {
        "w0": {"P": True, "Q": False, "R": True, "S": False},
        "w1": {"P": True, "Q": False, "R": False, "S": False},
    }
    model = KripkeModel(frame, valuation)

    rule1_clean = "necessarily P or Q"
    rule1_cf = "necessarily P and Q"
    rule2 = "R implies S"

    query_clean = "Is P or Q necessarily true from w0?"
    query_cf = "Is P and Q necessarily true from w0?"

    clean_label = True   # In all worlds, True or False = True
    cf_label = False     # In all worlds, True and False = False

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(frame, valuation, rule1_clean, rule2, query_clean, few_shot)
    cf_prompt = _build_part_a_prompt(frame, valuation, rule1_cf, rule2, query_cf, few_shot)

    return CircuitSamplePair(
        pair_type="connective_flip",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="True",
        counterfactual_target="False",
        clean_label=clean_label,
        counterfactual_label=cf_label,
        metadata={"connective_clean": "or", "connective_cf": "and"},
    )


def generate_all_circuit_pairs(n_per_type: int = 60, seed: int = 42, few_shot_style: str = "4shot") -> List[CircuitSamplePair]:
    rng = random.Random(seed)
    generators = [
        query_flip_pairs,
        modal_operator_flip_pairs,
        accessibility_flip_pairs,
        fact_flip_pairs,
        rule_location_swap_pairs,
        graded_operator_flip_pairs,
        connective_flip_pairs,
    ]
    all_pairs: List[CircuitSamplePair] = []
    for gen in generators:
        for _ in range(n_per_type):
            all_pairs.append(gen(rng, few_shot_style=few_shot_style))
    return all_pairs
