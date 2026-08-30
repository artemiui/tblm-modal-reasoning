from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .modal_grammar import (
    Box, Const, Diamond, Expr, Implies, KripkeFrame, KripkeModel,
    Not, Var, eval_modal_expr, format_proposition_facts,
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
    facts_str: str,
    rule1_text: str,
    rule2_text: str,
    query_text: str,
    few_shot: str = "",
) -> str:
    prompt = f"Facts: {facts_str}\n[Rule 1] {rule1_text}. [Rule 2] {rule2_text}.\nQuery: {query_text}\nAnswer:"
    if few_shot:
        return f"{few_shot.strip()}\n\n{prompt}"
    return prompt


def query_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip QUERY between modal-chain conclusion and linear-chain conclusion."""
    facts = {"P": True, "Q": True, "R": True, "S": False}
    facts_str = format_proposition_facts(facts, style="compact")

    rule1 = "necessarily P implies Q"
    rule2 = "R implies S"

    clean_query = "Is Q necessarily true?"
    cf_query = "Is S true?"

    clean_label = True
    cf_label = False

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(facts_str, rule1, rule2, clean_query, few_shot)
    cf_prompt = _build_part_a_prompt(facts_str, rule1, rule2, cf_query, few_shot)

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
    """Flip modal operator □ <-> ◇ in modal proposition rule."""
    facts = {"P": False, "Q": False, "R": True, "S": False}
    facts_str = format_proposition_facts(facts, style="compact")

    rule1_clean = "necessarily P implies Q"
    rule1_cf = "possibly P implies Q"
    rule2 = "R implies S"

    query = "Is Q true?"
    clean_label = False
    cf_label = True

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(facts_str, rule1_clean, rule2, query, few_shot)
    cf_prompt = _build_part_a_prompt(facts_str, rule1_cf, rule2, query, few_shot)

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


def modal_proposition_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip modal proposition / axiom premise (e.g. Axiom D: □P -> ◇P vs □P -> □Q)."""
    facts = {"P": True, "Q": False, "R": True, "S": False}
    facts_str = format_proposition_facts(facts, style="compact")

    rule1_clean = "necessarily P implies possibly P"  # Axiom D
    rule1_cf = "necessarily P implies necessarily Q"
    rule2 = "R implies S"
    query = "Is P possibly true?"

    clean_label = True
    cf_label = False

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(facts_str, rule1_clean, rule2, query, few_shot)
    cf_prompt = _build_part_a_prompt(facts_str, rule1_cf, rule2, query, few_shot)

    return CircuitSamplePair(
        pair_type="modal_proposition_flip",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="True",
        counterfactual_target="False",
        clean_label=clean_label,
        counterfactual_label=cf_label,
        metadata={"axiom": "Axiom D", "prop_clean": rule1_clean, "prop_cf": rule1_cf},
    )


# Backward compatibility alias
accessibility_flip_pairs = modal_proposition_flip_pairs


def fact_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip a proposition fact truth value."""
    facts_clean = {"P": True, "Q": True, "R": True, "S": False}
    facts_cf = {"P": False, "Q": True, "R": True, "S": False}

    facts_str_clean = format_proposition_facts(facts_clean, style="compact")
    facts_str_cf = format_proposition_facts(facts_cf, style="compact")

    rule1 = "necessarily P implies Q"
    rule2 = "R implies S"
    query = "Is P necessarily true?"

    clean_label = True
    cf_label = False

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(facts_str_clean, rule1, rule2, query, few_shot)
    cf_prompt = _build_part_a_prompt(facts_str_cf, rule1, rule2, query, few_shot)

    return CircuitSamplePair(
        pair_type="fact_flip",
        clean_prompt=clean_prompt,
        counterfactual_prompt=cf_prompt,
        clean_target="True",
        counterfactual_target="False",
        clean_label=clean_label,
        counterfactual_label=cf_label,
        metadata={"flipped_var": "P"},
    )


def rule_location_swap_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Swap order of modal proposition rule and linear rule."""
    facts = {"P": True, "Q": True, "R": True, "S": False}
    facts_str = format_proposition_facts(facts, style="compact")

    modal_rule = "necessarily P implies Q"
    linear_rule = "R implies S"
    query = "Is Q necessarily true?"

    clean_label = True

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(facts_str, modal_rule, linear_rule, query, few_shot)
    cf_prompt = _build_part_a_prompt(facts_str, linear_rule, modal_rule, query, few_shot)

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


def connective_flip_pairs(rng: random.Random, few_shot_style: str = "4shot") -> CircuitSamplePair:
    """Flip Boolean connective 'or' <-> 'and' under modal proposition context."""
    facts = {"P": True, "Q": False, "R": True, "S": False}
    facts_str = format_proposition_facts(facts, style="compact")

    rule1_clean = "necessarily P or Q"
    rule1_cf = "necessarily P and Q"
    rule2 = "R implies S"

    query_clean = "Is P or Q necessarily true?"
    query_cf = "Is P and Q necessarily true?"

    clean_label = True   # True or False = True
    cf_label = False     # True and False = False

    few_shot = FEW_SHOT_4_SHOT_MISTRAL if few_shot_style == "4shot" else (FEW_SHOT_6_SHOT_GEMMA if few_shot_style == "6shot" else "")

    clean_prompt = _build_part_a_prompt(facts_str, rule1_clean, rule2, query_clean, few_shot)
    cf_prompt = _build_part_a_prompt(facts_str, rule1_cf, rule2, query_cf, few_shot)

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
        modal_proposition_flip_pairs,
        fact_flip_pairs,
        rule_location_swap_pairs,
        connective_flip_pairs,
    ]
    all_pairs: List[CircuitSamplePair] = []
    for gen in generators:
        for _ in range(n_per_type):
            all_pairs.append(gen(rng, few_shot_style=few_shot_style))
    return all_pairs
