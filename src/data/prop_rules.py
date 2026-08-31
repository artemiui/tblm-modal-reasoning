from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Dict, List

from .modal_ast import And, Const, Expr, Iff, Not, Or, Var


@dataclass(frozen=True)
class RuleTemplate:
    name: str
    category: str
    formal_definition: str
    build: Callable[[Dict[str, Expr], random.Random], Expr]


def _build_identity(v: Dict[str, Expr], rng: random.Random) -> Expr:
    if rng.choice([True, False]):
        return And(v['A'], Const(True))
    else:
        return Or(v['A'], Const(False))


def _build_domination(v: Dict[str, Expr], rng: random.Random) -> Expr:
    if rng.choice([True, False]):
        return And(v['A'], Const(False))
    else:
        return Or(v['A'], Const(True))


def _build_idempotent(v: Dict[str, Expr], rng: random.Random) -> Expr:
    if rng.choice([True, False]):
        return And(v['A'], v['A'])
    else:
        return Or(v['A'], v['A'])


def _build_associative(v: Dict[str, Expr], rng: random.Random) -> Expr:
    if rng.choice([True, False]):
        return And(And(v['A'], v['B']), v['C'])
    else:
        return And(v['A'], And(v['B'], v['C']))


PROP_RULE_TEMPLATES: List[RuleTemplate] = [
    RuleTemplate(
        name='identity',
        category='basic',
        formal_definition='P and T <-> P ; P or F <-> P',
        build=_build_identity
    ),
    RuleTemplate(
        name='domination',
        category='basic',
        formal_definition='P and F <-> F ; P or T <-> T',
        build=_build_domination
    ),
    RuleTemplate(
        name='idempotent',
        category='basic',
        formal_definition='P and P <-> P ; P or P <-> P',
        build=_build_idempotent
    ),
    RuleTemplate(
        name='double_negation',
        category='basic',
        formal_definition='not(not P) <-> P',
        build=lambda v, rng: Not(Not(v['A']))
    ),
    RuleTemplate(
        name='excluded_middle',
        category='basic',
        formal_definition='P or not P <-> T',
        build=lambda v, rng: Or(v['A'], Not(v['A']))
    ),
    RuleTemplate(
        name='contradiction',
        category='basic',
        formal_definition='P and not P <-> F',
        build=lambda v, rng: And(v['A'], Not(v['A']))
    ),
    RuleTemplate(
        name='commutative',
        category='basic',
        formal_definition='P and Q <-> Q and P',
        build=lambda v, rng: And(v['A'], v['B'])
    ),
    RuleTemplate(
        name='associative',
        category='basic',
        formal_definition='(P and Q) and R <-> P and (Q and R)',
        build=_build_associative
    ),
    RuleTemplate(
        name='distributive',
        category='basic',
        formal_definition='P and (Q or R) <-> (P and Q) or (P and R)',
        build=lambda v, rng: And(v['A'], Or(v['B'], v['C']))
    ),
    RuleTemplate(
        name='demorgan',
        category='basic',
        formal_definition='not(P and Q) <-> (not P) or (not Q)',
        build=lambda v, rng: Or(Not(v['A']), Not(v['B']))
    ),
    RuleTemplate(
        name='absorption',
        category='basic',
        formal_definition='P and (P or Q) <-> P',
        build=lambda v, rng: And(v['A'], Or(v['A'], v['B']))
    ),
]


def build_one_hop_expr(rule_name: str, rng: random.Random) -> Expr:
    """Build a one-hop expression for a given rule name."""
    template = get_rule_template(rule_name)
    v = {'A': Var('A'), 'B': Var('B'), 'C': Var('C')}
    return template.build(v, rng)


def all_rule_names() -> List[str]:
    """Return a list of all propositional rule names."""
    return [t.name for t in PROP_RULE_TEMPLATES]


def get_rule_template(rule_name: str) -> RuleTemplate:
    """Get the rule template for a given rule name."""
    for t in PROP_RULE_TEMPLATES:
        if t.name == rule_name:
            return t
    raise ValueError(f"Unknown rule name: {rule_name}")
