from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Dict, List

from .modal_ast import And, Box, Const, Diamond, Expr, Iff, Implies, Not, Or, Var


@dataclass(frozen=True)
class RuleTemplate:
    name: str
    category: str
    formal_definition: str
    build: Callable[[Dict[str, Expr], random.Random], Expr]


MODAL_RULE_TEMPLATES: List[RuleTemplate] = [
    RuleTemplate(
        name='axiom_K',
        category='distribution',
        formal_definition='□(P -> Q) -> (□P -> □Q)',
        build=lambda v, rng: Implies(Box(Implies(v['A'], v['B'])), Implies(Box(v['A']), Box(v['B'])))
    ),
    RuleTemplate(
        name='axiom_T',
        category='reflexivity',
        formal_definition='□P -> P',
        build=lambda v, rng: Implies(Box(v['A']), v['A'])
    ),
    RuleTemplate(
        name='axiom_D',
        category='seriality',
        formal_definition='□P -> ◇P',
        build=lambda v, rng: Implies(Box(v['A']), Diamond(v['A']))
    ),
    RuleTemplate(
        name='axiom_4',
        category='transitivity',
        formal_definition='□P -> □□P',
        build=lambda v, rng: Implies(Box(v['A']), Box(Box(v['A'])))
    ),
    RuleTemplate(
        name='axiom_5',
        category='euclidean',
        formal_definition='◇P -> □◇P',
        build=lambda v, rng: Implies(Diamond(v['A']), Box(Diamond(v['A'])))
    ),
    RuleTemplate(
        name='axiom_B',
        category='symmetry',
        formal_definition='P -> □◇P',
        build=lambda v, rng: Implies(v['A'], Box(Diamond(v['A'])))
    ),
    RuleTemplate(
        name='modal_duality_box',
        category='duality',
        formal_definition='□P <-> ¬◇¬P',
        build=lambda v, rng: Iff(Box(v['A']), Not(Diamond(Not(v['A']))))
    ),
    RuleTemplate(
        name='modal_duality_diamond',
        category='duality',
        formal_definition='◇P <-> ¬□¬P',
        build=lambda v, rng: Iff(Diamond(v['A']), Not(Box(Not(v['A']))))
    ),
    RuleTemplate(
        name='modal_modus_ponens',
        category='inference',
        formal_definition='□(P -> Q) and □P -> □Q',
        build=lambda v, rng: Implies(And(Box(Implies(v['A'], v['B'])), Box(v['A'])), Box(v['B']))
    ),
    RuleTemplate(
        name='necessitation',
        category='inference',
        formal_definition='if P is a tautology, then □P is valid',
        build=lambda v, rng: Box(Or(v['A'], Not(v['A'])))
    ),
    RuleTemplate(
        name='box_and_distribution',
        category='distribution',
        formal_definition='□(P and Q) -> (□P and □Q)',
        build=lambda v, rng: Implies(Box(And(v['A'], v['B'])), And(Box(v['A']), Box(v['B'])))
    ),
    RuleTemplate(
        name='diamond_or_distribution',
        category='distribution',
        formal_definition='(◇P or ◇Q) -> ◇(P or Q)',
        build=lambda v, rng: Implies(Or(Diamond(v['A']), Diamond(v['B'])), Diamond(Or(v['A'], v['B'])))
    ),
]


def build_modal_expr(rule_name: str, rng: random.Random) -> Expr:
    """Build a modal expression for a given rule name."""
    template = get_modal_rule_template(rule_name)
    v = {'A': Var('A'), 'B': Var('B'), 'C': Var('C')}
    return template.build(v, rng)


def all_modal_rule_names() -> List[str]:
    """Return a list of all modal rule names."""
    return [t.name for t in MODAL_RULE_TEMPLATES]


def get_modal_rule_template(rule_name: str) -> RuleTemplate:
    """Get the rule template for a given modal rule name."""
    for t in MODAL_RULE_TEMPLATES:
        if t.name == rule_name:
            return t
    raise ValueError(f"Unknown modal rule name: {rule_name}")


AXIOM_FRAME_REQUIREMENTS: Dict[str, str] = {
    'axiom_K': 'arbitrary',
    'axiom_T': 'reflexive',
    'axiom_D': 'serial',
    'axiom_4': 'transitive',
    'axiom_5': 'euclidean',
    'axiom_B': 'symmetric',
    'modal_duality_box': 'arbitrary',
    'modal_duality_diamond': 'arbitrary',
    'modal_modus_ponens': 'arbitrary',
    'necessitation': 'arbitrary',
    'box_and_distribution': 'arbitrary',
    'diamond_or_distribution': 'arbitrary',
}
