from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Expr:
    op: str
    value: Optional[object] = None
    children: Tuple["Expr", ...] = ()


def Var(name: str) -> Expr:
    return Expr(op="var", value=name)


def Const(value: bool) -> Expr:
    return Expr(op="const", value=bool(value))


def Not(x: Expr) -> Expr:
    return Expr(op="not", children=(x,))


def And(x: Expr, y: Expr) -> Expr:
    return Expr(op="and", children=(x, y))


def Or(x: Expr, y: Expr) -> Expr:
    return Expr(op="or", children=(x, y))


def Implies(x: Expr, y: Expr) -> Expr:
    return Expr(op="implies", children=(x, y))


def Iff(x: Expr, y: Expr) -> Expr:
    return Expr(op="iff", children=(x, y))


def Xor(x: Expr, y: Expr) -> Expr:
    """Exclusive Disjunction operator (XOR)."""
    return Expr(op="xor", children=(x, y))


def Box(x: Expr) -> Expr:
    """Modal Necessity operator (\u25a1)."""
    return Expr(op="box", children=(x,))


def Diamond(x: Expr) -> Expr:
    """Modal Possibility operator (\u25c7)."""
    return Expr(op="diamond", children=(x,))


@dataclass
class KripkeFrame:
    worlds: List[str] = field(default_factory=lambda: ["w0", "w1"])
    # accessibility mapping: world -> list of accessible worlds
    accessibility: Dict[str, List[str]] = field(default_factory=lambda: {"w0": ["w0", "w1"], "w1": ["w1"]})

    def accessible_from(self, world: str) -> List[str]:
        return self.accessibility.get(world, [])

    def clone(self) -> "KripkeFrame":
        return KripkeFrame(
            worlds=list(self.worlds),
            accessibility={k: list(v) for k, v in self.accessibility.items()},
        )


@dataclass
class KripkeModel:
    frame: KripkeFrame
    valuation: Dict[str, Dict[str, bool]]  # world -> {var: bool}

    def get_fact(self, world: str, var_name: str) -> bool:
        if world not in self.valuation:
            return False
        return self.valuation[world].get(var_name, False)

    def clone(self) -> "KripkeModel":
        return KripkeModel(
            frame=self.frame.clone(),
            valuation={w: dict(facts) for w, facts in self.valuation.items()},
        )


def eval_modal_expr(expr: Expr, model: KripkeModel, current_world: str = "w0") -> bool:
    """Recursively evaluate a modal or propositional AST under a Kripke model."""
    if expr.op == "const":
        return bool(expr.value)
    if expr.op == "var":
        return model.get_fact(current_world, str(expr.value))
    if expr.op == "not":
        return not eval_modal_expr(expr.children[0], model, current_world)

    if expr.op in {"and", "or", "xor", "implies", "iff"}:
        left = eval_modal_expr(expr.children[0], model, current_world)
        right = eval_modal_expr(expr.children[1], model, current_world)
        if expr.op == "and":
            return left and right
        if expr.op == "or":
            return left or right
        if expr.op == "xor":
            return left != right
        if expr.op == "implies":
            return (not left) or right
        if expr.op == "iff":
            return left == right

    if expr.op == "box":
        acc_worlds = model.frame.accessible_from(current_world)
        if not acc_worlds:
            return True  # Vacuously true if no accessible worlds
        return all(eval_modal_expr(expr.children[0], model, w) for w in acc_worlds)

    if expr.op == "diamond":
        acc_worlds = model.frame.accessible_from(current_world)
        if not acc_worlds:
            return False  # False if no accessible worlds
        return any(eval_modal_expr(expr.children[0], model, w) for w in acc_worlds)

    raise ValueError(f"Unknown operator {expr.op!r}")


def collect_vars(expr: Expr) -> List[str]:
    out: List[str] = []

    def walk(node: Expr) -> None:
        if node.op == "var":
            out.append(str(node.value))
            return
        for child in node.children:
            walk(child)

    walk(expr)
    return sorted(set(out))


def rename_vars(expr: Expr, mapping: Dict[str, str]) -> Expr:
    if expr.op == "var":
        old_name = str(expr.value)
        return Var(mapping.get(old_name, old_name))
    if not expr.children:
        return expr
    return Expr(
        op=expr.op,
        value=expr.value,
        children=tuple(rename_vars(child, mapping) for child in expr.children),
    )


_PRECEDENCE = {
    "iff": 1,
    "implies": 2,
    "xor": 3,
    "or": 3,
    "and": 4,
    "not": 5,
    "box": 5,
    "diamond": 5,
    "var": 6,
    "const": 6,
}


def _maybe_wrap(child: Expr, parent_op: str) -> str:
    if _PRECEDENCE[child.op] < _PRECEDENCE[parent_op]:
        return f"({to_symbolic(child)})"
    return to_symbolic(child)


def to_symbolic(expr: Expr) -> str:
    if expr.op == "const":
        return "True" if expr.value else "False"
    if expr.op == "var":
        return str(expr.value)
    if expr.op == "not":
        inner = expr.children[0]
        rendered = to_symbolic(inner)
        if inner.op in {"var", "const", "not"}:
            return f"not {rendered}"
        return f"not ({rendered})"
    if expr.op == "box":
        inner = expr.children[0]
        return f"box({to_symbolic(inner)})"
    if expr.op == "diamond":
        inner = expr.children[0]
        return f"diamond({to_symbolic(inner)})"

    left, right = expr.children
    if expr.op == "and":
        return f"{_maybe_wrap(left, 'and')} and {_maybe_wrap(right, 'and')}"
    if expr.op == "or":
        return f"{_maybe_wrap(left, 'or')} or {_maybe_wrap(right, 'or')}"
    if expr.op == "xor":
        return f"{_maybe_wrap(left, 'xor')} xor {_maybe_wrap(right, 'xor')}"
    if expr.op == "implies":
        return f"{_maybe_wrap(left, 'implies')} -> {_maybe_wrap(right, 'implies')}"
    if expr.op == "iff":
        return f"{_maybe_wrap(left, 'iff')} <-> {_maybe_wrap(right, 'iff')}"
    raise ValueError(f"Unknown operator: {expr.op}")


def to_natural(expr: Expr) -> str:
    if expr.op == "const":
        return "true" if expr.value else "false"
    if expr.op == "var":
        return str(expr.value)
    if expr.op == "not":
        return f"not {to_natural(expr.children[0])}"
    if expr.op == "box":
        return f"necessarily {to_natural(expr.children[0])}"
    if expr.op == "diamond":
        return f"possibly {to_natural(expr.children[0])}"

    left, right = expr.children
    if expr.op == "and":
        return f"{to_natural(left)} and {to_natural(right)}"
    if expr.op == "or":
        return f"{to_natural(left)} or {to_natural(right)}"
    if expr.op == "xor":
        return f"{to_natural(left)} xor {to_natural(right)}"
    if expr.op == "implies":
        return f"{to_natural(left)} implies {to_natural(right)}"
    if expr.op == "iff":
        return f"{to_natural(left)} is equivalent to {to_natural(right)}"
    raise ValueError(f"Unknown operator: {expr.op}")


def format_accessibility_clause(frame: KripkeFrame, base_world: str = "w0", style: str = "token_markers") -> str:
    acc = frame.accessible_from(base_world)
    if style == "token_markers":
        return f"ACCESS_START {' '.join(acc)} ACCESS_END"
    return f"From {base_world}, accessible worlds are [{', '.join(acc)}]."


def format_world_facts(valuation: Dict[str, Dict[str, bool]], style: str = "standard") -> str:
    lines = []
    for w in sorted(valuation.keys()):
        facts = valuation[w]
        f_str = ", ".join(f"{k} is {v}" for k, v in sorted(facts.items()))
        lines.append(f"In world {w}: {f_str}")
    return "; ".join(lines) if style == "compact" else ". ".join(lines) + "."


# Few-shot exemplars for Part A (Hong et al. in-context matching)
FEW_SHOT_4_SHOT_MISTRAL = """Example 1:
ACCESS_START w0 w1 ACCESS_END
In world w0: P is True, Q is False. In world w1: P is True, Q is True.
[Rule 1] necessarily P implies Q. [Rule 2] R implies S.
Query: Is Q necessarily true from w0?
Answer: True.

Example 2:
ACCESS_START w0 w1 ACCESS_END
In world w0: P is True, Q is False. In world w1: P is False, Q is False.
[Rule 1] necessarily P implies Q. [Rule 2] R implies S.
Query: Is Q necessarily true from w0?
Answer: False.

Example 3:
ACCESS_START w0 w1 ACCESS_END
In world w0: A is False. In world w1: A is True.
[Rule 1] possibly A. [Rule 2] B implies C.
Query: Is A possibly true from w0?
Answer: True.

Example 4:
ACCESS_START w0 ACCESS_END
In world w0: A is False.
[Rule 1] possibly A. [Rule 2] B implies C.
Query: Is A possibly true from w0?
Answer: False.
"""

FEW_SHOT_6_SHOT_GEMMA = FEW_SHOT_4_SHOT_MISTRAL + """
Example 5:
ACCESS_START w0 w1 ACCESS_END
In world w0: P is True, Q is True. In world w1: P is True, Q is False.
[Rule 1] necessarily P implies Q. [Rule 2] A implies B.
Query: Is Q necessarily true from w0?
Answer: False.

Example 6:
ACCESS_START w0 w1 ACCESS_END
In world w0: P is False, Q is False. In world w1: P is True, Q is True.
[Rule 1] possibly P and Q. [Rule 2] C implies D.
Query: Is P and Q possibly true from w0?
Answer: True.
"""
