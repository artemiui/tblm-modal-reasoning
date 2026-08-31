from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.kripke import KripkeModel

@dataclass(frozen=True)
class Expr:
    op: str  # 'var', 'const', 'not', 'and', 'or', 'xor', 'implies', 'iff', 'box', 'diamond'
    value: Optional[object] = None
    children: Tuple['Expr', ...] = ()

def Var(name: str) -> Expr:
    return Expr(op='var', value=name)

def Const(value: bool) -> Expr:
    return Expr(op='const', value=value)

def Not(x: Expr) -> Expr:
    return Expr(op='not', children=(x,))

def And(x: Expr, y: Expr) -> Expr:
    return Expr(op='and', children=(x, y))

def Or(x: Expr, y: Expr) -> Expr:
    return Expr(op='or', children=(x, y))

def Xor(x: Expr, y: Expr) -> Expr:
    return Expr(op='xor', children=(x, y))

def Implies(x: Expr, y: Expr) -> Expr:
    return Expr(op='implies', children=(x, y))

def Iff(x: Expr, y: Expr) -> Expr:
    return Expr(op='iff', children=(x, y))

def Box(x: Expr) -> Expr:
    return Expr(op='box', children=(x,))

def Diamond(x: Expr) -> Expr:
    return Expr(op='diamond', children=(x,))

def eval_expr(expr: Expr, env: Dict[str, bool]) -> bool:
    if expr.op == 'var':
        return env[str(expr.value)]
    elif expr.op == 'const':
        return bool(expr.value)
    elif expr.op == 'not':
        return not eval_expr(expr.children[0], env)
    elif expr.op == 'and':
        return eval_expr(expr.children[0], env) and eval_expr(expr.children[1], env)
    elif expr.op == 'or':
        return eval_expr(expr.children[0], env) or eval_expr(expr.children[1], env)
    elif expr.op == 'xor':
        return eval_expr(expr.children[0], env) != eval_expr(expr.children[1], env)
    elif expr.op == 'implies':
        return not eval_expr(expr.children[0], env) or eval_expr(expr.children[1], env)
    elif expr.op == 'iff':
        return eval_expr(expr.children[0], env) == eval_expr(expr.children[1], env)
    else:
        raise ValueError(f"Operator {expr.op} not supported in eval_expr")

def eval_modal(expr: Expr, model: 'KripkeModel', world: str) -> bool:
    if expr.op == 'var':
        return model.valuation[world][str(expr.value)]
    elif expr.op == 'const':
        return bool(expr.value)
    elif expr.op == 'not':
        return not eval_modal(expr.children[0], model, world)
    elif expr.op == 'and':
        return eval_modal(expr.children[0], model, world) and eval_modal(expr.children[1], model, world)
    elif expr.op == 'or':
        return eval_modal(expr.children[0], model, world) or eval_modal(expr.children[1], model, world)
    elif expr.op == 'xor':
        return eval_modal(expr.children[0], model, world) != eval_modal(expr.children[1], model, world)
    elif expr.op == 'implies':
        return not eval_modal(expr.children[0], model, world) or eval_modal(expr.children[1], model, world)
    elif expr.op == 'iff':
        return eval_modal(expr.children[0], model, world) == eval_modal(expr.children[1], model, world)
    elif expr.op == 'box':
        return all(eval_modal(expr.children[0], model, w) for w in model.accessible(world))
    elif expr.op == 'diamond':
        return any(eval_modal(expr.children[0], model, w) for w in model.accessible(world))
    else:
        raise ValueError(f"Operator {expr.op} not supported in eval_modal")

def collect_vars(expr: Expr) -> List[str]:
    vars_set = set()
    def walk(e: Expr):
        if e.op == 'var':
            vars_set.add(str(e.value))
        for child in e.children:
            walk(child)
    walk(expr)
    return sorted(list(vars_set))

def rename_vars(expr: Expr, mapping: Dict[str, str]) -> Expr:
    if expr.op == 'var':
        name = str(expr.value)
        return Var(mapping.get(name, name))
    elif expr.op == 'const':
        return expr
    elif expr.op in ('not', 'box', 'diamond'):
        return Expr(op=expr.op, children=(rename_vars(expr.children[0], mapping),))
    else:
        return Expr(op=expr.op, children=(rename_vars(expr.children[0], mapping), rename_vars(expr.children[1], mapping)))

def modal_depth(expr: Expr) -> int:
    if expr.op in ('var', 'const'):
        return 0
    elif expr.op in ('not',):
        return modal_depth(expr.children[0])
    elif expr.op in ('box', 'diamond'):
        return 1 + modal_depth(expr.children[0])
    else:
        return max(modal_depth(expr.children[0]), modal_depth(expr.children[1]))

def is_modal(expr: Expr) -> bool:
    if expr.op in ('box', 'diamond'):
        return True
    return any(is_modal(c) for c in expr.children)

def is_propositional(expr: Expr) -> bool:
    return not is_modal(expr)

def _precedence(op: str) -> int:
    if op in ('var', 'const'):
        return 6
    if op in ('not', 'box', 'diamond'):
        return 5
    if op in ('and',):
        return 4
    if op in ('or', 'xor'):
        return 3
    if op in ('implies',):
        return 2
    if op in ('iff',):
        return 1
    return 0

def to_symbolic(expr: Expr) -> str:
    if expr.op == 'var':
        return str(expr.value)
    elif expr.op == 'const':
        return str(expr.value)
    elif expr.op == 'not':
        child_str = to_symbolic(expr.children[0])
        if _precedence(expr.children[0].op) < _precedence('not'):
            child_str = f"({child_str})"
        return f"not {child_str}"
    elif expr.op == 'box':
        child_str = to_symbolic(expr.children[0])
        if _precedence(expr.children[0].op) < _precedence('box'):
            child_str = f"({child_str})"
        return f"□{child_str}"
    elif expr.op == 'diamond':
        child_str = to_symbolic(expr.children[0])
        if _precedence(expr.children[0].op) < _precedence('diamond'):
            child_str = f"({child_str})"
        return f"◇{child_str}"
    else:
        op_map = {'and': 'and', 'or': 'or', 'xor': 'xor', 'implies': '->', 'iff': '<->'}
        op_sym = op_map[expr.op]
        left = to_symbolic(expr.children[0])
        right = to_symbolic(expr.children[1])
        
        if _precedence(expr.children[0].op) < _precedence(expr.op):
            left = f"({left})"
        if _precedence(expr.children[1].op) <= _precedence(expr.op): # left-associative
            right = f"({right})"
            
        return f"{left} {op_sym} {right}"

def to_semi_natural(expr: Expr) -> str:
    if expr.op == 'var':
        return str(expr.value)
    elif expr.op == 'const':
        return str(expr.value)
    elif expr.op == 'not':
        return f"Not({to_semi_natural(expr.children[0])})"
    elif expr.op == 'box':
        return f"Necessarily({to_semi_natural(expr.children[0])})"
    elif expr.op == 'diamond':
        return f"Possibly({to_semi_natural(expr.children[0])})"
    else:
        op_map = {'and': 'And', 'or': 'Or', 'xor': 'Xor', 'implies': 'Implies', 'iff': 'EquivalentTo'}
        op_str = op_map[expr.op]
        return f"{to_semi_natural(expr.children[0])} {op_str} {to_semi_natural(expr.children[1])}"

def to_verbose(expr: Expr) -> str:
    return to_semi_natural(expr)  # Functionality is same as to_semi_natural based on prompt instructions

import re

def tokenize(text: str) -> List[str]:
    # Handles: `□`, `◇`, `and`, `or`, `not`, `xor`, `->`, `<->`, `(`, `)`, `~`, `!`, `&`, `|`, uppercase identifiers, `True`, `False`.
    pattern = r'(<->|->|□|◇|and|or|not|xor|True|False|[A-Z]\w*|\(|\)|~|!|&|\|)'
    tokens = re.findall(pattern, text)
    return tokens

def parse_expression(text: str) -> Expr:
    tokens = tokenize(text)
    pos = 0

    def parse_iff() -> Expr:
        nonlocal pos
        left = parse_implies()
        while pos < len(tokens) and tokens[pos] == '<->':
            pos += 1
            right = parse_implies()
            left = Iff(left, right)
        return left

    def parse_implies() -> Expr:
        nonlocal pos
        left = parse_or()
        while pos < len(tokens) and tokens[pos] == '->':
            pos += 1
            right = parse_or()
            left = Implies(left, right)
        return left

    def parse_or() -> Expr:
        nonlocal pos
        left = parse_and()
        while pos < len(tokens) and tokens[pos] in ('or', '|', 'xor'):
            op = tokens[pos]
            pos += 1
            right = parse_and()
            if op == 'xor':
                left = Xor(left, right)
            else:
                left = Or(left, right)
        return left

    def parse_and() -> Expr:
        nonlocal pos
        left = parse_unary()
        while pos < len(tokens) and tokens[pos] in ('and', '&'):
            pos += 1
            right = parse_unary()
            left = And(left, right)
        return left

    def parse_unary() -> Expr:
        nonlocal pos
        if pos < len(tokens) and tokens[pos] in ('not', '~', '!'):
            pos += 1
            return Not(parse_unary())
        elif pos < len(tokens) and tokens[pos] == '□':
            pos += 1
            return Box(parse_unary())
        elif pos < len(tokens) and tokens[pos] == '◇':
            pos += 1
            return Diamond(parse_unary())
        return parse_primary()

    def parse_primary() -> Expr:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
        token = tokens[pos]
        if token == '(':
            pos += 1
            expr = parse_iff()
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError("Expected ')'")
            pos += 1
            return expr
        elif token == 'True':
            pos += 1
            return Const(True)
        elif token == 'False':
            pos += 1
            return Const(False)
        elif re.match(r'^[A-Z]\w*$', token):
            pos += 1
            return Var(token)
        else:
            raise ValueError(f"Unexpected token: {token}")

    if not tokens:
        raise ValueError("Empty expression")
        
    expr = parse_iff()
    if pos < len(tokens):
        raise ValueError(f"Unexpected token at end: {tokens[pos]}")
    return expr
