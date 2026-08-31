from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

from src.progress import log_event, resolve_log_path, setup_file_logger

from .modal_ast import (
    And, Box, Const, Diamond, Expr, Not, Or, Var,
    collect_vars, eval_expr, eval_modal, is_modal, modal_depth,
    rename_vars, to_semi_natural, to_symbolic,
)
from .kripke import (
    KripkeFrame, KripkeModel,
    generate_frame_for_axiom, generate_model, format_frame, format_valuation,
)
from .prop_rules import (
    PROP_RULE_TEMPLATES, all_rule_names as all_prop_rule_names,
    build_one_hop_expr, get_rule_template as get_prop_rule_template,
)
from .modal_rules import (
    MODAL_RULE_TEMPLATES, AXIOM_FRAME_REQUIREMENTS,
    all_modal_rule_names, build_modal_expr,
    get_modal_rule_template,
)
from .formatters import (
    PROMPT_ENDING_CHOICES, build_depth_prompt, build_modal_depth_prompt,
    build_prop_prompt, build_modal_prompt,
)
from .corruptions import (
    find_label_flipping_fact_corruption, find_modal_corruption,
)

VAR_POOL = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in {'F', 'T'}]

def _sample_facts(variables: Iterable[str], rng: random.Random) -> Dict[str, bool]:
    return {v: rng.choice([True, False]) for v in variables}

def _iter_bool_envs(variables: Sequence[str]) -> Iterable[Dict[str, bool]]:
    if not variables:
        yield {}
        return
    for values in product([True, False], repeat=len(variables)):
        yield dict(zip(variables, values))

def _choose_fresh_vars(rng: random.Random, used: set[str], count: int) -> List[str]:
    available = [v for v in VAR_POOL if v not in used]
    return rng.sample(available, count)

def _remap_expr_variables(expr: Expr, rng: random.Random) -> Tuple[Expr, Dict[str, str]]:
    vars_in_expr = sorted(list(collect_vars(expr)))
    fresh = _choose_fresh_vars(rng, set(), len(vars_in_expr))
    mapping = dict(zip(vars_in_expr, fresh))
    return rename_vars(expr, mapping), mapping

def _record_prop_one_hop(sample_id: int, rule_name: str, expr: Expr, rng: random.Random, prompt_order: str, prompt_ending: str) -> Dict[str, Any]:
    vars_in_expr = sorted(list(collect_vars(expr)))
    facts = _sample_facts(vars_in_expr, rng)
    label = eval_expr(expr, facts)
    
    try:
        corrupt_facts, corrupt_label = find_label_flipping_fact_corruption(expr, facts)
    except ValueError:
        corrupt_facts, corrupt_label = facts, label
        
    clean_prompt = build_prop_prompt(
        facts=facts,
        expr_text=to_symbolic(expr),
        prompt_order=prompt_order,
        prompt_ending=prompt_ending
    )
    
    corrupt_prompt = build_prop_prompt(
        facts=corrupt_facts,
        expr_text=to_symbolic(expr),
        prompt_order=prompt_order,
        prompt_ending=prompt_ending
    )
    
    return {
        "id": f"prop_1hop_{rule_name}_{sample_id}",
        "hop": 1,
        "rule": rule_name,
        "expr_symbolic": to_symbolic(expr),
        "facts": facts,
        "label": label,
        "clean_prompt_symbolic": clean_prompt,
        "corrupted_facts": corrupt_facts,
        "corrupted_label": corrupt_label,
        "corrupted_prompt_symbolic": corrupt_prompt,
        "meta": {"corruption": "fact_flip"}
    }

def _record_prop_two_hop(sample_id: int, rule_name: str, rng: random.Random, prompt_order: str, prompt_ending: str) -> Dict[str, Any]:
    expr = build_one_hop_expr(rule_name, rng)
    expr, _ = _remap_expr_variables(expr, rng)
    vars_in_expr = sorted(list(collect_vars(expr)))
    
    fresh = _choose_fresh_vars(rng, set(vars_in_expr), 2)
    m_var = fresh[0]
    g_var = fresh[1]
    
    op = rng.choice([And, Or])
    if rng.choice([True, False]):
        two_hop_expr = op(Var(m_var), Var(g_var))
        full_expr = op(expr, Var(g_var))
    else:
        two_hop_expr = op(Var(g_var), Var(m_var))
        full_expr = op(Var(g_var), expr)
        
    facts_l1 = _sample_facts(vars_in_expr, rng)
    facts_l2 = {g_var: rng.choice([True, False])}
    
    all_facts = {**facts_l1, **facts_l2}
    label = eval_expr(full_expr, all_facts)
    
    try:
        corrupt_facts, corrupt_label = find_label_flipping_fact_corruption(full_expr, all_facts)
    except ValueError:
        corrupt_facts, corrupt_label = all_facts, label
    
    clean_prompt = build_depth_prompt(
        hop=2,
        facts=all_facts,
        query_expr_text=to_symbolic(two_hop_expr),
        derived_steps=[(m_var, to_symbolic(expr))],
        prompt_order=prompt_order,
        prompt_ending=prompt_ending
    )
    corrupt_prompt = build_depth_prompt(
        hop=2,
        facts=corrupt_facts,
        query_expr_text=to_symbolic(two_hop_expr),
        derived_steps=[(m_var, to_symbolic(expr))],
        prompt_order=prompt_order,
        prompt_ending=prompt_ending
    )
    
    return {
        "id": f"prop_2hop_{rule_name}_{sample_id}",
        "hop": 2,
        "rule": rule_name,
        "expr_symbolic": to_symbolic(full_expr),
        "facts": all_facts,
        "label": label,
        "clean_prompt_symbolic": clean_prompt,
        "corrupted_facts": corrupt_facts,
        "corrupted_label": corrupt_label,
        "corrupted_prompt_symbolic": corrupt_prompt,
        "meta": {"corruption": "fact_flip"}
    }

def _record_modal_one_hop(sample_id: int, axiom_name: str, rng: random.Random, n_worlds: int, prompt_style: str, prompt_ending: str) -> Dict[str, Any]:
    for _ in range(10):
        try:
            expr = build_modal_expr(axiom_name, rng)
            expr, _ = _remap_expr_variables(expr, rng)
            
            frame = generate_frame_for_axiom(axiom_name, rng, n_worlds)
            model = generate_model(frame, sorted(list(collect_vars(expr))), rng)
            eval_world = rng.choice(frame.worlds)
            
            label = eval_modal(expr, model, eval_world)
            
            corrupted_model, corrupted_expr, corrupted_label, corruption_type = find_modal_corruption(
                expr, model, eval_world, rng
            )
            
            clean_prompt = build_modal_prompt(
                frame_worlds=list(model.frame.worlds),
                frame_accessibility={k: list(v) for k, v in model.frame.relation.items()},
                valuation={w: dict(model.valuation[w]) for w in model.frame.worlds},
                eval_world=eval_world,
                expr_text=to_symbolic(expr),
                prompt_style=prompt_style,
                prompt_ending=prompt_ending
            )
            corrupt_prompt = build_modal_prompt(
                frame_worlds=list(corrupted_model.frame.worlds),
                frame_accessibility={k: list(v) for k, v in corrupted_model.frame.relation.items()},
                valuation={w: dict(corrupted_model.valuation[w]) for w in corrupted_model.frame.worlds},
                eval_world=eval_world,
                expr_text=to_symbolic(corrupted_expr),
                prompt_style=prompt_style,
                prompt_ending=prompt_ending
            )
            
            return {
                "id": f"modal_1hop_{axiom_name}_{sample_id}",
                "hop": 1,
                "axiom": axiom_name,
                "frame": {
                    "worlds": list(model.frame.worlds),
                    "relation": {k: list(v) for k, v in model.frame.relation.items()}
                },
                "valuation": {w: dict(model.valuation[w]) for w in model.frame.worlds},
                "eval_world": eval_world,
                "expr_symbolic": to_symbolic(expr),
                "expr_semi_natural": to_semi_natural(expr),
                "label": label,
                "label_corrupted": corrupted_label,
                "clean_prompt_symbolic": clean_prompt,
                "corrupted_prompt_symbolic": corrupt_prompt,
                "corrupted_frame": {
                    "worlds": list(corrupted_model.frame.worlds),
                    "relation": {k: list(v) for k, v in corrupted_model.frame.relation.items()}
                } if corruption_type == "relation" else None,
                "corrupted_valuation": {w: dict(corrupted_model.valuation[w]) for w in corrupted_model.frame.worlds} if corruption_type == "valuation" else None,
                "corrupted_expr_symbolic": to_symbolic(corrupted_expr) if corruption_type == "operator" else None,
                "modal_depth": modal_depth(expr),
                "n_worlds": n_worlds,
                "meta": {"corruption": corruption_type}
            }
        except ValueError:
            continue
    raise ValueError(f"Could not generate record for {axiom_name} after 10 retries.")

def _record_modal_two_hop(sample_id: int, axiom_name: str, rng: random.Random, n_worlds: int, prompt_style: str, prompt_ending: str) -> Dict[str, Any]:
    for _ in range(10):
        try:
            expr = build_modal_expr(axiom_name, rng)
            expr, _ = _remap_expr_variables(expr, rng)
            vars_in_expr = sorted(list(collect_vars(expr)))
            
            fresh = _choose_fresh_vars(rng, set(vars_in_expr), 2)
            m_var = fresh[0]
            g_var = fresh[1]
            
            op = rng.choice([And, Or])
            two_hop_expr = op(Var(m_var), Var(g_var)) if rng.choice([True, False]) else op(Var(g_var), Var(m_var))
            
            frame = generate_frame_for_axiom(axiom_name, rng, n_worlds)
            model = generate_model(frame, vars_in_expr + [g_var], rng)
            eval_world = rng.choice(frame.worlds)
            
            l1_val = eval_modal(expr, model, eval_world)
            combined_env = {m_var: l1_val, g_var: model.valuation[eval_world][g_var]}
            label = eval_expr(two_hop_expr, combined_env)
            
            full_expr = op(expr, Var(g_var)) if two_hop_expr.children[0] == Var(m_var) else op(Var(g_var), expr)
            
            corrupted_model, corrupted_expr, corrupted_label, corruption_type = find_modal_corruption(
                full_expr, model, eval_world, rng
            )
            
            clean_prompt = build_modal_depth_prompt(
                hop=2,
                frame_worlds=list(model.frame.worlds),
                frame_accessibility={k: list(v) for k, v in model.frame.relation.items()},
                valuation={w: dict(model.valuation[w]) for w in model.frame.worlds},
                eval_world=eval_world,
                query_expr_text=to_symbolic(two_hop_expr),
                derived_steps=[(m_var, to_symbolic(expr))],
                prompt_style=prompt_style,
                prompt_ending=prompt_ending
            )
            corrupt_expr_for_step = corrupted_expr if corruption_type == "operator" else expr
            corrupt_prompt = build_modal_depth_prompt(
                hop=2,
                frame_worlds=list(corrupted_model.frame.worlds),
                frame_accessibility={k: list(v) for k, v in corrupted_model.frame.relation.items()},
                valuation={w: dict(corrupted_model.valuation[w]) for w in corrupted_model.frame.worlds},
                eval_world=eval_world,
                query_expr_text=to_symbolic(two_hop_expr),
                derived_steps=[(m_var, to_symbolic(corrupt_expr_for_step))],
                prompt_style=prompt_style,
                prompt_ending=prompt_ending
            )
            
            return {
                "id": f"modal_2hop_{axiom_name}_{sample_id}",
                "hop": 2,
                "axiom": axiom_name,
                "frame": {
                    "worlds": list(model.frame.worlds),
                    "relation": {k: list(v) for k, v in model.frame.relation.items()}
                },
                "valuation": {w: dict(model.valuation[w]) for w in model.frame.worlds},
                "eval_world": eval_world,
                "expr_symbolic": to_symbolic(full_expr),
                "expr_semi_natural": to_semi_natural(full_expr),
                "label": label,
                "label_corrupted": corrupted_label,
                "clean_prompt_symbolic": clean_prompt,
                "corrupted_prompt_symbolic": corrupt_prompt,
                "modal_depth": modal_depth(full_expr),
                "n_worlds": n_worlds,
                "meta": {"corruption": corruption_type}
            }
        except ValueError:
            continue
    raise ValueError(f"Could not generate 2-hop record for {axiom_name} after 10 retries.")

def generate_prop_records(per_rule_per_hop: int, seed: int, prompt_order: str, prompt_ending: str) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for rule in all_prop_rule_names():
        for i in range(per_rule_per_hop):
            expr, _ = _remap_expr_variables(build_one_hop_expr(rule, rng), rng)
            records.append(_record_prop_one_hop(i, rule, expr, rng, prompt_order, prompt_ending))
            records.append(_record_prop_two_hop(i, rule, rng, prompt_order, prompt_ending))
    return records

def generate_modal_records(per_axiom_per_hop: int, seed: int, n_worlds: int, prompt_style: str, prompt_ending: str) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for axiom in all_modal_rule_names():
        for i in range(per_axiom_per_hop):
            try:
                records.append(_record_modal_one_hop(i, axiom, rng, n_worlds, prompt_style, prompt_ending))
                records.append(_record_modal_two_hop(i, axiom, rng, n_worlds, prompt_style, prompt_ending))
            except ValueError as e:
                print(f"Skipping sample {i} for axiom {axiom}: {e}")
    return records

def generate_combined_records(prop_per_rule: int, modal_per_axiom: int, seed: int, n_worlds: int, prompt_style: str, prompt_ending: str) -> List[Dict[str, Any]]:
    prop = generate_prop_records(prop_per_rule, seed, 'facts_first', prompt_ending)
    mod = generate_modal_records(modal_per_axiom, seed, n_worlds, prompt_style, prompt_ending)
    rng = random.Random(seed)
    combined = prop + mod
    rng.shuffle(combined)
    return combined

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate modal/propositional logic datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['propositional', 'modal', 'combined'], default='modal')
    parser.add_argument('--per_rule_per_hop', type=int, default=200)
    parser.add_argument('--per_axiom_per_hop', type=int, default=200)
    parser.add_argument('--n_worlds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt_order', choices=['facts_first', 'expr_first'], default='facts_first')
    parser.add_argument('--prompt_style', choices=['symbolic', 'semi_natural', 'verbose'], default='symbolic')
    parser.add_argument('--prompt_ending', choices=list(PROMPT_ENDING_CHOICES), default='answer_suffix')
    args = parser.parse_args()
    
    log_path = resolve_log_path(args.output)
    logger = setup_file_logger("generate_dataset", log_path)
    if args.mode == 'propositional':
        records = generate_prop_records(args.per_rule_per_hop, args.seed, args.prompt_order, args.prompt_ending)
    elif args.mode == 'modal':
        records = generate_modal_records(args.per_axiom_per_hop, args.seed, args.n_worlds, args.prompt_style, args.prompt_ending)
    else:
        records = generate_combined_records(args.per_rule_per_hop, args.per_axiom_per_hop, args.seed, args.n_worlds, args.prompt_style, args.prompt_ending)
        
    _write_jsonl(args.output, records)
    log_event(logger, {"event": "DATASET_GENERATED", "count": len(records), "mode": args.mode, "output": str(args.output)})

if __name__ == '__main__':
    main()
