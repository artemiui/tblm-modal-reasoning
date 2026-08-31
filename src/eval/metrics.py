from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from src.progress import log_event, setup_file_logger
from .io_utils import read_jsonl

def _mean(items: Iterable[float]) -> float:
    items_list = list(items)
    if not items_list:
        return 0.0
    return sum(items_list) / len(items_list)

def summarize_predictions(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
        
    count = len(rows)
    acc_clean = _mean([float(r.get('correct_clean', False)) for r in rows])
    acc_corr = _mean([float(r.get('correct_corrupted', False)) for r in rows])
    dual_corr = _mean([float(r.get('correct_clean', False) and r.get('correct_corrupted', False)) for r in rows])
    
    acc_one_hop = _mean([float(r.get('correct_clean', False)) for r in rows if r.get('hop') == 'one_hop'])
    acc_two_hop = _mean([float(r.get('correct_clean', False)) for r in rows if r.get('hop') == 'two_hop'])
    
    rule_acc = {}
    axiom_acc = {}
    system_acc = {}
    depth_acc = {}
    world_acc = {}
    
    for row in rows:
        c = float(row.get('correct_clean', False))
        if 'rule' in row:
            rule_acc.setdefault(str(row['rule']), []).append(c)
        if 'axiom' in row:
            axiom_acc.setdefault(str(row['axiom']), []).append(c)
        if 'system' in row:
            system_acc.setdefault(str(row['system']), []).append(c)
        if 'modal_depth' in row:
            depth_acc.setdefault(str(row['modal_depth']), []).append(c)
            
        if 'frame' in row and isinstance(row['frame'], dict) and 'worlds' in row['frame']:
            wc = len(row['frame']['worlds'])
            world_acc.setdefault(str(wc), []).append(c)
            
    return {
        'count': count,
        'accuracy_clean': acc_clean,
        'accuracy_corrupted': acc_corr,
        'dual_correct_rate': dual_corr,
        'accuracy_by_hop': {
            'one_hop': acc_one_hop,
            'two_hop': acc_two_hop
        },
        'accuracy_by_rule': {k: _mean(v) for k, v in rule_acc.items()},
        'accuracy_by_axiom': {k: _mean(v) for k, v in axiom_acc.items()},
        'accuracy_by_system': {k: _mean(v) for k, v in system_acc.items()},
        'accuracy_by_modal_depth': {k: _mean(v) for k, v in depth_acc.items()},
        'accuracy_by_world_count': {k: _mean(v) for k, v in world_acc.items()}
    }

def compare_modal_vs_propositional(modal_rows: List[Dict[str, object]], prop_rows: List[Dict[str, object]]) -> Dict[str, float]:
    m_acc = sum([float(r.get('correct_clean', False)) for r in modal_rows]) / max(1, len(modal_rows))
    p_acc = sum([float(r.get('correct_clean', False)) for r in prop_rows]) / max(1, len(prop_rows))
    return {
        'modal_accuracy': m_acc,
        'prop_accuracy': p_acc,
        'accuracy_gap': p_acc - m_acc
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after', type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    summary = summarize_predictions(rows)
    
    out_dict = {'before': summary}
    
    if args.after:
        after_rows = read_jsonl(args.after)
        after_summary = summarize_predictions(after_rows)
        out_dict['after'] = after_summary
        
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2)

if __name__ == '__main__':
    main()
