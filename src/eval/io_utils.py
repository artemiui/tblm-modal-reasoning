from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    """Read a JSONL file into a list of dicts."""
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')


def balanced_sample_by_rule(rows: List[Dict[str, object]], max_samples: int, seed: int = 42) -> List[Dict[str, object]]:
    """Sample rows with balanced representation across rules/axioms.
    Uses the 'rule' or 'axiom' field for balancing."""
    if max_samples <= 0 or len(rows) <= max_samples:
        return list(rows)
    rng = random.Random(seed)
    # Group by rule/axiom
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        key = str(row.get('rule', row.get('axiom', 'unknown')))
        groups.setdefault(key, []).append(row)
    # Allocate evenly
    n_groups = len(groups)
    per_group = max(1, max_samples // n_groups)
    sampled = []
    for key in sorted(groups):
        pool = groups[key]
        rng.shuffle(pool)
        sampled.extend(pool[:per_group])
    rng.shuffle(sampled)
    return sampled[:max_samples]
