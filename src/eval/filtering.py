from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.progress import log_event, resolve_log_path, setup_file_logger
from .io_utils import read_jsonl, write_jsonl

def filter_dual_correct(rows: List[Dict[str, object]], require_label_change: bool = False) -> List[Dict[str, object]]:
    kept = []
    for row in rows:
        if not row.get('correct_clean', False):
            continue
        if not row.get('correct_corrupted', False):
            continue
        if require_label_change and bool(row.get('label')) == bool(row.get('label_corrupted')):
            continue
        kept.append(row)
    return kept

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require_label_change', action='store_true')
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    kept = filter_dual_correct(rows, require_label_change=args.require_label_change)
    write_jsonl(args.output, kept)

if __name__ == '__main__':
    main()
