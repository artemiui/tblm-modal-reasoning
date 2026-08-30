from __future__ import annotations

import argparse
import json
from pathlib import Path
from .mi_pairs import generate_modal_mi_dataset
from .circuit_pairs import generate_all_circuit_pairs
from ..progress import setup_file_logger, log_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Modal Logic Dataset Generator")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset_type", choices=["modal_mi", "modal_circuit"], default="modal_mi")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--prompt_order", choices=["facts_first", "expr_first"], default="facts_first")
    parser.add_argument("--one_hop_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_file_logger(__name__)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset_type == "modal_mi":
        rows = generate_modal_mi_dataset(
            n_samples=args.n_samples,
            seed=args.seed,
            prompt_order=args.prompt_order,
            one_hop_ratio=args.one_hop_ratio,
        )
        with args.output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        log_event(logger, {"dataset_type": "modal_mi", "count": len(rows), "output": str(args.output)})
    else:
        pairs = generate_all_circuit_pairs(n_per_type=max(1, args.n_samples // 7), seed=args.seed)
        rows = [
            {
                "pair_type": p.pair_type,
                "clean_prompt": p.clean_prompt,
                "counterfactual_prompt": p.counterfactual_prompt,
                "clean_target": p.clean_target,
                "counterfactual_target": p.counterfactual_target,
                "clean_label": p.clean_label,
                "counterfactual_label": p.counterfactual_label,
                "metadata": p.metadata,
            }
            for p in pairs
        ]
        with args.output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        log_event(logger, {"dataset_type": "modal_circuit", "count": len(rows), "output": str(args.output)})


if __name__ == "__main__":
    main()
