from __future__ import annotations

import argparse
import json
from pathlib import Path
from helpers.modal_problem_generation import generate_cot_question_query_based
from helpers.patching_helpers_custom import logits_diff, basic_metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Modal Circuit CMA Patching Sweep")
    parser.add_argument("--model_id", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--output_dir", type=Path, default=Path("results/part_a/gemma9b"))
    parser.add_argument("--n_samples", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting CMA Patching Sweep for {args.model_id} with {args.n_samples} samples...")

    # Generate sample pairs
    pairs = [generate_cot_question_query_based(length_of_chain=2, num_cot_samples=4) for _ in range(args.n_samples)]
    print(f"Generated {len(pairs)} modal prompt pairs successfully.")

    summary = {
        "model_id": args.model_id,
        "n_samples": len(pairs),
        "status": "ready_for_gpu_execution",
    }
    (args.output_dir / "patching_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Sweep metadata written to {args.output_dir / 'patching_summary.json'}")


if __name__ == "__main__":
    main()
