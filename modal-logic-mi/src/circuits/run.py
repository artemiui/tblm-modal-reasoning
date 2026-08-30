from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml
from ..model_loading import load_hooked_transformer
from ..data_gen.circuit_pairs import generate_all_circuit_pairs
from .head_discovery import discover_circuit
from .head_classify import classify_heads
from .sufficiency_table import verify_sufficiency, export_sufficiency_table
from ..viz.circuit_diagram import render_circuit_diagram
from ..progress import setup_file_logger, log_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Part A Runner: Circuit Discovery & CMA")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_id = cfg["model_id"]
    output_dir = Path(cfg.get("output_dir", f"results/part_a/{model_id.replace('/', '_')}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_file_logger(__name__, output_dir / "part_a.log")
    log_event(logger, {"stage": "part_a_start", "model_id": model_id, "config": str(args.config)})

    model = load_hooked_transformer(
        model_id,
        device=cfg.get("device", "cuda"),
        source=cfg.get("model_source", "huggingface"),
        torch_dtype=cfg.get("dtype", "float16"),
    )

    pairs = generate_all_circuit_pairs(
        n_per_type=cfg.get("samples_per_type", 30),
        seed=cfg.get("seed", 42),
        few_shot_style=cfg.get("few_shot_style", "4shot"),
    )
    pairs_by_type = {}
    for p in pairs:
        pairs_by_type.setdefault(p.pair_type, []).append(p)

    # Step 1: Discover Circuit
    top_heads, effect_matrix = discover_circuit(model, pairs, threshold=cfg.get("threshold", 0.05))
    circuit_heads = [(int(h["layer"]), int(h["head"])) for h in top_heads]

    # Step 2: Classify Heads
    families = classify_heads(model, top_heads, pairs_by_type)

    # Step 3: Sufficiency Table
    suff_table = verify_sufficiency(model, circuit_heads, families, pairs)
    export_sufficiency_table(suff_table, output_dir)

    # Step 4: Render Visualizations
    render_circuit_diagram(families, output_dir / "circuit_diagram.png")

    summary = {
        "model_id": model_id,
        "n_candidate_heads": len(circuit_heads),
        "circuit_heads": circuit_heads,
        "families": {k: [list(h) for h in v] for k, v in families.items()},
        "sufficiency_table": suff_table,
    }
    (output_dir / "circuit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_event(logger, {"stage": "part_a_done", "summary": str(output_dir / "circuit_summary.json")})


if __name__ == "__main__":
    main()
