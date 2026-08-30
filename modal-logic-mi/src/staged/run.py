from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml
from ..model_loading import load_hooked_transformer
from ..data_gen.mi_pairs import generate_modal_mi_dataset
from .mlp_staging import run_mlp_staging_analysis
from .info_transmission import run_information_transmission
from .fact_retrospection import run_fact_retrospection_contrast
from .specialized_heads import run_specialized_heads_analysis
from ..progress import setup_file_logger, log_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Part B Runner: Mechanistic Principles")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_id = cfg["model_id"]
    output_dir = Path(cfg.get("output_dir", f"results/part_b/{model_id.replace('/', '_')}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_file_logger(__name__, output_dir / "part_b.log")
    log_event(logger, {"stage": "part_b_start", "model_id": model_id, "config": str(args.config)})

    model = load_hooked_transformer(
        model_id,
        device=cfg.get("device", "cuda"),
        source=cfg.get("model_source", "huggingface"),
        torch_dtype=cfg.get("dtype", "float16"),
    )

    # Generate or load dataset
    samples = generate_modal_mi_dataset(
        n_samples=cfg.get("n_samples", 100),
        seed=cfg.get("seed", 42),
        prompt_order=cfg.get("prompt_order", "facts_first"),
    )

    # 1. Staged Computation (4-region MLP zero/mean ablation)
    mlp_res = run_mlp_staging_analysis(model, samples, output_dir / "mlp_analysis")

    # 2. Information Transmission
    info_res = run_information_transmission(model, samples, output_dir / "info_transmission")

    # 3. Fact Retrospection Contrast
    retro_res = run_fact_retrospection_contrast(model, samples, output_dir / "fact_retrospection")

    # 4. Specialized Attention Heads
    heads_res = run_specialized_heads_analysis(model, samples, output_dir / "specialized_heads")

    summary = {
        "model_id": model_id,
        "mlp_staging": mlp_res,
        "information_transmission": info_res,
        "fact_retrospection": retro_res,
        "specialized_heads": heads_res,
    }
    (output_dir / "part_b_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log_event(logger, {"stage": "part_b_done", "summary": str(output_dir / "part_b_summary.json")})


if __name__ == "__main__":
    main()
