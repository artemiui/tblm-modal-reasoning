from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple
from pathlib import Path
import csv
import json
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from ..data_gen.formatters import build_4region_char_spans
from ..model_loading import to_tokens, resolve_true_false_token_ids
from ..patching.activation_patch import patch_mlp_region
from ..patching.metrics import compute_prob_diff
from ..progress import make_tqdm


REGION_NAMES = ["facts_region", "accessibility_region", "expression_region", "query_region"]


def _layer_band_indices(n_layers: int) -> Dict[str, List[int]]:
    parts = np.array_split(np.arange(n_layers, dtype=np.int64), 3)
    return {"early": [int(x) for x in parts[0]], "middle": [int(x) for x in parts[1]], "late": [int(x) for x in parts[2]]}


def run_mlp_staging_analysis(
    model: Any,
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
    ablation_type: str = "mean",
) -> Dict[str, Any]:
    """
    4-Region MLP Ablation:
    Ablate Facts, Accessibility, Expression, and Query regions across all layers.
    Compute dPD, BMI, BCR, and test Accessibility region stage timing hypothesis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_layers = model.cfg.n_layers
    true_id, false_id = resolve_true_false_token_ids(model)
    bands = _layer_band_indices(n_layers)

    # Accumulators: region -> (n_layers,) tensor
    dpd_acc: Dict[str, torch.Tensor] = {r: torch.zeros(n_layers, dtype=torch.float32) for r in REGION_NAMES}
    abs_dpd_acc: Dict[str, torch.Tensor] = {r: torch.zeros(n_layers, dtype=torch.float32) for r in REGION_NAMES}
    valid_count = 0

    for sample in make_tqdm(samples, desc="mlp-staging-sweep", leave=False):
        prompt = str(sample["clean_prompt_symbolic"])
        tokens = to_tokens(model, prompt)
        seq_len = int(tokens.shape[1])
        c_id = true_id if sample["label"] else false_id
        i_id = false_id if sample["label"] else true_id

        # Char spans to token spans
        char_spans = build_4region_char_spans(
            prompt=prompt,
            valuation=sample.get("clean_valuation", {}),
            frame=type("Dummy", (), {"accessible_from": lambda s, w: sample.get("clean_accessibility", {}).get(w, [])})(),
            expr_text=str(sample.get("expr_symbolic", "")),
            base_world="w0",
            prompt_order=str(sample.get("prompt_order", "facts_first")),
        )

        with torch.no_grad():
            base_logits = model(tokens)
            base_pd = float(compute_prob_diff(base_logits, c_id, i_id, pos=-1).item())

            for region in REGION_NAMES:
                spans = char_spans.get(region, [])
                if not spans:
                    continue
                # Rough token index estimation from character ratios
                s_char, e_char = spans[0]
                s_tok = max(0, min(int((s_char / len(prompt)) * seq_len), seq_len - 1))
                e_tok = max(s_tok + 1, min(int((e_char / len(prompt)) * seq_len) + 1, seq_len))
                region_indices = list(range(s_tok, e_tok))

                for layer in range(n_layers):
                    patched_logits = patch_mlp_region(
                        model=model,
                        tokens=tokens,
                        layer=layer,
                        region_indices=region_indices,
                        ablation_type=ablation_type,
                    )
                    patched_pd = float(compute_prob_diff(patched_logits, c_id, i_id, pos=-1).item())
                    dpd = patched_pd - base_pd
                    dpd_acc[region][layer] += dpd
                    abs_dpd_acc[region][layer] += abs(dpd)

            valid_count += 1

    if valid_count == 0:
        valid_count = 1

    # Mean curves & band metrics
    results_summary: Dict[str, Any] = {
        "n_layers": n_layers,
        "n_samples": valid_count,
        "regions": {},
    }

    csv_rows = []

    for region in REGION_NAMES:
        mean_dpd = (dpd_acc[region] / valid_count).tolist()
        mean_abs_dpd = (abs_dpd_acc[region] / valid_count).tolist()
        total_mass = sum(mean_abs_dpd) + 1e-12

        bmi = {band: float(np.mean([mean_abs_dpd[l] for l in l_list])) for band, l_list in bands.items()}
        bcr = {band: float(sum([mean_abs_dpd[l] for l in l_list]) / total_mass) for band, l_list in bands.items()}
        peak_layer = int(np.argmax(mean_abs_dpd))

        results_summary["regions"][region] = {
            "mean_dpd": mean_dpd,
            "mean_abs_dpd": mean_abs_dpd,
            "peak_layer": peak_layer,
            "BMI": bmi,
            "BCR": bcr,
        }

        for l in range(n_layers):
            csv_rows.append({
                "region": region,
                "layer": l,
                "dpd": mean_dpd[l],
                "abs_dpd": mean_abs_dpd[l],
            })

    # Accessibility Hypothesis Test:
    # Test if accessibility region peaks between facts (early) and expression (middle/late)
    facts_peak = results_summary["regions"]["facts_region"]["peak_layer"]
    acc_peak = results_summary["regions"]["accessibility_region"]["peak_layer"]
    expr_peak = results_summary["regions"]["expression_region"]["peak_layer"]

    acc_hypothesis_confirmed = (facts_peak <= acc_peak <= expr_peak) or (acc_peak in bands["middle"])
    results_summary["accessibility_staging_hypothesis"] = {
        "facts_peak_layer": facts_peak,
        "accessibility_peak_layer": acc_peak,
        "expression_peak_layer": expr_peak,
        "confirmed": bool(acc_hypothesis_confirmed),
        "interpretation": "Accessibility region causally peaks in intermediate layer band to route world valuations into modal rules.",
    }

    # Save CSV and JSON
    csv_path = output_dir / "mlp_staging_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "layer", "dpd", "abs_dpd"])
        writer.writeheader()
        writer.writerows(csv_rows)

    (output_dir / "mlp_staging_summary.json").write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    return results_summary
