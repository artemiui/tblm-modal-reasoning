from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from pathlib import Path
import csv
import json
from ..model_loading import to_tokens, resolve_true_false_token_ids
from ..patching.cma import cma_sufficiency_test
from ..data_gen.circuit_pairs import CircuitSamplePair


def verify_sufficiency(
    model: Any,
    circuit_heads: Sequence[Tuple[int, int]],
    families: Dict[str, List[Tuple[int, int]]],
    pairs: Sequence[CircuitSamplePair],
) -> List[Dict[str, Any]]:
    """
    Produce ablation table:
      - Full Circuit (C)
      - C - MOH
      - C - WAH
      - C - QRLH
      - C - QRMH
      - C - FPH
      - C - DH
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    true_id, false_id = resolve_true_false_token_ids(model)

    ablation_conditions: Dict[str, List[Tuple[int, int]]] = {
        "Full Circuit (C)": list(circuit_heads),
    }
    for fam_name, fam_heads in families.items():
        if fam_heads:
            ablation_conditions[f"C - {fam_name}"] = [h for h in circuit_heads if h not in fam_heads]

    results: List[Dict[str, Any]] = []

    for cond_name, active_heads in ablation_conditions.items():
        retained_cld_vals: List[float] = []

        for pair in pairs[:20]:
            clean_toks = to_tokens(model, pair.clean_prompt)
            cf_toks = to_tokens(model, pair.counterfactual_prompt)
            if clean_toks.shape[1] != cf_toks.shape[1]:
                continue

            c_id = true_id if pair.clean_label else false_id
            i_id = false_id if pair.clean_label else true_id

            clean_logits = model(clean_toks)
            cf_logits, cf_cache = model.run_with_cache(cf_toks)

            clean_ld = float((clean_logits[:, -1, c_id] - clean_logits[:, -1, i_id]).item())
            cf_ld = float((cf_logits[:, -1, c_id] - cf_logits[:, -1, i_id]).item())

            cld = cma_sufficiency_test(
                model=model,
                clean_tokens=clean_toks,
                corrupted_cache=cf_cache,
                circuit_heads=active_heads,
                correct_id=c_id,
                incorrect_id=i_id,
                clean_ld=clean_ld,
                corrupted_ld=cf_ld,
                n_layers=n_layers,
                n_heads=n_heads,
            )
            retained_cld_vals.append(cld)

        mean_cld = sum(retained_cld_vals) / max(1, len(retained_cld_vals))
        results.append({
            "Condition": cond_name,
            "Active Heads": len(active_heads),
            "Calibrated Logit Diff (%)": round(mean_cld * 100, 2),
        })

    return results


def export_sufficiency_table(results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sufficiency_table.csv"
    md_path = output_dir / "sufficiency_table.md"
    tex_path = output_dir / "sufficiency_table.tex"

    # CSV
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Condition", "Active Heads", "Calibrated Logit Diff (%)"])
        writer.writeheader()
        writer.writerows(results)

    # Markdown
    md_lines = ["| Condition | Active Heads | Calibrated Logit Diff (%) |", "|---|---:|---:|"]
    for r in results:
        md_lines.append(f"| {r['Condition']} | {r['Active Heads']} | {r['Calibrated Logit Diff (%)']}% |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # LaTeX
    tex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Condition & Active Heads & Calibrated Logit Diff (\\%) \\\\",
        "\\midrule",
    ]
    for r in results:
        tex_lines.append(f"{r['Condition']} & {r['Active Heads']} & {r['Calibrated Logit Diff (%)']}\\% \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\caption{Circuit Sufficiency and Ablation Table.}", "\\end{table}"])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
