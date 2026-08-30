from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from ..plot_style import apply_paper_style, MODAL_ROLE_COLORS, stylize_axis


def render_circuit_diagram(
    families: Dict[str, List[Tuple[int, int]]],
    output_png: Path,
) -> None:
    """
    Render publication-quality circuit diagram (mirroring Hong et al. Figure 3)
    showing information routing through QRLH, MOH, WAH, FPH, QRMH, and DH.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    family_order = ["QRLH", "MOH", "WAH", "FPH", "QRMH", "DH"]
    family_labels = {
        "QRLH": "Queried-Rule\nLocating (QRLH)",
        "MOH": "Modal-Operator\nHeads (MOH)",
        "WAH": "World-Accessibility\nHeads (WAH)",
        "FPH": "Fact-Processing\nHeads (FPH)",
        "QRMH": "Queried-Rule\nMover (QRMH)",
        "DH": "Decision\nHeads (DH)",
    }
    family_colors = {
        "QRLH": "#0072B2",
        "MOH": "#E69F00",
        "WAH": "#56B4E9",
        "FPH": "#D55E00",
        "QRMH": "#009E73",
        "DH": "#F0E442",
    }

    positions = {
        "QRLH": (0.15, 0.75),
        "MOH": (0.15, 0.25),
        "WAH": (0.45, 0.25),
        "FPH": (0.45, 0.75),
        "QRMH": (0.75, 0.75),
        "DH": (0.85, 0.35),
    }

    for fam in family_order:
        x, y = positions[fam]
        heads = families.get(fam, [])
        head_str = ", ".join(f"L{l}H{h}" for l, h in heads[:3]) if heads else "None"
        if len(heads) > 3:
            head_str += f" (+{len(heads)-3})"

        bbox_props = dict(boxstyle="round,pad=0.6", fc=family_colors[fam], ec="black", lw=1.5, alpha=0.85)
        ax.text(
            x, y, f"{family_labels[fam]}\n[{head_str}]",
            ha="center", va="center", bbox=bbox_props, fontsize=11.5, weight="bold", color="black"
        )

    # Draw information flow arrows
    arrow_props = dict(arrowstyle="->", lw=2.0, color="#333333", shrinkA=15, shrinkB=15)
    ax.annotate("", xy=positions["WAH"], xytext=positions["MOH"], arrowprops=arrow_props)
    ax.annotate("", xy=positions["FPH"], xytext=positions["QRLH"], arrowprops=arrow_props)
    ax.annotate("", xy=positions["QRMH"], xytext=positions["FPH"], arrowprops=arrow_props)
    ax.annotate("", xy=positions["QRMH"], xytext=positions["WAH"], arrowprops=arrow_props)
    ax.annotate("", xy=positions["DH"], xytext=positions["QRMH"], arrowprops=arrow_props)
    ax.annotate("", xy=positions["DH"], xytext=positions["WAH"], arrowprops=arrow_props)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Modal Logic Circuit Architecture", fontsize=15.0, weight="bold", pad=15)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
