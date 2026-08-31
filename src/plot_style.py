from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_style() -> None:
    """Apply publication-quality matplotlib styling."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 360,
        "savefig.dpi": 360,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# Colour palette for modal axioms
AXIOM_COLORS = {
    "axiom_K": "#1f77b4",
    "axiom_T": "#ff7f0e",
    "axiom_D": "#2ca02c",
    "axiom_4": "#d62728",
    "axiom_5": "#9467bd",
    "axiom_B": "#8c564b",
    "modal_duality_box": "#e377c2",
    "modal_duality_diamond": "#e377c2",
    "modal_modus_ponens": "#7f7f7f",
    "necessitation": "#bcbd22",
    "box_and_distribution": "#17becf",
    "diamond_or_distribution": "#17becf",
}

# Colour palette for propositional rules
PROP_RULE_COLORS = {
    "identity": "#1f77b4",
    "domination": "#ff7f0e",
    "idempotent": "#2ca02c",
    "double_negation": "#d62728",
    "excluded_middle": "#9467bd",
    "contradiction": "#8c564b",
    "commutative": "#e377c2",
    "associative": "#7f7f7f",
    "distributive": "#bcbd22",
    "demorgan": "#17becf",
    "absorption": "#aec7e8",
}

MODEL_COLORS = {
    "Qwen3-0.6B": "#1f77b4",
    "Qwen3-1.7B": "#ff7f0e",
    "Qwen3-4B": "#2ca02c",
    "Qwen3-8B": "#d62728",
}
