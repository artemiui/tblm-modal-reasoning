from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Any, Mapping

PAPER_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.size": 13.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 13.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

MODAL_ROLE_COLORS = {
    "fact_retrieval": "#D55E00",
    "splitting": "#0072B2",
    "transmission": "#009E73",
    "accessibility_filtering": "#CC79A7",
    "modal_operator": "#E69F00",
    "world_accessibility": "#56B4E9",
    "decision": "#F0E442",
    "random": "#999999",
}

MODAL_REGION_COLORS = {
    "facts_region": "#E74C3C",
    "accessibility_region": "#9B59B6",
    "expression_region": "#3498DB",
    "query_region": "#2ECC71",
}


def apply_paper_style(overrides: Mapping[str, Any] | None = None) -> None:
    plt.rcParams.update(PAPER_RC_PARAMS)
    if overrides:
        plt.rcParams.update(dict(overrides))


def stylize_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", which="both", direction="out", length=4.0, width=1.0)


def stylize_colorbar(cbar: Any) -> None:
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=10.5, length=3.0)
