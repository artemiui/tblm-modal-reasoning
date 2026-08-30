from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence
import matplotlib.pyplot as plt
import numpy as np
from ..plot_style import apply_paper_style, stylize_axis, MODAL_REGION_COLORS


def plot_stage_bar_chart(
    data: Dict[str, Dict[str, float]],
    error_bars: Dict[str, Dict[str, float]],
    output_png: Path,
    title: str = "Mean |dLD| by Token Category across Layer Bands",
    ylabel: str = "Mean |dLD| (SEM)",
) -> None:
    """
    Produce grouped bar chart across Early / Middle / Late layer groups with SEM error bars.
    """
    apply_paper_style()
    categories = list(data.keys())
    stages = ["early", "middle", "late"]
    stage_labels = ["Early", "Middle", "Late"]
    stage_colors = ["#3498DB", "#9B59B6", "#E67E22"]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12.5, 6.5))

    for idx, (st, label, col) in enumerate(zip(stages, stage_labels, stage_colors)):
        means = [data[cat].get(st, 0.0) for cat in categories]
        sems = [error_bars[cat].get(st, 0.0) for cat in categories]
        offset = (idx - 1) * width
        ax.bar(x + offset, means, width, yerr=sems, label=label, color=col, capsize=4, edgecolor="black", alpha=0.85)

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.legend(frameon=True)
    stylize_axis(ax)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
