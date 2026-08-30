from __future__ import annotations

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
from ..plot_style import apply_paper_style, stylize_axis, stylize_colorbar


def plot_layer_head_heatmap(
    matrix: np.ndarray,
    output_png: Path,
    title: str = "Attention Head Indirect Effect Heatmap",
    xlabel: str = "Head Index",
    ylabel: str = "Layer Index",
) -> None:
    """Plot 2D (n_layers, n_heads) heatmap."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    vmax = float(np.percentile(np.abs(matrix), 99)) if matrix.size > 0 else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", origin="lower", vmin=0, vmax=max(vmax, 1e-4))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    stylize_axis(ax)

    cbar = fig.colorbar(im, ax=ax)
    stylize_colorbar(cbar)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
