"""
Confusion-matrix and action-usage plots.

`plot_confusion_grid` renders one row of confusion matrices for arbitrary
methods (single-shot, PPO, ...) across an SNR sweep. Several methods can
be stacked by calling it multiple times into a pre-built figure, or by
using `plot_confusion_compare` for the standard 2-row layout.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Single confusion matrix in an axis
# ---------------------------------------------------------------------------

def _draw_one_cm(ax, cm: np.ndarray, labels: Sequence[str],
                 title: str,
                 xlabel: str = "Decided",
                 ylabel: Optional[str] = None,
                 fontsize_title: int = 12,
                 fontsize_ticks: int = 8,
                 fontsize_cell: int = 7,
                 fontsize_axis: int = 11):
    H = len(labels)
    ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xticks(range(H))
    ax.set_yticks(range(H))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize_ticks)
    ax.set_yticklabels(labels, fontsize=fontsize_ticks)
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    for i in range(H):
        for j in range(H):
            if cm[i, j] > 0.02:
                color = "white" if cm[i, j] < 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color=color, fontsize=fontsize_cell)


# ---------------------------------------------------------------------------
# Multi-method × multi-SNR confusion grid
# ---------------------------------------------------------------------------

def plot_confusion_compare(
        confs_per_method: List[Tuple[str, Dict]],
        labels: Sequence[str],
        snr_db_list: Sequence[float],
        outpath: str,
        suptitle: Optional[str] = None,
        figsize_per_panel: Tuple[float, float] = (4.4, 4.3),
        fontsize_title: int = 12,
        fontsize_ticks: int = 8,
        fontsize_cell: int = 7,
        fontsize_axis: int = 11,
        dpi: int = 120,
        ):
    """N-method × M-SNR grid of confusion matrices.

    Parameters
    ----------
    confs_per_method : list of (method_name, conf_dict)
        Each `conf_dict` maps snr_db -> {"cm": (H, H), "avg_steps": float, ...}
    labels : hypothesis labels
    snr_db_list : SNR levels (columns)
    """
    H = len(labels)
    n_rows = len(confs_per_method)
    n_cols = len(snr_db_list)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols,
                 figsize_per_panel[1] * n_rows),
    )
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    for col, snr in enumerate(snr_db_list):
        for row, (tag, conf_set) in enumerate(confs_per_method):
            ax = axes[row, col]
            cm = conf_set[snr]["cm"]
            acc = float(np.trace(cm) / H)
            extra = ""
            if "avg_steps" in conf_set[snr] and conf_set[snr]["avg_steps"] != 1:
                extra = f", k={conf_set[snr]['avg_steps']:.1f}"
            title = f"{tag}, SNR={snr:.0f} dB\nacc={100 * acc:.1f}%{extra}"
            ylabel = f"{tag}\nTrue" if col == 0 else None
            _draw_one_cm(ax, cm, labels, title,
                         ylabel=ylabel,
                         fontsize_title=fontsize_title,
                         fontsize_ticks=fontsize_ticks,
                         fontsize_cell=fontsize_cell,
                         fontsize_axis=fontsize_axis)

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize_title + 2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Action-usage heatmap
# ---------------------------------------------------------------------------

_TEAL_GREY = LinearSegmentedColormap.from_list(
    "tealgrey",
    ["#f5f5f5", "#cfe5e2", "#7fb8b3", "#3a8a86", "#0f4a48"],
)


def plot_action_usage(action_counts: np.ndarray,
                      hyp_labels: Sequence[str],
                      pair_list: Sequence[Tuple[int, int]],
                      outpath: str,
                      title: Optional[str] = None,
                      fontsize_ticks: int = 10,
                      fontsize_cell: int = 9,
                      fontsize_axis: int = 11,
                      figsize: Tuple[float, float] = (8.2, 6.5),
                      dpi: int = 120):
    """Plot row-normalized action usage as a teal-grey heatmap.

    `action_counts[i, a]` = raw count of times PPO took action `a` when
    the true fault was hypothesis `i`. Each row is divided by its max so
    the dominant pair per fault reads as 1.0.
    """
    H = len(hyp_labels)
    M = len(pair_list)

    row_max = action_counts.max(axis=1, keepdims=True)
    row_max = np.where(row_max < 1, 1, row_max)
    ac_norm = action_counts / row_max

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ac_norm, cmap=_TEAL_GREY, aspect="auto", vmin=0, vmax=1)

    pair_labels = [f"N{i + 1}\u2192N{j + 1}" for (i, j) in pair_list]
    ax.set_xticks(range(M))
    ax.set_xticklabels(pair_labels, fontsize=fontsize_ticks)
    ax.set_yticks(range(H))
    ax.set_yticklabels(hyp_labels, fontsize=fontsize_ticks)

    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    ax.set_xlabel("Action: I/O pair selected by policy",
                  fontsize=fontsize_axis, labelpad=10)
    ax.set_ylabel("True fault hypothesis",
                  fontsize=fontsize_axis, labelpad=10)
    if title:
        ax.set_title(title, fontsize=fontsize_axis + 1, pad=14)

    for i in range(H):
        for j in range(M):
            v = ac_norm[i, j]
            if v < 0.05:
                continue
            color = "white" if v > 0.55 else "#1a1a1a"
            txt = f"{v:.2f}" if v < 0.995 else "1.0"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=fontsize_cell,
                    fontweight="bold" if v > 0.4 else "normal")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Relative usage (row-normalized)",
                   fontsize=fontsize_axis, labelpad=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=fontsize_ticks)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
