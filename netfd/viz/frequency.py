"""
Frequency-domain plots: Bode, pointwise psi(omega), identifiability heatmap.
"""

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from netfd.systems.synthesis import GlobalSystem
from netfd.systems.nu_gap import psi_omega


# ---------------------------------------------------------------------------
# MIMO Bode (max singular value vs frequency)
# ---------------------------------------------------------------------------

def plot_bode(sys_list: List[GlobalSystem], labels: List[str],
              omega: Optional[np.ndarray] = None,
              ax=None,
              title: str = "Bode (max singular value)"):
    if omega is None:
        omega = np.logspace(-2, 2, 401)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    for sys, lbl in zip(sys_list, labels):
        H = sys.freqresp(omega)
        sv_max = np.array([np.linalg.svd(H[k], compute_uv=False)[0]
                           for k in range(len(omega))])
        ax.semilogx(omega, 20 * np.log10(sv_max + 1e-20), label=lbl, linewidth=1.6)

    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel("Max singular value [dB]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Pointwise chordal distance psi(omega)
# ---------------------------------------------------------------------------

def plot_psi_omega(sys0: GlobalSystem, sys_list: List[GlobalSystem],
                   labels: List[str],
                   omega: Optional[np.ndarray] = None,
                   ax=None,
                   title: str = r"Pointwise chordal distance $\psi(\omega)$"):
    if omega is None:
        omega = np.logspace(-2, 2, 401)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    for sys, lbl in zip(sys_list, labels):
        psi = psi_omega(sys0, sys, omega)
        sup = float(np.max(psi))
        idx = int(np.argmax(psi))
        ax.semilogx(omega, psi,
                    label=f"{lbl}  (sup={sup:.3f} @ {omega[idx]:.2f})",
                    linewidth=1.6)
        ax.axvline(omega[idx], color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel(r"$\psi(\omega)$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# ν-gap identifiability heatmap
# ---------------------------------------------------------------------------

def plot_identifiability_matrix(matrix: np.ndarray,
                                x_labels: List[str], y_labels: List[str],
                                ax=None,
                                title: str = "ν-gap identifiability matrix",
                                xlabel: str = "Excitation node",
                                ylabel: str = "Fault node"):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] < 0.5 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, label=r"$\delta_\nu$")
    return ax
