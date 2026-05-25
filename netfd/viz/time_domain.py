"""
Time-domain simulation and plotting for `GlobalSystem`.

`simulate_time_response` is a small ZOH-discretized simulator used both for
plotting and for the offline scenario precomputation in `sequential.env`.
"""

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from netfd.systems.synthesis import GlobalSystem


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_time_response(sys: GlobalSystem,
                           v_signal: np.ndarray,
                           t: np.ndarray) -> np.ndarray:
    """Simulate y(t) = sys * v(t) with zero-order-hold discretization.

    Parameters
    ----------
    sys      : GlobalSystem
    v_signal : (T, nu)
    t        : (T,) uniformly spaced time vector

    Returns
    -------
    y : (T, ny)
    """
    if v_signal.shape != (len(t), sys.nu):
        raise ValueError(f"v_signal shape {v_signal.shape} != ({len(t)}, {sys.nu})")

    dt = float(t[1] - t[0])
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    nx, nu = sys.nx, sys.nu

    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A
    M[:nx, nx:] = B
    Md = expm(M * dt)
    Ad = Md[:nx, :nx]
    Bd = Md[:nx, nx:]

    x = np.zeros(nx)
    y = np.zeros((len(t), sys.ny))
    for k in range(len(t)):
        y[k] = C @ x + D @ v_signal[k]
        x = Ad @ x + Bd @ v_signal[k]
    return y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_time_response(sys_list: List[GlobalSystem], labels: List[str],
                       v_signal: np.ndarray, t: np.ndarray,
                       output_indices: Optional[List[int]] = None,
                       ax=None,
                       title: str = "Time response"):
    """Plot one or more system responses on a shared time axis."""
    if ax is None:
        n_out = sys_list[0].ny if output_indices is None else len(output_indices)
        fig, axes = plt.subplots(n_out, 1, figsize=(8, 1.7 * n_out), sharex=True)
        if n_out == 1:
            axes = [axes]
    else:
        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]

    if output_indices is None:
        output_indices = list(range(sys_list[0].ny))

    colors = plt.cm.tab10(np.linspace(0, 1, len(sys_list)))
    for sys, lbl, c in zip(sys_list, labels, colors):
        y = simulate_time_response(sys, v_signal, t)
        for ax_i, oi in zip(axes, output_indices):
            ax_i.plot(t, y[:, oi], label=lbl, linewidth=1.4, color=c)

    for ax_i, oi in zip(axes, output_indices):
        ax_i.set_ylabel(f"y_{oi}")
        ax_i.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8, loc="upper right")
    return axes
