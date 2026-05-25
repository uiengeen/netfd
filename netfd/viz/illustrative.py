"""
Paper-quality illustrative figures.

`plot_benchmark_9node_layout`
    Hand-laid-out version of the 9-node benchmark, highlighting the four
    sub-structures (series/parallel hub, feedback loop, isolated spoke).
    Used as the canonical topology figure.

`plot_dynamics_fault`, `plot_signal_fault`, `plot_topology_fault`
    Three schematic fault-class diagrams drawn on a 6-node fully-connected
    mesh; used to explain what a node-parameter fault, an actuator/sensor
    fault, and an edge-weight fault each look like physically.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx


# =============================================================================
# 9-node benchmark, hand-crafted layout
# =============================================================================

# Coordinates chosen so the main series runs horizontally, parallel branches
# fan vertically, the feedback loop sits below, and the isolated spoke
# drops downward from N2.
BENCHMARK_9NODE_LAYOUT = {
    "N1": (0.0,  2.0),
    "N2": (1.5,  2.0),
    "N3": (3.0,  3.2),   # upper parallel branch
    "N4": (3.0,  2.0),   # middle parallel branch
    "N7": (3.0,  0.8),   # lower parallel branch (receives feedback)
    "N5": (4.5,  2.0),   # merge
    "N6": (6.0,  2.0),   # main output
    "N8": (5.25, -0.4),  # feedback node
    "N9": (1.5, -1.0),   # isolated spoke
}


def _draw_arrow(ax, src, dst, *, color="#333333", lw=1.6, rad=0.0, shrink=14):
    arrow = FancyArrowPatch(
        posA=src, posB=dst, arrowstyle="->",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw,
        shrinkA=shrink, shrinkB=shrink,
        mutation_scale=18, zorder=1,
    )
    ax.add_patch(arrow)


def _draw_node(ax, pos, label, *,
               face="#A0CBE2", edge="#1f3a5f",
               text_color="#0b1a2b", radius=0.28,
               fontsize=20):
    circ = plt.Circle(pos, radius, facecolor=face, edgecolor=edge,
                      linewidth=2.0, zorder=3)
    ax.add_patch(circ)
    ax.text(pos[0], pos[1], label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)


def plot_benchmark_9node_layout(save_path: Optional[str] = None,
                                annotated: bool = False,
                                ax=None,
                                dpi: int = 160):
    """Plot the 9-node benchmark with a hand-crafted layout.

    Parameters
    ----------
    save_path : path to save; if None, do not save.
    annotated : if True, add colored region annotations for the four
                substructures (parallel hub, feedback loop, bypass branch).
    ax        : pre-existing axis to draw into; if None, create one.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(11, 7))

    ax.set_xlim(-0.8, 7.0)
    ax.set_ylim(-2.2, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    if annotated:
        ax.add_patch(FancyBboxPatch(
            (1.15, 0.3), 3.7, 3.4,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.4, linestyle="--",
            edgecolor="#2a7a2a", facecolor="#eaffea", alpha=0.5, zorder=0,
        ))
        ax.text(3.0, 3.95,
                "Hybrid Parallel-Series: Nodes 2, 3, 4, 5, 7",
                ha="center", fontsize=22, color="#2a7a2a", style="italic")
        ax.add_patch(FancyBboxPatch(
            (4.7, -0.85), 1.1, 0.9,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.4, linestyle="--",
            edgecolor="#b35c00", facecolor="#fff3e3", alpha=0.5, zorder=0,
        ))
        ax.text(5.25, -1.25, "Feedback Loop: Node 8",
                ha="center", fontsize=22, color="#b35c00", style="italic")
        ax.add_patch(FancyBboxPatch(
            (1.15, -1.45), 0.7, 0.9,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.4, linestyle="--",
            edgecolor="#7a2a7a", facecolor="#f6e7ff", alpha=0.5, zorder=0,
        ))
        ax.text(1.5, -1.85, "Bypass Branch: Node 9",
                ha="center", fontsize=22, color="#7a2a7a", style="italic")

    # Edges first, nodes on top
    lo = BENCHMARK_9NODE_LAYOUT
    _draw_arrow(ax, lo["N1"], lo["N2"])
    _draw_arrow(ax, lo["N2"], lo["N3"], rad=0.2)
    _draw_arrow(ax, lo["N2"], lo["N4"])
    _draw_arrow(ax, lo["N2"], lo["N7"], rad=-0.2)
    _draw_arrow(ax, lo["N3"], lo["N5"], rad=-0.2)
    _draw_arrow(ax, lo["N4"], lo["N5"])
    _draw_arrow(ax, lo["N7"], lo["N5"], rad=0.2)
    _draw_arrow(ax, lo["N5"], lo["N6"])
    _draw_arrow(ax, lo["N6"], lo["N8"], rad=-0.35, color="#b35c00")
    _draw_arrow(ax, lo["N8"], lo["N7"], rad=-0.35, color="#b35c00")
    _draw_arrow(ax, lo["N2"], lo["N9"], color="#7a2a7a")

    for name, pos in lo.items():
        _draw_node(ax, pos, name)

    if save_path and own_fig:
        plt.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return ax


# =============================================================================
# Three fault-class schematic diagrams on a 6-node mesh
# =============================================================================

_MESH_N = 6
_MESH_COLOR_NORMAL = "#A0CBE2"
_MESH_COLOR_FAULT = "#d45721"
_MESH_COLOR_NODE_BORDER = "black"
_MESH_COLOR_EDGE_NORMAL = "#888888"
_MESH_COLOR_EDGE_FAULT = "#d95319"

_MESH_SIZE_NORMAL = 800
_MESH_SIZE_FAULT = 2000
_MESH_WIDTH_NORMAL = 1.5
_MESH_WIDTH_FAULT = 4.5

_MESH_FS_LABEL = 11
_MESH_FS_ANNOT = 22


def _build_mesh():
    G = nx.DiGraph()
    G.add_nodes_from(range(_MESH_N))
    for i in range(_MESH_N):
        for j in range(_MESH_N):
            if i != j:
                G.add_edge(i, j)
    return G


def _mesh_layout():
    angles = np.linspace(0, 2 * np.pi, _MESH_N, endpoint=False)
    return {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}


def _draw_mesh_base(ax, G, pos, node_colors, node_sizes,
                    fault_edges=None,
                    fault_node_for_annot=None, annot_text=None,
                    annot_offset=(0.45, 0.30)):
    U = G.to_undirected()
    fault_set = [set(f) for f in (fault_edges or [])]
    normal_edges = [e for e in U.edges() if set(e) not in fault_set]
    hi_edges = [e for e in U.edges() if set(e) in fault_set]

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           edgecolors=_MESH_COLOR_NODE_BORDER, linewidths=1.5)
    nx.draw_networkx_edges(U, pos, ax=ax,
                           edgelist=normal_edges,
                           width=_MESH_WIDTH_NORMAL,
                           edge_color=_MESH_COLOR_EDGE_NORMAL, arrows=False)
    if hi_edges:
        nx.draw_networkx_edges(U, pos, ax=ax,
                               edgelist=hi_edges,
                               width=_MESH_WIDTH_FAULT,
                               edge_color=_MESH_COLOR_EDGE_FAULT, arrows=False)
    nx.draw_networkx_labels(
        G, pos, {i: str(i) for i in range(_MESH_N)}, ax=ax,
        font_size=_MESH_FS_LABEL, font_weight="bold",
    )
    if annot_text is not None and fault_node_for_annot is not None:
        fx, fy = pos[fault_node_for_annot]
        ox, oy = annot_offset
        ax.text(fx + ox, fy + oy, annot_text,
                fontsize=_MESH_FS_ANNOT, ha="left", va="center")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_dynamics_fault(ax,
                        fault_node: int = 2,
                        annot: str = r"$\Sigma_i(\omega_i^{\rm f},\, \zeta_i^{\rm f})$"):
    """Node-parameter (dynamics) fault: one node enlarged + colored."""
    G, pos = _build_mesh(), _mesh_layout()
    node_colors = [_MESH_COLOR_FAULT if i == fault_node else _MESH_COLOR_NORMAL
                   for i in range(_MESH_N)]
    node_sizes = [_MESH_SIZE_FAULT if i == fault_node else _MESH_SIZE_NORMAL
                  for i in range(_MESH_N)]
    _draw_mesh_base(ax, G, pos, node_colors, node_sizes,
                    fault_node_for_annot=fault_node, annot_text=annot,
                    annot_offset=(0.2, 0.15))


def plot_signal_fault(ax,
                      fault_node: int = 3,
                      annot: str = r"$\Sigma_i(\kappa_i^{\rm f},\, \gamma_i^{\rm f})$"):
    """Actuator/sensor (signal) fault: one node and all its edges highlighted."""
    G, pos = _build_mesh(), _mesh_layout()
    node_colors = [_MESH_COLOR_FAULT if i == fault_node else _MESH_COLOR_NORMAL
                   for i in range(_MESH_N)]
    node_sizes = [_MESH_SIZE_FAULT if i == fault_node else _MESH_SIZE_NORMAL
                  for i in range(_MESH_N)]
    fault_edges = [(fault_node, j) for j in range(_MESH_N) if j != fault_node]
    _draw_mesh_base(ax, G, pos, node_colors, node_sizes,
                    fault_edges=fault_edges,
                    fault_node_for_annot=fault_node, annot_text=annot,
                    annot_offset=(0.42, 0.1))


def plot_topology_fault(ax,
                        fault_node_a: int = 0,
                        fault_node_b: int = 1,
                        annot: str = r"$L^{\rm faulty}$"):
    """Edge (topology) fault: highlight one edge between two adjacent nodes."""
    G, pos = _build_mesh(), _mesh_layout()
    node_colors = [_MESH_COLOR_NORMAL for _ in range(_MESH_N)]
    node_sizes = [_MESH_SIZE_FAULT if i in (fault_node_a, fault_node_b)
                  else _MESH_SIZE_NORMAL for i in range(_MESH_N)]
    fault_edges = [(fault_node_a, fault_node_b)]

    pa, pb = pos[fault_node_a], pos[fault_node_b]
    mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
    norm = np.sqrt(mx ** 2 + my ** 2) + 1e-9
    ox, oy = mx / norm * 0.25, my / norm * 0.25

    _draw_mesh_base(ax, G, pos, node_colors, node_sizes,
                    fault_edges=fault_edges)
    ax.text(mx + ox, my + oy, annot,
            fontsize=_MESH_FS_ANNOT + 2, ha="center", va="center")
