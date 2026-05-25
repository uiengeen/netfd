"""
Topology visualization.

`plot_topology` is the canonical layout-agnostic networkx renderer. For
the paper, a hand-laid-out version of the 9-node benchmark is provided in
`netfd.viz.illustrative.plot_benchmark_9node_layout`.
"""

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from netfd.systems.network import NetworkConfig


def plot_topology(cfg: NetworkConfig,
                  fault_node: Optional[int] = None,
                  injection_nodes: Optional[List[int]] = None,
                  observation_nodes: Optional[List[int]] = None,
                  ax=None,
                  title: Optional[str] = None,
                  seed: int = 42):
    """Plot a network topology with role-based node coloring.

    Color legend
    ------------
        red    : fault node
        purple : injection AND observation
        blue   : injection only
        green  : observation only
        gray   : neither
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Use a directed graph so edge directions are respected. networkx will
    # ignore edge weights in layout if we don't tell it to, but for arrows
    # we'll fall back to undirected layout positions.
    G = nx.DiGraph()
    for i in range(cfg.n_nodes):
        G.add_node(i)
    for i in range(cfg.n_nodes):
        for j in range(cfg.n_nodes):
            if cfg.adjacency[i, j] != 0.0:
                G.add_edge(i, j, weight=float(cfg.adjacency[i, j]))

    pos = nx.spring_layout(G.to_undirected(), seed=seed)

    inj = injection_nodes if injection_nodes is not None else cfg.injection_nodes
    obs = observation_nodes if observation_nodes is not None else cfg.observation_nodes

    node_colors = []
    for i in range(cfg.n_nodes):
        if fault_node is not None and i == fault_node:
            node_colors.append("#e74c3c")
        elif i in inj and i in obs:
            node_colors.append("#9b59b6")
        elif i in inj:
            node_colors.append("#3498db")
        elif i in obs:
            node_colors.append("#2ecc71")
        else:
            node_colors.append("#bdc3c7")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=720, edgecolors="black",
                           linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.4, alpha=0.7,
                           arrows=True, arrowsize=14, ax=ax)
    labels = {i: cfg.nodes[i].name for i in range(cfg.n_nodes)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10,
                            font_weight="bold", ax=ax)

    edge_labels = {(i, j): f"{cfg.adjacency[i, j]:.2f}"
                   for i in range(cfg.n_nodes) for j in range(cfg.n_nodes)
                   if cfg.adjacency[i, j] != 0.0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax)

    ax.set_title(title or f"Topology: {cfg.name}")
    ax.set_axis_off()
    return ax


def plot_adjacency_matrix(adjacency: np.ndarray,
                          node_names: Optional[List[str]] = None,
                          ax=None,
                          title: str = "Adjacency matrix"):
    """Heatmap of the adjacency matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    n = adjacency.shape[0]
    if node_names is None:
        node_names = [f"N{i + 1}" for i in range(n)]

    im = ax.imshow(adjacency, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(node_names, fontsize=9)
    ax.set_yticklabels(node_names, fontsize=9)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if adjacency[i, j] != 0:
                ax.text(j, i, f"{adjacency[i, j]:.1f}",
                        ha="center", va="center",
                        color="white" if adjacency[i, j] > 0.5 else "black",
                        fontsize=8)
    plt.colorbar(im, ax=ax)
    return ax
