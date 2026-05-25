"""
Topology factories.

The 9-node network is the main benchmark used throughout the experiments.
Its node parameters were hand-tuned so that every node has a distinct
fault signature in the chosen frequency band; do not edit unless you
re-run the validation experiments.

Additional toy topologies (ring, star, two-node) are provided for unit
tests and for illustrating different propagation patterns in the paper.
"""

from typing import List, Optional, Tuple

import numpy as np

from netfd.systems.components import NodeConfig
from netfd.systems.network import NetworkConfig


# ---------------------------------------------------------------------------
# 9-node benchmark
# ---------------------------------------------------------------------------

BENCHMARK_9NODE_WNS: List[float] = [3.0, 4.0, 5.5, 7.5, 1.5, 2.0, 2.8, 4.2, 9.0]
BENCHMARK_9NODE_ZETAS: List[float] = [0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.15]
BENCHMARK_9NODE_GAINS: List[float] = [1.0] * 9

# Directed edges (0-indexed). Combines series, parallel, feedback, and an
# isolated spoke; designed to be non-trivial to diagnose.
BENCHMARK_9NODE_EDGES: List[Tuple[int, int]] = [
    (0, 1),  # N1 -> N2
    (1, 2),  # N2 -> N3
    (1, 3),  # N2 -> N4
    (1, 6),  # N2 -> N7
    (1, 8),  # N2 -> N9
    (2, 4),  # N3 -> N5
    (3, 4),  # N4 -> N5
    (6, 4),  # N7 -> N5
    (4, 5),  # N5 -> N6
    (5, 7),  # N6 -> N8
    (7, 6),  # N8 -> N7  (feedback)
]


def make_benchmark_9node(edge_weight: float = 1.0,
                         injection_nodes: Optional[List[int]] = None,
                         observation_nodes: Optional[List[int]] = None,
                         ) -> NetworkConfig:
    """Build the 9-node benchmark used throughout the experiments.

    Defaults: injection at N1 only, observation at N2..N9.
    """
    nodes = [
        NodeConfig(name=f"N{i + 1}",
                   wn=BENCHMARK_9NODE_WNS[i],
                   zeta=BENCHMARK_9NODE_ZETAS[i],
                   gain=BENCHMARK_9NODE_GAINS[i])
        for i in range(9)
    ]
    adj = np.zeros((9, 9))
    for (i, j) in BENCHMARK_9NODE_EDGES:
        adj[i, j] = edge_weight

    if injection_nodes is None:
        injection_nodes = [0]
    if observation_nodes is None:
        observation_nodes = list(range(1, 9))

    return NetworkConfig(
        nodes=nodes, adjacency=adj,
        injection_nodes=injection_nodes,
        observation_nodes=observation_nodes,
        name="benchmark_9node",
    )


# ---------------------------------------------------------------------------
# Toy topologies (illustrative)
# ---------------------------------------------------------------------------

def make_ring(n_nodes: int = 6, edge_weight: float = 1.0) -> NetworkConfig:
    """Ring topology: i -> (i + 1) mod n_nodes."""
    nodes = [
        NodeConfig(name=f"N{i}",
                   wn=2.0 + 0.3 * i,
                   zeta=0.10 + 0.02 * i,
                   gain=1.0)
        for i in range(n_nodes)
    ]
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        adj[i, (i + 1) % n_nodes] = edge_weight
    return NetworkConfig(
        nodes=nodes, adjacency=adj,
        injection_nodes=list(range(n_nodes)),
        observation_nodes=list(range(n_nodes)),
        name=f"ring_{n_nodes}",
    )


def make_star(n_nodes: int = 6, edge_weight: float = 1.0) -> NetworkConfig:
    """Star topology: node 0 is the hub, nodes 1..n-1 are leaves, hub -> leaf."""
    nodes = [
        NodeConfig(name=f"N{i}",
                   wn=2.0 + 0.3 * i,
                   zeta=0.10 + 0.02 * i,
                   gain=1.0)
        for i in range(n_nodes)
    ]
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(1, n_nodes):
        adj[0, i] = edge_weight
    return NetworkConfig(
        nodes=nodes, adjacency=adj,
        injection_nodes=[0],
        observation_nodes=list(range(1, n_nodes)),
        name=f"star_{n_nodes}",
    )


def make_two_node(edge_weight: float = 1.0) -> NetworkConfig:
    """Minimal feedback toy: N0 <-> N1. Used for analytical cross-checks."""
    nodes = [
        NodeConfig(name="N0", wn=2.0, zeta=0.10, gain=1.0),
        NodeConfig(name="N1", wn=2.5, zeta=0.12, gain=1.0),
    ]
    adj = np.array([[0.0, edge_weight],
                    [edge_weight, 0.0]])
    return NetworkConfig(
        nodes=nodes, adjacency=adj,
        injection_nodes=[0],
        observation_nodes=[1],
        name="two_node",
    )
