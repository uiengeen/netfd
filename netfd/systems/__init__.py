"""Control-theoretic modeling layer: nodes, networks, synthesis, ν-gap."""

from netfd.systems.components import (
    NodeConfig, Node, build_node, build_node_library, block_diag_augmented,
)
from netfd.systems.network import NetworkConfig, FaultSpec
from netfd.systems.synthesis import (
    GlobalSystem, synthesize, synthesize_faulty, synthesize_edge_faulty,
)
from netfd.systems.nu_gap import psi_omega, nu_gap, winding_number_ok
from netfd.systems.topologies import (
    make_benchmark_9node, make_ring, make_star, make_two_node,
    BENCHMARK_9NODE_EDGES, BENCHMARK_9NODE_WNS,
    BENCHMARK_9NODE_ZETAS, BENCHMARK_9NODE_GAINS,
)

__all__ = [
    "NodeConfig", "Node", "build_node", "build_node_library",
    "block_diag_augmented",
    "NetworkConfig", "FaultSpec",
    "GlobalSystem", "synthesize", "synthesize_faulty", "synthesize_edge_faulty",
    "psi_omega", "nu_gap", "winding_number_ok",
    "make_benchmark_9node", "make_ring", "make_star", "make_two_node",
    "BENCHMARK_9NODE_EDGES", "BENCHMARK_9NODE_WNS",
    "BENCHMARK_9NODE_ZETAS", "BENCHMARK_9NODE_GAINS",
]
