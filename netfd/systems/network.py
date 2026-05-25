"""
Network-level configuration: topology, injection/observation sets, and fault
specifications.

A `NetworkConfig` is fully self-describing: nodes + adjacency + injection
indices + observation indices. A `FaultSpec` is a single-node parametric
perturbation that produces a new `NodeConfig`; the network synthesis layer
applies it by re-synthesizing the closed-loop system with one node replaced.
"""

from dataclasses import dataclass
from typing import List, Literal

import numpy as np

from netfd.systems.components import NodeConfig


# ---------------------------------------------------------------------------
# Fault specification
# ---------------------------------------------------------------------------

@dataclass
class FaultSpec:
    """A single-node parametric fault.

    Two supported types of multiplicative perturbation:
      - stiffness:  wn^2_new = (1 - severity) * wn^2_old
      - damping:    zeta_new = (1 + severity) * zeta_old
    """
    node_index: int
    fault_type: Literal["stiffness", "damping"]
    severity: float
    label: str = ""

    def apply(self, node: NodeConfig) -> NodeConfig:
        new = NodeConfig(name=node.name + "_f", wn=node.wn,
                         zeta=node.zeta, gain=node.gain)
        if self.fault_type == "stiffness":
            new.wn = node.wn * np.sqrt(max(1.0 - self.severity, 1e-6))
        elif self.fault_type == "damping":
            new.zeta = node.zeta * (1.0 + self.severity)
        else:
            raise ValueError(f"Unknown fault type: {self.fault_type}")
        new.validate()
        return new


# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    nodes: List[NodeConfig]
    adjacency: np.ndarray            # (n, n), weighted; entry [i, j] = edge i -> j
    injection_nodes: List[int]       # node indices receiving external excitation
    observation_nodes: List[int]     # node indices whose outputs are measured
    name: str = "network"

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def validate(self) -> None:
        n = self.n_nodes
        if self.adjacency.shape != (n, n):
            raise ValueError(
                f"adjacency shape {self.adjacency.shape} != ({n}, {n})"
            )
        for idx in self.injection_nodes:
            if not 0 <= idx < n:
                raise ValueError(f"injection index {idx} out of range")
        for idx in self.observation_nodes:
            if not 0 <= idx < n:
                raise ValueError(f"observation index {idx} out of range")
        for node in self.nodes:
            node.validate()
