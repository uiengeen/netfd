"""
Node-level building blocks.

A node is a second-order strictly-proper SISO LTI system:

    x_dot = [[0,     1     ]] x + [[0]] u
            [[-wn^2, -2*z*wn]]     [[k]]
    y     = [[1, 0]] x

`NodeConfig` is the symbolic specification (wn, zeta, gain). `Node` is the
numerical (A, B, C, D) realization built from a `NodeConfig`. Several nodes
combine into a block-diagonal augmented system that the network synthesis
layer closes the loop on.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Symbolic node specification
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    """Specification of one second-order SISO node."""
    name: str
    wn: float           # natural frequency [rad/s]
    zeta: float         # damping ratio [-]
    gain: float = 1.0   # input gain k

    def validate(self) -> None:
        if not self.wn > 0:
            raise ValueError(f"Node {self.name}: wn must be > 0, got {self.wn}")
        if not self.zeta > 0:
            raise ValueError(
                f"Node {self.name}: zeta must be > 0 (stable node), got {self.zeta}"
            )
        if self.gain == 0:
            raise ValueError(f"Node {self.name}: gain must be nonzero")


# ---------------------------------------------------------------------------
# Numerical state-space realization
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Numerical (A, B, C, D) state-space realization of one node."""
    name: str
    A: np.ndarray   # (nx, nx)
    B: np.ndarray   # (nx, nu)
    C: np.ndarray   # (ny, nx)
    D: np.ndarray   # (ny, nu)

    @property
    def nx(self) -> int: return self.A.shape[0]
    @property
    def nu(self) -> int: return self.B.shape[1]
    @property
    def ny(self) -> int: return self.C.shape[0]


def build_node(cfg: NodeConfig) -> Node:
    """Build a `Node` from a `NodeConfig` (controllable canonical form)."""
    cfg.validate()
    wn, z, k = cfg.wn, cfg.zeta, cfg.gain
    A = np.array([[0.0, 1.0],
                  [-wn * wn, -2.0 * z * wn]])
    B = np.array([[0.0], [k]])
    C = np.array([[1.0, 0.0]])
    D = np.zeros((1, 1))
    return Node(name=cfg.name, A=A, B=B, C=C, D=D)


def build_node_library(cfg_list: List[NodeConfig]) -> List[Node]:
    return [build_node(c) for c in cfg_list]


# ---------------------------------------------------------------------------
# Block-diagonal augmentation
# ---------------------------------------------------------------------------

def block_diag_augmented(nodes: List[Node]):
    """Stack all nodes into block-diagonal augmented matrices.

    For n SISO nodes with total state dimension nx_total:
        A_aug : (nx_total, nx_total)
        B_aug : (nx_total, n)       one input column per node
        C_aug : (n,        nx_total) one output row per node
        D_aug : (n, n)               zero (strictly-proper nodes)
    """
    n = len(nodes)
    nx_total = sum(nd.nx for nd in nodes)

    A_aug = np.zeros((nx_total, nx_total))
    B_aug = np.zeros((nx_total, n))
    C_aug = np.zeros((n, nx_total))
    D_aug = np.zeros((n, n))

    row = 0
    for i, nd in enumerate(nodes):
        nx = nd.nx
        A_aug[row:row + nx, row:row + nx] = nd.A
        B_aug[row:row + nx, i:i + 1] = nd.B
        C_aug[i:i + 1, row:row + nx] = nd.C
        row += nx

    return A_aug, B_aug, C_aug, D_aug
