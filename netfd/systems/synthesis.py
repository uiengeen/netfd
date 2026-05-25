"""
Closed-loop synthesis of the global networked system P_global : v -> y_obs.

Given a `NetworkConfig`, build:
  - block-diagonal augmented (A_aug, B_aug, C_aug, D_aug) over all node states
  - linking matrix L from the adjacency (using transposed convention so that
    L[i, j] = weight of edge j -> i)
  - injection matrix M selecting which node inputs receive external v
  - observation matrix C_obs selecting which node outputs are measured

The interconnection is
    u_aug = L y_aug + M v
    y_obs = C_obs y_aug
With D_aug = 0 (strictly-proper nodes) this is loop-free and collapses to
    A_cl = A_aug + B_aug @ L @ C_aug
    B_cl = B_aug @ M
    C_cl = C_obs @ C_aug
    D_cl = 0

Faults are applied by replacing one node's `NodeConfig` and re-synthesizing.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from netfd.systems.components import (
    Node, NodeConfig, build_node_library, block_diag_augmented,
)
from netfd.systems.network import NetworkConfig, FaultSpec


# ---------------------------------------------------------------------------
# Global closed-loop system
# ---------------------------------------------------------------------------

@dataclass
class GlobalSystem:
    """Closed-loop state-space realization of the networked system."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    C_full: np.ndarray   # full per-node output matrix (used by `with_obs`)
    name: str = ""

    @property
    def nx(self) -> int: return self.A.shape[0]
    @property
    def nu(self) -> int: return self.B.shape[1]
    @property
    def ny(self) -> int: return self.C.shape[0]

    def is_stable(self, tol: float = 1e-9) -> bool:
        return bool(np.all(np.real(np.linalg.eigvals(self.A)) < -tol))

    def freqresp(self, omega: np.ndarray) -> np.ndarray:
        """Frequency response P(j*omega) on a grid. Shape (Nw, ny, nu) complex."""
        ny, nu, nx = self.ny, self.nu, self.nx
        out = np.zeros((len(omega), ny, nu), dtype=complex)
        I = np.eye(nx)
        for k, w in enumerate(omega):
            try:
                inv = np.linalg.solve(1j * w * I - self.A, self.B)
                out[k] = self.C @ inv + self.D
            except np.linalg.LinAlgError:
                out[k] = np.full((ny, nu), np.inf)
        return out

    def with_obs(self, obs_indices: List[int]) -> "GlobalSystem":
        """Return a new `GlobalSystem` with a different observation selection.

        Re-uses (A, B) and the cached full C; only C and D are recomputed.
        """
        C_new = self.C_full[obs_indices, :]
        D_new = np.zeros((len(obs_indices), self.B.shape[1]))
        return GlobalSystem(
            A=self.A, B=self.B, C=C_new, D=D_new, C_full=self.C_full,
            name=self.name + f"_obs{obs_indices}",
        )


# ---------------------------------------------------------------------------
# Interconnection matrices
# ---------------------------------------------------------------------------

def build_linking_matrix(adjacency: np.ndarray) -> np.ndarray:
    """For SISO nodes, L equals the transposed adjacency.

    Convention: u_i = sum_j L[i, j] * y_j, where adjacency[j, i] is the
    weight of edge j -> i, so L = adjacency.T.
    """
    return np.array(adjacency, dtype=float).T.copy()


def build_injection_matrix(n_nodes: int, injection_indices: List[int]) -> np.ndarray:
    """M maps external excitation v (size n_v) into u_aug (size n_nodes)."""
    n_v = len(injection_indices)
    M = np.zeros((n_nodes, n_v))
    for col, idx in enumerate(injection_indices):
        M[idx, col] = 1.0
    return M


def build_observation_matrix(n_nodes: int, observation_indices: List[int]) -> np.ndarray:
    """C_obs selects rows of y_aug that are actually measured."""
    n_y = len(observation_indices)
    C_obs = np.zeros((n_y, n_nodes))
    for row, idx in enumerate(observation_indices):
        C_obs[row, idx] = 1.0
    return C_obs


def check_well_posedness(L: np.ndarray, D_aug: np.ndarray,
                         tol: float = 1e-10) -> Tuple[bool, float]:
    """Check det(I - L @ D_aug) != 0. With D_aug = 0 this is trivial."""
    n = L.shape[0]
    M_check = np.eye(n) - L @ D_aug
    det = float(np.linalg.det(M_check))
    return abs(det) > tol, det


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(cfg: NetworkConfig,
               nodes_override: Optional[List[Node]] = None,
               name_suffix: str = "") -> GlobalSystem:
    """Synthesize the closed-loop system from a `NetworkConfig`.

    `nodes_override` lets the caller replace per-node realizations (used by
    `synthesize_faulty` to inject faults without rebuilding the config).
    """
    cfg.validate()
    nodes = nodes_override if nodes_override is not None \
        else build_node_library(cfg.nodes)
    if len(nodes) != cfg.n_nodes:
        raise ValueError(
            f"nodes count {len(nodes)} != cfg.n_nodes {cfg.n_nodes}"
        )

    A_aug, B_aug, C_aug, D_aug = block_diag_augmented(nodes)
    L = build_linking_matrix(cfg.adjacency)
    M = build_injection_matrix(cfg.n_nodes, cfg.injection_nodes)
    C_obs = build_observation_matrix(cfg.n_nodes, cfg.observation_nodes)

    ok, det = check_well_posedness(L, D_aug)
    if not ok:
        raise RuntimeError(f"Algebraic loop: det(I - L D_aug) = {det}")

    A_cl = A_aug + B_aug @ L @ C_aug
    B_cl = B_aug @ M
    C_cl = C_obs @ C_aug
    D_cl = np.zeros((C_cl.shape[0], B_cl.shape[1]))

    sys = GlobalSystem(A=A_cl, B=B_cl, C=C_cl, D=D_cl, C_full=C_aug,
                       name=cfg.name + name_suffix)

    if not sys.is_stable():
        max_re = float(np.max(np.real(np.linalg.eigvals(sys.A))))
        print(f"  [warning] synthesized {sys.name} not strictly stable, "
              f"max Re(eig) = {max_re:.4f}")
    return sys


def synthesize_faulty(cfg: NetworkConfig, fault: FaultSpec) -> GlobalSystem:
    """Apply `fault` to the specified node and re-synthesize the network."""
    new_cfgs = list(cfg.nodes)
    new_cfgs[fault.node_index] = fault.apply(cfg.nodes[fault.node_index])
    nodes_f = build_node_library(new_cfgs)
    suffix = (f"_faulty[{cfg.nodes[fault.node_index].name}:"
              f"{fault.fault_type}:{fault.severity}]")
    return synthesize(cfg, nodes_override=nodes_f, name_suffix=suffix)


def synthesize_edge_faulty(cfg: NetworkConfig, edge: Tuple[int, int],
                           new_weight: float) -> GlobalSystem:
    """Replace one edge weight in the adjacency and re-synthesize.

    Used for topology (edge) faults.
    """
    i, j = edge
    adj_new = cfg.adjacency.copy()
    adj_new[i, j] = new_weight
    cfg_new = NetworkConfig(
        nodes=cfg.nodes, adjacency=adj_new,
        injection_nodes=cfg.injection_nodes,
        observation_nodes=cfg.observation_nodes,
        name=cfg.name + f"_edge{i}-{j}={new_weight}",
    )
    return synthesize(cfg_new)
