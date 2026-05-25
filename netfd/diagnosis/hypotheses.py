"""
Hypothesis-set builders.

A "hypothesis set" is the ordered list of `GlobalSystem` instances the
diagnostic distinguishes between. By convention `healthy` is always included
and listed first: an unobservable fault produces a response equal to the
nominal, so the diagnostic should classify it as healthy rather than mis-
assigning it to some fault hypothesis.

Three flavors:
  - node faults  : healthy + per-node parametric fault
  - edge faults  : healthy + per-edge weight perturbation
  - mixed faults : healthy + selected node faults + selected edge faults
"""

from typing import List, Optional, Sequence, Tuple

from netfd.systems.network import NetworkConfig, FaultSpec
from netfd.systems.synthesis import (
    GlobalSystem, synthesize, synthesize_faulty, synthesize_edge_faulty,
)


# ---------------------------------------------------------------------------
# Node fault hypotheses
# ---------------------------------------------------------------------------

def build_node_fault_hypotheses(cfg: NetworkConfig,
                                fault_type: str,
                                severity: float,
                                node_indices: Optional[Sequence[int]] = None,
                                ) -> Tuple[List[GlobalSystem], List[str], int]:
    """Build healthy + per-node parametric fault hypotheses.

    If `node_indices` is None, faults are applied to all nodes in order.

    Returns
    -------
    systems     : list of GlobalSystem (healthy first)
    labels      : list of str
    healthy_idx : index of the healthy hypothesis (always 0)
    """
    if node_indices is None:
        node_indices = list(range(cfg.n_nodes))

    systems = [synthesize(cfg)]
    labels = ["healthy"]

    for k in node_indices:
        fs = FaultSpec(node_index=k, fault_type=fault_type, severity=severity)
        sysf = synthesize_faulty(cfg, fs)
        if not sysf.is_stable():
            print(f"  [warn] unstable node fault N{k + 1}, skipping")
            continue
        systems.append(sysf)
        labels.append(f"N{k + 1}")

    return systems, labels, 0


# ---------------------------------------------------------------------------
# Edge fault hypotheses
# ---------------------------------------------------------------------------

def build_edge_fault_hypotheses(cfg: NetworkConfig,
                                edges: Sequence[Tuple[int, int]],
                                new_weight: float,
                                ) -> Tuple[List[GlobalSystem],
                                           List[str],
                                           int,
                                           List[Optional[Tuple[int, int]]]]:
    """Build healthy + per-edge weight perturbation hypotheses.

    Returns
    -------
    systems     : list of GlobalSystem (healthy first)
    labels      : list of str
    healthy_idx : 0
    edge_tags   : parallel list of (i, j) tuples, None for the healthy entry
    """
    systems = [synthesize(cfg)]
    labels = ["healthy"]
    edge_tags: List[Optional[Tuple[int, int]]] = [None]

    for (i, j) in edges:
        sysf = synthesize_edge_faulty(cfg, (i, j), new_weight)
        if not sysf.is_stable():
            print(f"  [warn] unstable edge fault E{i + 1}-{j + 1}={new_weight}, "
                  f"skipping")
            continue
        systems.append(sysf)
        labels.append(f"E{i + 1}-{j + 1}")
        edge_tags.append((i, j))

    return systems, labels, 0, edge_tags


# ---------------------------------------------------------------------------
# Mixed (node + edge) fault hypotheses
# ---------------------------------------------------------------------------

def build_mixed_fault_hypotheses(cfg: NetworkConfig,
                                 node_indices: Sequence[int],
                                 node_fault_type: str,
                                 node_severity: float,
                                 edges: Sequence[Tuple[int, int]],
                                 edge_new_weight: float,
                                 ) -> Tuple[List[GlobalSystem], List[str], int]:
    """Build healthy + selected node faults + selected edge faults.

    Returns
    -------
    systems     : list of GlobalSystem (healthy first, then nodes, then edges)
    labels      : list of str
    healthy_idx : 0
    """
    systems = [synthesize(cfg)]
    labels = ["healthy"]

    for k in node_indices:
        fs = FaultSpec(node_index=k,
                       fault_type=node_fault_type, severity=node_severity)
        sysf = synthesize_faulty(cfg, fs)
        if not sysf.is_stable():
            print(f"  [warn] unstable node fault N{k + 1}, skipping")
            continue
        systems.append(sysf)
        labels.append(f"N{k + 1}")

    for (i, j) in edges:
        sysf = synthesize_edge_faulty(cfg, (i, j), edge_new_weight)
        if not sysf.is_stable():
            print(f"  [warn] unstable edge fault E{i + 1}-{j + 1}, skipping")
            continue
        systems.append(sysf)
        labels.append(f"E{i + 1}-{j + 1}")

    return systems, labels, 0
