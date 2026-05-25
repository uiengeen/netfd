"""
YAML configuration loader.

Two layers of YAML are supported:
  - **network configs** (`configs/networks/*.yaml`): topology + node
    parameters + injection/observation defaults
  - **experiment configs** (`configs/experiments/*.yaml`): everything else,
    optionally referencing a network config via the `network` key

`load_experiment(path)` returns a plain dict; the calling experiment script
extracts the fields it cares about and passes them to library functions.
This keeps the YAML schema flexible and explicit at the call site.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from netfd.systems.components import NodeConfig
from netfd.systems.network import NetworkConfig
from netfd.systems.topologies import make_benchmark_9node


# ---------------------------------------------------------------------------
# Network YAML
# ---------------------------------------------------------------------------

def load_network(path: str) -> NetworkConfig:
    """Load a `NetworkConfig` from a YAML file.

    Two formats are supported.

    1. Preset (recommended for the benchmark):

        preset: benchmark_9node
        edge_weight: 1.0          # optional
        injection_nodes: [0]      # optional override
        observation_nodes: [1, 2, 3, 4, 5, 6, 7, 8]   # optional override

    2. Explicit:

        name: my_network
        nodes:
          - {name: N1, wn: 3.0, zeta: 0.15, gain: 1.0}
          - {name: N2, wn: 4.0, zeta: 0.12, gain: 1.0}
          ...
        edges:                # list of [from, to, weight] (0-indexed)
          - [0, 1, 1.0]
          - [1, 2, 1.0]
        injection_nodes: [0]
        observation_nodes: [1]
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # ---- preset ----
    if "preset" in data:
        preset = data["preset"]
        if preset == "benchmark_9node":
            return make_benchmark_9node(
                edge_weight=float(data.get("edge_weight", 1.0)),
                injection_nodes=data.get("injection_nodes"),
                observation_nodes=data.get("observation_nodes"),
            )
        raise ValueError(f"Unknown network preset: {preset}")

    # ---- explicit ----
    nodes = [NodeConfig(name=n["name"], wn=float(n["wn"]),
                        zeta=float(n["zeta"]), gain=float(n.get("gain", 1.0)))
             for n in data["nodes"]]
    n = len(nodes)
    adj = np.zeros((n, n))
    for entry in data["edges"]:
        i, j, w = entry
        adj[int(i), int(j)] = float(w)

    return NetworkConfig(
        nodes=nodes, adjacency=adj,
        injection_nodes=list(data["injection_nodes"]),
        observation_nodes=list(data["observation_nodes"]),
        name=data.get("name", "network"),
    )


# ---------------------------------------------------------------------------
# Experiment YAML
# ---------------------------------------------------------------------------

def load_experiment(path: str) -> Dict[str, Any]:
    """Load an experiment YAML, resolving the referenced network if any.

    Returns a dict with at least:
      - `network`   : NetworkConfig (resolved from `network_config` path key,
                      or built in-place from `network`)
      - everything else as written
    """
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "network_config" in data:
        net_path = Path(data["network_config"])
        if not net_path.is_absolute():
            net_path = (path.parent / net_path).resolve()
        data["network"] = load_network(str(net_path))
    elif "network" in data and isinstance(data["network"], dict):
        # Inline network definition.
        inline_path = path.parent / "__inline_network__.yaml"
        with open(inline_path, "w") as f:
            yaml.safe_dump(data["network"], f)
        try:
            data["network"] = load_network(str(inline_path))
        finally:
            inline_path.unlink(missing_ok=True)
    else:
        raise ValueError(
            f"{path}: must contain either 'network_config' (path) or "
            f"'network' (inline NetworkConfig dict)."
        )

    return data


# ---------------------------------------------------------------------------
# Helpers for typed extraction
# ---------------------------------------------------------------------------

def parse_edges(edge_list: List[List[int]]) -> List[Tuple[int, int]]:
    """Convert a YAML list-of-lists into a list of (i, j) tuples."""
    return [(int(i), int(j)) for (i, j) in edge_list]


def parse_omega_grid(spec: Dict[str, Any]) -> np.ndarray:
    """Build a frequency grid from a YAML spec.

    Accepted form:
        omega_grid: {low: -2, high: 2, num: 801}     # log-spaced (10**low..10**high)
    """
    return np.logspace(float(spec["low"]), float(spec["high"]),
                       int(spec["num"]))
