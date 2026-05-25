"""
Experiment 01 — Topology illustration
=====================================

Render the canonical paper figures:
  - 9-node benchmark layout (plain and annotated with sub-structures)
  - adjacency matrix heatmap
  - three fault-class schematic diagrams (dynamics / signal / topology)
    on a 6-node mesh

All figures are static and depend only on the network definition; no
hypothesis set, simulation, or training is needed.
"""

import sys
from pathlib import Path

# Project-root path bootstrap (lets the script run via `python -m experiments.01_...`
# or `python experiments/01_...py` without `pip install -e .`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from experiments._common import (
    make_parser, apply_overrides, banner, section, resolve_path,
)
from netfd.io import load_experiment, ensure_outdir
from netfd.viz import (
    plot_benchmark_9node_layout, plot_adjacency_matrix,
    plot_dynamics_fault, plot_signal_fault, plot_topology_fault,
)


def main():
    parser = make_parser(
        default_config="configs/experiments/01_topology_illustration.yaml",
        description="Render benchmark topology and fault-class illustrations.",
    )
    args = parser.parse_args()

    cfg = load_experiment(str(resolve_path(args.config)))
    apply_overrides(cfg, args.override)

    outdir = ensure_outdir(args.outdir or cfg["outdir"])
    dpi = int(cfg.get("dpi", 160))
    net = cfg["network"]

    banner("Experiment 01: Topology illustration",
           lines=[f"network: {net.name} (n={net.n_nodes})",
                  f"outdir : {outdir}"])

    section("benchmark 9-node, plain layout")
    plot_benchmark_9node_layout(
        save_path=str(outdir / "benchmark_topology.png"),
        annotated=False, dpi=dpi,
    )

    section("benchmark 9-node, annotated layout")
    plot_benchmark_9node_layout(
        save_path=str(outdir / "benchmark_topology_annotated.png"),
        annotated=True, dpi=dpi,
    )

    section("adjacency matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_adjacency_matrix(
        net.adjacency,
        node_names=[n.name for n in net.nodes],
        ax=ax, title="Adjacency matrix (benchmark 9-node)",
    )
    fig.tight_layout()
    fig.savefig(outdir / "adjacency_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    section("fault-class schematic diagrams")
    for name, plot_fn in [
        ("fault_dynamics", plot_dynamics_fault),
        ("fault_signal",   plot_signal_fault),
        ("fault_topology", plot_topology_fault),
    ]:
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_fn(ax)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {outdir / (name + '.png')}")

    print(f"\nAll outputs saved under: {outdir}/\n")


if __name__ == "__main__":
    main()
