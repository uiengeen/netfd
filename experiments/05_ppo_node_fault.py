"""
Experiment 05 — Pair-switching PPO for NODE faults
==================================================

Main result 1. PPO with single-channel observation per step matches the
single-shot baseline that observes ALL N2..N9 simultaneously.

Hypothesis set: healthy + 9 stiffness faults on each node.
Action space: 8 (injection, observation) pairs (N1 -> N2..N9).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import (
    make_parser, apply_overrides, banner, resolve_path,
)
from experiments._ppo_pipeline import run_ppo_pipeline
from netfd.io import load_experiment, ensure_outdir, parse_omega_grid


def main():
    parser = make_parser(
        default_config="configs/experiments/05_ppo_node_fault.yaml",
        description="PPO vs single-shot for node faults (main result 1).",
    )
    args = parser.parse_args()
    cfg = load_experiment(str(resolve_path(args.config)))
    apply_overrides(cfg, args.override)

    outdir = ensure_outdir(args.outdir or cfg["outdir"])
    net = cfg["network"]
    omega = parse_omega_grid(cfg["omega_grid"])

    banner("Experiment 05: PPO node-fault diagnosis",
           lines=[f"network: {net.name}",
                  f"hypothesis kind: {cfg['hypotheses']['kind']}",
                  f"PPO budget: {cfg['ppo']['total_timesteps']:,} env steps",
                  f"outdir: {outdir}"])

    run_ppo_pipeline(
        net=net,
        hyp_cfg=cfg["hypotheses"],
        pair_cfg=cfg["pair_list"],
        omega=omega,
        probing_cfg=cfg["probing"],
        noise_cfg=cfg["noise"],
        reward_cfg=cfg["reward"],
        ppo_cfg_dict=cfg["ppo"],
        outdir=outdir,
        rng_seed=int(cfg["ppo"]["seed"]),
        suptitle="Node faults: sequential PPO vs single-shot full-obs",
    )
    print(f"\nAll outputs saved under: {outdir}/\n")


if __name__ == "__main__":
    main()
