"""
Experiment 06 — Pair-switching PPO for EDGE (topology) faults
=============================================================

Main result 2. Same action space as exp 05, but the hypothesis set is now
11 edge faults instead of node faults.

Hypothesis set: healthy + 11 edge faults (edge weight 1.0 -> new_weight).
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
        default_config="configs/experiments/06_ppo_edge_fault.yaml",
        description="PPO vs single-shot for edge faults (main result 2).",
    )
    args = parser.parse_args()
    cfg = load_experiment(str(resolve_path(args.config)))
    apply_overrides(cfg, args.override)

    outdir = ensure_outdir(args.outdir or cfg["outdir"])
    net = cfg["network"]
    omega = parse_omega_grid(cfg["omega_grid"])

    banner("Experiment 06: PPO edge-fault diagnosis",
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
        suptitle="Edge faults: sequential PPO vs single-shot full-obs",
    )
    print(f"\nAll outputs saved under: {outdir}/\n")


if __name__ == "__main__":
    main()
