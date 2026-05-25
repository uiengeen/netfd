"""
Experiment 02 — Structural blindness
====================================

Argument 1: in single-shot diagnosis, the set of node faults that can be
correctly identified is bounded by

    reachable(fault) ∩ obs_set != empty.

Faults whose downstream reachable set does not intersect the observed
nodes are STRUCTURALLY BLIND: no probe design and no diagnostic algorithm
can recover their identity. We make this visible by sweeping the
observation set from a single node up to all of N2..N9 and showing that
accuracy grows monotonically.

Diagnosis is ideal (noise-free) here so the effect is purely structural.
"""

import sys
from pathlib import Path
from dataclasses import replace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from experiments._common import (
    make_parser, apply_overrides, banner, section, resolve_path,
)
from netfd.io import load_experiment, ensure_outdir, save_json, parse_omega_grid
from netfd.systems import NetworkConfig, synthesize, psi_omega
from netfd.diagnosis import (
    build_node_fault_hypotheses, find_top_peaks,
    design_multi_sine_probe, diag_freq_weighted,
)
from netfd.viz import simulate_time_response


def _override_observation(net: NetworkConfig, obs_indices) -> NetworkConfig:
    """Return a copy of the network with a different observation set."""
    return NetworkConfig(
        nodes=net.nodes,
        adjacency=net.adjacency,
        injection_nodes=net.injection_nodes,
        observation_nodes=list(obs_indices),
        name=net.name + f"_obs{list(obs_indices)}",
    )


def run_one_obs_set(net: NetworkConfig, obs_indices, fault_cfg, omega,
                    probing_cfg):
    """Run noise-free freq-weighted diagnosis for one observation set."""
    cfg_obs = _override_observation(net, obs_indices)
    sys_hyps, labels, healthy_idx = build_node_fault_hypotheses(
        cfg_obs,
        fault_type=fault_cfg["type"],
        severity=float(fault_cfg["severity"]),
    )
    sys_hyps = [s.with_obs(obs_indices) for s in sys_hyps]
    H = len(sys_hyps)
    P_nom = sys_hyps[healthy_idx]

    # Peaks for fault hypotheses (healthy has none).
    fault_peak_freqs, fault_peak_psi = [], []
    for k, sysH in enumerate(sys_hyps):
        if k == healthy_idx:
            continue
        psi = psi_omega(P_nom, sysH, omega)
        f, p = find_top_peaks(
            psi, omega,
            top_k=probing_cfg["top_k_peaks"],
            prominence=probing_cfg["peak_prominence"],
            rel_thresh=probing_cfg["peak_relative_thresh"],
        )
        fault_peak_freqs.append(f)
        fault_peak_psi.append(p)

    # Re-align (healthy gets empty arrays).
    peak_freqs, peak_psi = [], []
    it = iter(zip(fault_peak_freqs, fault_peak_psi))
    for k in range(H):
        if k == healthy_idx:
            peak_freqs.append(np.array([]))
            peak_psi.append(np.array([]))
        else:
            f, p = next(it)
            peak_freqs.append(f)
            peak_psi.append(p)
    sup_psi = np.array([float(np.max(p)) if len(p) > 0 else 0.0
                        for p in peak_psi])

    # Multi-sine probe (fault hypotheses only).
    t = np.arange(0.0, probing_cfg["duration"], probing_cfg["dt"])
    v = design_multi_sine_probe(
        fault_peak_freqs, fault_peak_psi, t, probing_cfg["amp"],
        inv_detect_floor=probing_cfg["inv_detect_floor"],
    )

    # Clean predictions, ideal classification.
    y_pred = [simulate_time_response(s, v, t) for s in sys_hyps]
    conf = np.zeros((H, H))
    for true_k in range(H):
        d, _ = diag_freq_weighted(y_pred[true_k], y_pred,
                                  peak_freqs, peak_psi, t,
                                  inv_detect_floor=probing_cfg["inv_detect_floor"])
        conf[true_k, d] += 1
    acc = float(np.trace(conf) / H)

    return {
        "labels": labels,
        "acc": acc,
        "conf": conf,
        "sup_psi": sup_psi,
        "healthy_idx": healthy_idx,
    }


def _plot_blindness_grid(results, sweep, outpath):
    """Plot a 1 x N grid of confusion matrices, one per obs set."""
    n_cols = len(sweep)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 5.0))
    if n_cols == 1:
        axes = [axes]
    for ax, item in zip(axes, sweep):
        info = results[item["name"]]
        cm = info["conf"]
        H = len(info["labels"])
        ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{item['name']}\nacc = {100 * info['acc']:.1f}%",
                     fontsize=11)
        ax.set_xticks(range(H))
        ax.set_yticks(range(H))
        ax.set_xticklabels(info["labels"], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(info["labels"], fontsize=7)
        ax.set_xlabel("Decided")
        ax.set_ylabel("True")
        for i in range(H):
            for j in range(H):
                if cm[i, j] > 0.02:
                    color = "white" if cm[i, j] < 0.5 else "black"
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            color=color, fontsize=6)
    fig.suptitle("Structural blindness: accuracy vs observation coverage",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = make_parser(
        default_config="configs/experiments/02_blindness.yaml",
        description="Sweep observation sets and show the structural blindness effect.",
    )
    args = parser.parse_args()

    cfg = load_experiment(str(resolve_path(args.config)))
    apply_overrides(cfg, args.override)

    outdir = ensure_outdir(args.outdir or cfg["outdir"])
    net = cfg["network"]
    omega = parse_omega_grid(cfg["omega_grid"])

    banner("Experiment 02: Structural blindness",
           lines=[f"network: {net.name}",
                  f"fault  : {cfg['fault']['type']} sev={cfg['fault']['severity']}",
                  f"sweep  : {len(cfg['observation_sweep'])} observation sets",
                  f"outdir : {outdir}"])

    results = {}
    for item in cfg["observation_sweep"]:
        section(f"obs set: {item['name']} -> {item['indices']}")
        info = run_one_obs_set(
            net=net,
            obs_indices=item["indices"],
            fault_cfg=cfg["fault"],
            omega=omega,
            probing_cfg=cfg["probing"],
        )
        print(f"  ideal freq-weighted accuracy: {100 * info['acc']:.1f}%")
        results[item["name"]] = info

    _plot_blindness_grid(results, cfg["observation_sweep"],
                         outpath=outdir / "blindness_sweep.png")

    # Summary
    summary = {
        item["name"]: {
            "obs_indices": item["indices"],
            "acc": results[item["name"]]["acc"],
        }
        for item in cfg["observation_sweep"]
    }
    save_json(summary, str(outdir / "summary.json"))

    print(f"\nAll outputs saved under: {outdir}/\n")


if __name__ == "__main__":
    main()
