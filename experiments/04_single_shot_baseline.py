"""
Experiment 04 — Single-shot baseline with full observation
==========================================================

Validation experiment for the PPO main results. Show that the single-shot
baseline they compare against is itself strong: with full observation and
a properly designed multi-sine probe, the freq-weighted residual diagnostic
is essentially perfect at moderate-to-high SNR and degrades gracefully.

Hypothesis set: healthy + node faults (all 9) + edge faults (all 11) = 21.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from experiments._common import (
    make_parser, apply_overrides, banner, section, resolve_path,
)
from netfd.io import load_experiment, ensure_outdir, save_json, parse_omega_grid
from netfd.systems import psi_omega, BENCHMARK_9NODE_EDGES
from netfd.diagnosis import (
    build_mixed_fault_hypotheses,
    find_top_peaks, design_multi_sine_probe, diag_freq_weighted,
)
from netfd.viz import simulate_time_response, plot_confusion_compare


def single_shot_evaluate(sys_hyps_full, healthy_idx, all_obs_nodes,
                         omega, t, snr_db_list, n_mc, rng, probing_cfg):
    """Run single-shot full-obs diagnosis at each SNR with Monte Carlo."""
    H = len(sys_hyps_full)
    sys_full = [s.with_obs(all_obs_nodes) for s in sys_hyps_full]
    P_nom = sys_full[healthy_idx]

    # Peaks for fault hypotheses (vs healthy).
    fault_peak_freqs, fault_peak_psi = [], []
    for k, sysH in enumerate(sys_full):
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

    # Re-align (healthy gets empty).
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

    v = design_multi_sine_probe(
        fault_peak_freqs, fault_peak_psi, t, probing_cfg["amp"],
        inv_detect_floor=probing_cfg["inv_detect_floor"],
    )
    y_pred = [simulate_time_response(s, v, t) for s in sys_full]
    y_nom = simulate_time_response(P_nom, v, t)
    ref_rms = float(np.sqrt(np.mean(y_nom ** 2)))

    confs = {}
    for snr_db in snr_db_list:
        sigma = ref_rms * 10.0 ** (-snr_db / 20.0)
        cm = np.zeros((H, H))
        for true_k in range(H):
            y_clean = y_pred[true_k]
            T, ny = y_clean.shape
            for _ in range(n_mc):
                y_meas = y_clean + rng.standard_normal((T, ny)) * sigma
                d, _ = diag_freq_weighted(
                    y_meas, y_pred, peak_freqs, peak_psi, t,
                    inv_detect_floor=probing_cfg["inv_detect_floor"],
                )
                cm[true_k, d] += 1
        cm /= n_mc
        confs[snr_db] = {"cm": cm, "avg_steps": 1.0}
    return confs, ref_rms


def main():
    parser = make_parser(
        default_config="configs/experiments/04_single_shot_baseline.yaml",
        description="Single-shot full-obs reference for the PPO main results.",
    )
    args = parser.parse_args()

    cfg = load_experiment(str(resolve_path(args.config)))
    apply_overrides(cfg, args.override)

    outdir = ensure_outdir(args.outdir or cfg["outdir"])
    net = cfg["network"]
    omega = parse_omega_grid(cfg["omega_grid"])
    rng = np.random.default_rng(int(cfg.get("seed", 2025)))

    # Resolve hypothesis spec
    hyp = cfg["hypotheses"]
    node_indices = hyp["node_fault"].get("node_indices")
    if node_indices is None:
        node_indices = list(range(net.n_nodes))
    edges = hyp["edge_fault"].get("edges")
    if edges is None:
        edges = list(BENCHMARK_9NODE_EDGES)

    banner("Experiment 04: Single-shot baseline (full obs)",
           lines=[f"network: {net.name}",
                  f"hypotheses: 1 healthy + {len(node_indices)} node + "
                  f"{len(edges)} edge = {1 + len(node_indices) + len(edges)}",
                  f"outdir : {outdir}"])

    section("build hypothesis set")
    sys_hyps_full, labels, healthy_idx = build_mixed_fault_hypotheses(
        cfg=net,
        node_indices=node_indices,
        node_fault_type=hyp["node_fault"]["type"],
        node_severity=float(hyp["node_fault"]["severity"]),
        edges=edges,
        edge_new_weight=float(hyp["edge_fault"]["new_weight"]),
    )
    print(f"  H = {len(sys_hyps_full)} hypotheses, healthy_idx={healthy_idx}")

    section("single-shot diagnosis at each SNR")
    t = np.arange(0.0, cfg["probing"]["duration"],
                  cfg["probing"]["dt"], dtype=np.float32)
    confs, ref_rms = single_shot_evaluate(
        sys_hyps_full=sys_hyps_full, healthy_idx=healthy_idx,
        all_obs_nodes=net.observation_nodes,
        omega=omega, t=t,
        snr_db_list=cfg["noise"]["snr_db_list"],
        n_mc=int(cfg["noise"]["n_mc"]),
        rng=rng, probing_cfg=cfg["probing"],
    )
    print(f"  reference RMS = {ref_rms:.4f}")
    for snr in cfg["noise"]["snr_db_list"]:
        acc = float(np.trace(confs[snr]["cm"]) / len(labels))
        print(f"  SNR={snr:5.1f} dB: acc = {100 * acc:5.1f}%")

    section("save figures and summary")
    plot_confusion_compare(
        confs_per_method=[("single-shot", confs)],
        labels=labels,
        snr_db_list=cfg["noise"]["snr_db_list"],
        outpath=str(outdir / "confusion_singleshot.png"),
        suptitle=("Single-shot full-obs diagnosis: "
                  f"H={len(labels)} hypotheses"),
        figsize_per_panel=(5.5, 5.0),
    )

    summary = {
        "n_hypotheses": len(labels),
        "ref_rms": ref_rms,
        "acc_by_snr": {
            str(snr): float(np.trace(confs[snr]["cm"]) / len(labels))
            for snr in cfg["noise"]["snr_db_list"]
        },
    }
    save_json(summary, str(outdir / "summary.json"))
    np.savez(
        str(outdir / "results.npz"),
        labels=np.asarray(labels),
        snr_db_list=np.asarray(cfg["noise"]["snr_db_list"]),
        cms=np.stack([confs[s]["cm"] for s in cfg["noise"]["snr_db_list"]]),
    )
    print(f"\nAll outputs saved under: {outdir}/\n")


if __name__ == "__main__":
    main()
