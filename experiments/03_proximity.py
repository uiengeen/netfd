"""
Experiment 03 — Intrinsic proximity via symmetric edge faults
=============================================================

Argument 2: when two edges are topologically symmetric AND their adjacent
nodes have near-identical dynamics, single-shot diagnosis confuses the
symmetric pair regardless of probe design.

Uses a variant of the benchmark network in which N3 and N4 have nearly
identical parameters, so the symmetric edge pairs collide.
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
from netfd.systems import synthesize, psi_omega
from netfd.diagnosis import (
    build_edge_fault_hypotheses,
    get_peaks_for_hypotheses,
    design_multi_sine_probe, diag_freq_weighted,
)
from netfd.viz import simulate_time_response


def find_symmetric_indices(edge_tags, symmetric_pairs):
    """Map declared symmetric edge pairs to hypothesis-index pairs."""
    edge_to_idx = {tuple(e): i for i, e in enumerate(edge_tags) if e is not None}
    sym_idx_pairs = []
    sym_set = set()
    for (a, b) in symmetric_pairs:
        a, b = tuple(a), tuple(b)
        if a in edge_to_idx and b in edge_to_idx:
            sym_idx_pairs.append((edge_to_idx[a], edge_to_idx[b]))
            sym_set.update([edge_to_idx[a], edge_to_idx[b]])
    return sym_idx_pairs, sym_set


def plot_proximity_grid(conf_dict, labels, sym_idx_set, sym_idx_pairs,
                        outpath, dpi=150):
    """Plot ideal + per-SNR confusion matrices with red boxes on symmetric pairs."""
    n = len(conf_dict)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5))
    if n == 1:
        axes = [axes]
    for ax, (key, cm) in zip(axes, conf_dict.items()):
        H = cm.shape[0]
        ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        acc = float(np.trace(cm) / H)
        ax.set_title(f"{key}\nacc={100 * acc:.1f}%", fontsize=22)
        ax.set_xticks(range(H))
        ax.set_yticks(range(H))
        x_labels = [f"{lbl}*" if i in sym_idx_set else lbl
                    for i, lbl in enumerate(labels)]
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(x_labels, fontsize=12)
        ax.set_xlabel("Decided", fontsize=22)
        ax.set_ylabel("True", fontsize=22)
        for (a, b) in sym_idx_pairs:
            ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=1.8))
            ax.add_patch(plt.Rectangle((a - 0.5, b - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=1.8))
        for i in range(H):
            for j in range(H):
                if cm[i, j] > 0.02:
                    color = "white" if cm[i, j] < 0.5 else "black"
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            color=color, fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_one_mode(net, mode_name, new_weight, omega, probing_cfg,
                 noise_cfg, symmetric_pairs, rng, outdir):
    section(f"mode '{mode_name}' (edge weight -> {new_weight})")

    sysN = synthesize(net)
    if not sysN.is_stable():
        raise RuntimeError("nominal unstable")
    P_nom = sysN.with_obs(net.observation_nodes)

    # Edge fault hypotheses (each at observation_nodes).
    sys_hyps_full, labels, healthy_idx, edge_tags = build_edge_fault_hypotheses(
        net, edges=[e for e in _benchmark_edges(net)], new_weight=new_weight,
    )
    sys_hyps = [s.with_obs(net.observation_nodes) for s in sys_hyps_full]
    H = len(sys_hyps)
    sym_idx_pairs, sym_idx_set = find_symmetric_indices(edge_tags, symmetric_pairs)

    print(f"  {H} hypotheses (1 healthy + {H - 1} edge faults)")
    print(f"  symmetric pairs (red-boxed): "
          f"{[(labels[a], labels[b]) for (a, b) in sym_idx_pairs]}")

    # Peaks from fault hypotheses only.
    fault_systems = [sys_hyps[k] for k in range(H) if k != healthy_idx]
    fault_peak_freqs, fault_peak_psi = get_peaks_for_hypotheses(
        P_nom, fault_systems, omega,
        top_k=probing_cfg["top_k_peaks"],
        prominence=probing_cfg["peak_prominence"],
        rel_thresh=probing_cfg["peak_relative_thresh"],
    )

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

    # Symmetric-pair sup psi diagnostic check.
    print("  symmetric-pair sup psi:")
    for (a, b) in sym_idx_pairs:
        s = float(np.max(psi_omega(sys_hyps[a], sys_hyps[b], omega)))
        print(f"    {labels[a]} vs {labels[b]}: {s:.5f}")

    # Multi-sine probe.
    t = np.arange(0.0, probing_cfg["duration"], probing_cfg["dt"])
    v = design_multi_sine_probe(
        fault_peak_freqs, fault_peak_psi, t, probing_cfg["amp"],
        inv_detect_floor=probing_cfg["inv_detect_floor"],
    )

    # Predictions.
    y_pred = [simulate_time_response(s, v, t) for s in sys_hyps]
    y_nom = simulate_time_response(P_nom, v, t)
    ref_rms = float(np.sqrt(np.mean(y_nom ** 2)))

    # Ideal classification.
    conf_ideal = np.zeros((H, H))
    for true_k in range(H):
        d, _ = diag_freq_weighted(y_pred[true_k], y_pred,
                                  peak_freqs, peak_psi, t,
                                  inv_detect_floor=probing_cfg["inv_detect_floor"])
        conf_ideal[true_k, d] += 1
    acc_ideal = float(np.trace(conf_ideal) / H)
    print(f"  IDEAL acc = {100 * acc_ideal:.1f}%")

    # Noisy MC at each SNR.
    conf_noisy = {}
    acc_by_snr = []
    for snr in noise_cfg["snr_db_list"]:
        cm = np.zeros((H, H))
        sigma = ref_rms * 10.0 ** (-snr / 20.0)
        for true_k in range(H):
            y_clean = y_pred[true_k]
            T, ny = y_clean.shape
            for _ in range(noise_cfg["n_mc"]):
                noise = rng.standard_normal((T, ny)) * sigma
                y_meas = y_clean + noise
                d, _ = diag_freq_weighted(y_meas, y_pred,
                                          peak_freqs, peak_psi, t,
                                          inv_detect_floor=probing_cfg["inv_detect_floor"])
                cm[true_k, d] += 1
        cm /= noise_cfg["n_mc"]
        conf_noisy[f"SNR={snr:.0f} dB"] = cm
        acc_by_snr.append(float(np.trace(cm) / H))
        print(f"  SNR={snr:5.1f} dB acc = {100 * acc_by_snr[-1]:5.1f}%")

    cd = {"IDEAL": conf_ideal}
    cd.update(conf_noisy)
    plot_proximity_grid(
        cd, labels, sym_idx_set, sym_idx_pairs,
        outpath=outdir / f"{mode_name}_confusion.png",
    )

    return {
        "labels": labels,
        "acc_ideal": acc_ideal,
        "acc_by_snr": acc_by_snr,
        "sym_idx_pairs": sym_idx_pairs,
    }


def _benchmark_edges(net):
    """Recover the list of directed edges from the adjacency matrix."""
    n = net.adjacency.shape[0]
    return [(i, j) for i in range(n) for j in range(n)
            if net.adjacency[i, j] != 0]


def main():
    parser = make_parser(
        default_config="configs/experiments/03_proximity.yaml",
        description="Show intrinsic proximity from symmetric edge faults.",
    )
    args = parser.parse_args()

    cfg = load_experiment(str(resolve_path(args.config)))
    apply_overrides(cfg, args.override)

    outdir = ensure_outdir(args.outdir or cfg["outdir"])
    net = cfg["network"]
    omega = parse_omega_grid(cfg["omega_grid"])
    rng = np.random.default_rng(int(cfg.get("seed", 2025)))

    banner("Experiment 03: Intrinsic proximity",
           lines=[f"network: {net.name}",
                  f"modes  : {list(cfg['modes'].keys())}",
                  f"outdir : {outdir}"])

    results = {}
    for mode_name, new_weight in cfg["modes"].items():
        results[mode_name] = run_one_mode(
            net=net, mode_name=mode_name, new_weight=float(new_weight),
            omega=omega, probing_cfg=cfg["probing"], noise_cfg=cfg["noise"],
            symmetric_pairs=cfg["symmetric_pairs"], rng=rng, outdir=outdir,
        )

    summary = {
        mode: {
            "acc_ideal": info["acc_ideal"],
            "acc_by_snr": dict(zip(
                [str(s) for s in cfg["noise"]["snr_db_list"]],
                info["acc_by_snr"],
            )),
        }
        for mode, info in results.items()
    }
    save_json(summary, str(outdir / "summary.json"))

    print(f"\nAll outputs saved under: {outdir}/\n")


if __name__ == "__main__":
    main()
