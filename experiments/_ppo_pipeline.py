"""
Shared pipeline for the three PPO experiments (05/06/07).

Each PPO experiment differs only in:
  - the hypothesis set (node / edge / mixed faults)
  - PPO hyper-parameters (already in the YAML)
  - reward / episode parameters (already in the YAML)

So the orchestration is identical and lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from netfd.systems import (
    NetworkConfig, psi_omega, BENCHMARK_9NODE_EDGES,
)
from netfd.diagnosis import (
    build_node_fault_hypotheses,
    build_edge_fault_hypotheses,
    build_mixed_fault_hypotheses,
    find_top_peaks, design_multi_sine_probe, diag_freq_weighted,
)
from netfd.sequential import (
    AFDEnv, AFDEnvConfig, PairScenario,
    build_pair_scenarios, precompute_nu_gap_matrix,
    PPOConfig, train, PPOPolicy, evaluate_ppo,
)
from netfd.viz import (
    simulate_time_response,
    plot_confusion_compare, plot_action_usage, plot_training_curves,
)
from netfd.io import save_json


# =============================================================================
# Hypothesis-set dispatch
# =============================================================================

def build_hypotheses_from_yaml(net: NetworkConfig, hyp_cfg: Dict
                               ) -> Tuple[List, List[str], int]:
    """Dispatch to the right hypothesis builder based on `hyp_cfg['kind']`."""
    kind = hyp_cfg["kind"]
    if kind == "node":
        return build_node_fault_hypotheses(
            cfg=net,
            fault_type=hyp_cfg["fault_type"],
            severity=float(hyp_cfg["severity"]),
            node_indices=hyp_cfg.get("node_indices"),
        )
    if kind == "edge":
        edges = hyp_cfg.get("edges") or list(BENCHMARK_9NODE_EDGES)
        edges = [tuple(e) for e in edges]
        systems, labels, healthy_idx, _edges = build_edge_fault_hypotheses(
            cfg=net, edges=edges,
            new_weight=float(hyp_cfg["new_weight"]),
        )
        return systems, labels, healthy_idx
    if kind == "mixed":
        node_idx = hyp_cfg.get("node_indices")
        if node_idx is None:
            node_idx = list(range(net.n_nodes))
        edges = hyp_cfg.get("edges") or list(BENCHMARK_9NODE_EDGES)
        edges = [tuple(e) for e in edges]
        return build_mixed_fault_hypotheses(
            cfg=net,
            node_indices=node_idx,
            node_fault_type=hyp_cfg["node_fault_type"],
            node_severity=float(hyp_cfg["node_severity"]),
            edges=edges,
            edge_new_weight=float(hyp_cfg["edge_new_weight"]),
        )
    raise ValueError(f"Unknown hypothesis kind: {kind}")


# =============================================================================
# Single-shot baseline (matched probe, full obs)
# =============================================================================

def single_shot_baseline(sys_hyps_full, healthy_idx, all_obs_nodes,
                         omega, t, snr_db_list, n_mc, rng, probing_cfg):
    """Reference single-shot diagnostic against PPO."""
    H = len(sys_hyps_full)
    sys_full = [s.with_obs(all_obs_nodes) for s in sys_hyps_full]
    P_nom = sys_full[healthy_idx]

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


# =============================================================================
# Main pipeline
# =============================================================================

@dataclass
class PpoPipelineOutputs:
    labels: List[str]
    pair_list: List[Tuple[int, int]]
    nu_gap_matrix: np.ndarray
    single_shot_confs: Dict
    ppo_confs: Dict
    log: Dict


def run_ppo_pipeline(net: NetworkConfig,
                     hyp_cfg: Dict,
                     pair_cfg: Dict,
                     omega: np.ndarray,
                     probing_cfg: Dict,
                     noise_cfg: Dict,
                     reward_cfg: Dict,
                     ppo_cfg_dict: Dict,
                     outdir: Path,
                     rng_seed: int = 2025,
                     suptitle: str = "") -> PpoPipelineOutputs:
    """End-to-end orchestration shared by exp 05/06/07."""
    rng = np.random.default_rng(rng_seed)

    # ---- 1) Hypotheses ----
    sys_hyps_full, labels, healthy_idx = build_hypotheses_from_yaml(net, hyp_cfg)
    H = len(sys_hyps_full)
    print(f"[1] H = {H} hypotheses: {labels}")

    # ---- 2) ν-gap matrix (full obs, used as reward feature) ----
    print("[2] ν-gap matrix (full obs)...")
    sys_full = [s.with_obs(net.observation_nodes) for s in sys_hyps_full]
    D = precompute_nu_gap_matrix(sys_full, omega)
    print(f"    max off-diagonal = {D[~np.eye(H, dtype=bool)].max():.3f}")

    # ---- 3) Pair scenarios ----
    pair_list = [(int(pair_cfg["injection"]), int(o))
                 for o in pair_cfg["observations"]]
    print(f"[3] Building {len(pair_list)} pair scenarios...")
    pair_scenarios = build_pair_scenarios(
        sys_hyps_full=sys_hyps_full,
        healthy_idx=healthy_idx,
        pair_list=pair_list,
        omega_grid=omega,
        probe_duration=float(probing_cfg["duration"]),
        sim_dt=float(probing_cfg["dt"]),
        probe_amp=float(probing_cfg["amp"]),
        top_k_peaks=int(probing_cfg["top_k_peaks"]),
        peak_prominence=float(probing_cfg["peak_prominence"]),
        peak_rel_thresh=float(probing_cfg["peak_relative_thresh"]),
        inv_detect_floor=float(probing_cfg["inv_detect_floor"]),
        verbose=True,
    )
    ref_rms_per_pair = np.array([s.y_norm[healthy_idx] for s in pair_scenarios])

    # ---- 4) Single-shot baseline ----
    print(f"[4] Single-shot baseline ({noise_cfg['n_mc_eval']} MC)...")
    t = np.arange(0.0, probing_cfg["duration"], probing_cfg["dt"], dtype=np.float32)
    ss_confs, ref_rms_ss = single_shot_baseline(
        sys_hyps_full=sys_hyps_full, healthy_idx=healthy_idx,
        all_obs_nodes=net.observation_nodes,
        omega=omega, t=t,
        snr_db_list=noise_cfg["snr_db_eval"],
        n_mc=int(noise_cfg["n_mc_eval"]),
        rng=rng, probing_cfg=probing_cfg,
    )
    for snr in noise_cfg["snr_db_eval"]:
        acc = float(np.trace(ss_confs[snr]["cm"]) / H)
        print(f"    SNR={snr:5.1f} dB: single-shot acc = {100 * acc:5.1f}%")

    # ---- 5) PPO training ----
    mean_ref_rms = float(np.mean(ref_rms_per_pair))
    env_cfg = AFDEnvConfig(
        K_max=int(reward_cfg["K_max"]),
        belief_threshold=float(reward_cfg["belief_threshold"]),
        alpha_info=float(reward_cfg["alpha_info"]),
        beta_amb=float(reward_cfg["beta_amb"]),
        step_cost=float(reward_cfg["step_cost"]),
        R_succ=float(reward_cfg["R_succ"]),
        R_fail=float(reward_cfg["R_fail"]),
        probe_duration=float(probing_cfg["duration"]),
        sim_dt=float(probing_cfg["dt"]),
        probe_amp=float(probing_cfg["amp"]),
        sigma_meas=0.1,
        likelihood_temp=1.0,
        randomize_snr=True,
        snr_db_low=float(noise_cfg["snr_db_train_range"][0]),
        snr_db_high=float(noise_cfg["snr_db_train_range"][1]),
        ref_rms_for_snr=mean_ref_rms,
    )

    def env_factory(seed):
        # Copy cfg so per-env SNR sampling does not race.
        from dataclasses import replace
        env = AFDEnv(pair_scenarios, D, replace(env_cfg))
        env.reset(seed=seed)
        return env

    print(f"[5] PPO training "
          f"(SNR randomized in {noise_cfg['snr_db_train_range']} dB):")
    ppo_cfg = PPOConfig(
        total_timesteps=int(ppo_cfg_dict["total_timesteps"]),
        num_envs=int(ppo_cfg_dict["num_envs"]),
        num_steps=int(ppo_cfg_dict["num_steps"]),
        learning_rate=float(ppo_cfg_dict["learning_rate"]),
        ent_coef=float(ppo_cfg_dict["ent_coef"]),
        hidden=int(ppo_cfg_dict["hidden"]),
        seed=int(ppo_cfg_dict["seed"]),
        verbose=True,
    )
    agent, log = train(env_factory, ppo_cfg)

    # ---- 6) PPO evaluation ----
    print(f"[6] PPO evaluation ({noise_cfg['n_mc_eval']} MC per (fault, SNR))...")
    policy = PPOPolicy(agent, deterministic=True)
    ppo_confs = evaluate_ppo(
        policy=policy,
        pair_scenarios=pair_scenarios,
        nu_gap_matrix=D,
        env_cfg=env_cfg,
        ref_rms_per_pair=ref_rms_per_pair,
        snr_db_list=noise_cfg["snr_db_eval"],
        n_mc=int(noise_cfg["n_mc_eval"]),
        base_seed=12345,
    )
    for snr in noise_cfg["snr_db_eval"]:
        acc = float(np.trace(ppo_confs[snr]["cm"]) / H)
        k_avg = ppo_confs[snr]["avg_steps"]
        print(f"    SNR={snr:5.1f} dB: PPO acc = {100 * acc:5.1f}% "
              f"#probes = {k_avg:.2f}")

    # ---- 7) Figures ----
    print("[7] Saving figures and results...")
    plot_confusion_compare(
        confs_per_method=[("single-shot", ss_confs), ("PPO", ppo_confs)],
        labels=labels,
        snr_db_list=noise_cfg["snr_db_eval"],
        outpath=str(outdir / "confusion_compare.png"),
        suptitle=suptitle,
        figsize_per_panel=(5.0, 5.0),
        fontsize_title=12, fontsize_ticks=8, fontsize_cell=7,
    )
    plot_training_curves(
        log, outpath=str(outdir / "ppo_training_curves.png"),
        title="PPO training curves",
    )
    snr_lowest = noise_cfg["snr_db_eval"][-1]
    plot_action_usage(
        ppo_confs[snr_lowest]["action_counts"],
        hyp_labels=labels,
        pair_list=pair_list,
        outpath=str(outdir / "action_usage.png"),
        title=f"Learned view-switching strategy "
              f"(PPO @ SNR={snr_lowest:.0f} dB)",
    )

    # ---- 8) Summaries ----
    summary = {
        "n_hypotheses": H,
        "pair_list": [[int(i), int(j)] for (i, j) in pair_list],
        "snr_db_eval": list(noise_cfg["snr_db_eval"]),
        "ppo_total_steps": ppo_cfg.total_timesteps,
        "results": [
            {
                "snr_db": float(snr),
                "acc_single_shot": float(np.trace(ss_confs[snr]["cm"]) / H),
                "acc_ppo": float(np.trace(ppo_confs[snr]["cm"]) / H),
                "avg_probes": float(ppo_confs[snr]["avg_steps"]),
            }
            for snr in noise_cfg["snr_db_eval"]
        ],
    }
    save_json(summary, str(outdir / "summary.json"))

    np.savez(
        str(outdir / "results.npz"),
        labels=np.asarray(labels),
        pair_list=np.asarray(pair_list),
        nu_gap=D,
        snr_db_list=np.asarray(noise_cfg["snr_db_eval"]),
        ss_cms=np.stack([ss_confs[s]["cm"] for s in noise_cfg["snr_db_eval"]]),
        ppo_cms=np.stack([ppo_confs[s]["cm"] for s in noise_cfg["snr_db_eval"]]),
        log_step=np.asarray(log["step"]),
        log_ep_accuracy=np.asarray(log["ep_accuracy"]),
        log_ep_return=np.asarray(log["ep_return"]),
        log_ep_final_entropy=np.asarray(log["ep_final_entropy"]),
        log_policy_loss=np.asarray(log["policy_loss"]),
        log_value_loss=np.asarray(log["value_loss"]),
        log_entropy=np.asarray(log["entropy"]),
    )

    return PpoPipelineOutputs(
        labels=labels, pair_list=pair_list, nu_gap_matrix=D,
        single_shot_confs=ss_confs, ppo_confs=ppo_confs, log=log,
    )
