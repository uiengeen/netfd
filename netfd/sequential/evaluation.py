"""
PPO policy evaluation on the AFD environment.

Runs Monte-Carlo episodes at fixed SNR levels and aggregates a confusion
matrix, average episode length, and per-fault action usage.
"""

from dataclasses import asdict, replace
from typing import Dict, List, Sequence

import numpy as np

from netfd.sequential.env import AFDEnv, AFDEnvConfig, PairScenario
from netfd.sequential.ppo import PPOPolicy


def evaluate_ppo(policy: PPOPolicy,
                 pair_scenarios: List[PairScenario],
                 nu_gap_matrix: np.ndarray,
                 env_cfg: AFDEnvConfig,
                 ref_rms_per_pair: np.ndarray,
                 snr_db_list: Sequence[float],
                 n_mc: int,
                 base_seed: int = 12345,
                 ) -> Dict[float, Dict]:
    """Evaluate a PPO policy at each SNR (with fixed sigma per SNR).

    Returns
    -------
    out : { snr_db -> {"cm": (H, H), "avg_steps": float,
                        "action_counts": (H, M)} }
    """
    M = len(pair_scenarios)
    H = pair_scenarios[0].y_clean.shape[0]
    out: Dict[float, Dict] = {}

    mean_ref_rms = float(np.mean(ref_rms_per_pair))

    for snr_db in snr_db_list:
        sigma = mean_ref_rms * 10.0 ** (-snr_db / 20.0)
        eval_cfg = replace(env_cfg,
                           sigma_meas=float(sigma),
                           randomize_snr=False)
        env = AFDEnv(pair_scenarios, nu_gap_matrix, eval_cfg)

        cm = np.zeros((H, H))
        step_log = []
        action_counts = np.zeros((H, M))

        for true_k in range(H):
            for trial in range(n_mc):
                obs, info = env.reset(seed=base_seed + 10_000 * true_k + trial)
                env.true_idx = true_k
                obs = env.belief.copy()
                done = False
                steps = 0
                while not done:
                    a = policy.act(obs)
                    action_counts[true_k, a] += 1
                    obs, r, term, trunc, info = env.step(a)
                    done = term or trunc
                    steps += 1
                cm[true_k, info["decided"]] += 1
                step_log.append(steps)
        cm /= n_mc

        out[snr_db] = {
            "cm": cm,
            "avg_steps": float(np.mean(step_log)),
            "action_counts": action_counts,
        }
    return out
