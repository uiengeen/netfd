"""
Active fault diagnosis as a sequential active-hypothesis-testing problem.

Hidden state: true fault index F* in {0, ..., H - 1}.
Action:       a in {0, ..., M - 1}, selecting an (injection, observation)
              pair. The probing signal for each pair is pre-designed
              offline (ν-gap-informed multi-sine).
Observation:  belief b_k in R^H (sums to 1).
Likelihood:   Gaussian on the time-domain residual under the selected pair.
Reward:       r = alpha (H_{k-1} - H_k)                 information gain
                 - beta * sum_{top-2} b_i b_j (1 - dnu_{ij})  separability
                 - gamma                                step cost
              terminal bonus +R_succ or +R_fail on argmax(b).
Termination:  max(b) > belief_threshold  OR  k = K_max.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from netfd.systems.synthesis import GlobalSystem
from netfd.systems.nu_gap import psi_omega
from netfd.viz.time_domain import simulate_time_response
from netfd.diagnosis.probing import (
    find_top_peaks, design_multi_sine_probe,
    DEFAULT_TOP_K_PEAKS, DEFAULT_PEAK_PROMINENCE,
    DEFAULT_PEAK_REL_THRESH, DEFAULT_INV_DETECT_FLOOR,
)


# =============================================================================
# Environment configuration
# =============================================================================

@dataclass
class AFDEnvConfig:
    # Episode
    K_max: int = 10
    belief_threshold: float = 0.9

    # Reward shaping
    alpha_info: float = 1.0
    beta_amb: float = 0.5
    step_cost: float = 0.05
    R_succ: float = 5.0
    R_fail: float = -5.0

    # Probing / simulation
    probe_duration: float = 4.0
    sim_dt: float = 0.01
    probe_amp: float = 1.0

    # Measurement
    sigma_meas: float = 0.05
    likelihood_temp: float = 1.0

    # SNR randomization (training-time robustness).
    # If `randomize_snr` is True, sigma_meas is sampled each reset() from
    # SNR ~ U(snr_db_low, snr_db_high) relative to `ref_rms_for_snr`.
    randomize_snr: bool = False
    snr_db_low: float = -20.0
    snr_db_high: float = 0.0
    ref_rms_for_snr: float = 1.0


# =============================================================================
# Per-(injection, observation) pair offline precomputation
# =============================================================================

@dataclass
class PairScenario:
    """Everything needed at training/eval time for one I/O pair."""
    inj_node: int
    obs_node: int
    sys_hyps_pair: List[GlobalSystem]
    signal: np.ndarray                  # (T, 1)
    y_clean: np.ndarray                 # (H, T, 1)
    y_norm: np.ndarray                  # (H,)
    peak_freqs_per_hyp: list
    peak_psi_per_hyp: list


def build_pair_scenarios(sys_hyps_full: List[GlobalSystem],
                         healthy_idx: int,
                         pair_list: List[Tuple[int, int]],
                         omega_grid: np.ndarray,
                         probe_duration: float,
                         sim_dt: float,
                         probe_amp: float,
                         top_k_peaks: int = DEFAULT_TOP_K_PEAKS,
                         peak_prominence: float = DEFAULT_PEAK_PROMINENCE,
                         peak_rel_thresh: float = DEFAULT_PEAK_REL_THRESH,
                         inv_detect_floor: float = DEFAULT_INV_DETECT_FLOOR,
                         verbose: bool = True,
                         ) -> List[PairScenario]:
    """Precompute everything per (injection, observation) pair.

    Returns one `PairScenario` per pair, containing the multi-sine probe
    signal and the clean per-hypothesis time-domain responses.
    """
    t = np.arange(0.0, probe_duration, sim_dt, dtype=np.float32)
    scenarios: List[PairScenario] = []

    for (inj, obs) in pair_list:
        sys_pair = [s.with_obs([obs]) for s in sys_hyps_full]
        P_nom_pair = sys_pair[healthy_idx]

        # Top peaks of psi(omega) for each fault hypothesis vs healthy.
        peak_freqs, peak_psi = [], []
        for k, sysH in enumerate(sys_pair):
            if k == healthy_idx:
                peak_freqs.append(np.array([]))
                peak_psi.append(np.array([]))
                continue
            psi = psi_omega(P_nom_pair, sysH, omega_grid)
            if np.max(psi) < 1e-8:
                # Structurally invisible fault under this pair.
                peak_freqs.append(np.array([]))
                peak_psi.append(np.array([]))
            else:
                f, p = find_top_peaks(psi, omega_grid,
                                      top_k=top_k_peaks,
                                      prominence=peak_prominence,
                                      rel_thresh=peak_rel_thresh)
                peak_freqs.append(f)
                peak_psi.append(p)

        # Multi-sine probe: only fault hypotheses contribute.
        fault_freqs = [peak_freqs[k] for k in range(len(sys_pair))
                       if k != healthy_idx and len(peak_freqs[k]) > 0]
        fault_psis = [peak_psi[k] for k in range(len(sys_pair))
                      if k != healthy_idx and len(peak_psi[k]) > 0]
        if len(fault_freqs) == 0:
            # Fallback: log-spaced sweep if no peaks are visible at all.
            v = np.zeros((len(t), 1), dtype=np.float32)
            for w in np.linspace(1.0, 8.0, 5):
                v[:, 0] += float(probe_amp) * np.sin(w * t)
        else:
            v = design_multi_sine_probe(
                fault_freqs, fault_psis, t, probe_amp,
                inv_detect_floor=inv_detect_floor,
            ).astype(np.float32)

        H = len(sys_pair)
        y_clean = np.zeros((H, len(t), 1), dtype=np.float32)
        for k in range(H):
            y_clean[k] = simulate_time_response(sys_pair[k], v, t).astype(np.float32)
        y_norm = np.array(
            [float(np.sqrt(np.mean(y_clean[k] ** 2))) for k in range(H)],
            dtype=np.float32,
        )

        if verbose:
            n_visible = sum(1 for k in range(H)
                            if k != healthy_idx and len(peak_freqs[k]) > 0)
            print(f"    pair (N{inj + 1} -> N{obs + 1}): "
                  f"visible faults = {n_visible}/{H - 1}, "
                  f"signal RMS = {float(np.sqrt(np.mean(v ** 2))):.4f}, "
                  f"y_nom RMS = {y_norm[healthy_idx]:.4f}")

        scenarios.append(PairScenario(
            inj_node=inj, obs_node=obs,
            sys_hyps_pair=sys_pair,
            signal=v, y_clean=y_clean, y_norm=y_norm,
            peak_freqs_per_hyp=peak_freqs,
            peak_psi_per_hyp=peak_psi,
        ))
    return scenarios


def precompute_nu_gap_matrix(sys_hyps: List[GlobalSystem],
                             omega_grid: np.ndarray) -> np.ndarray:
    """Symmetric H x H matrix of pairwise sup_omega psi between hypotheses."""
    H = len(sys_hyps)
    D = np.zeros((H, H), dtype=np.float32)
    for i in range(H):
        for j in range(i + 1, H):
            psi = psi_omega(sys_hyps[i], sys_hyps[j], omega_grid)
            D[i, j] = float(np.max(psi))
            D[j, i] = D[i, j]
    return D


# =============================================================================
# Environment
# =============================================================================

class _Space:
    """Minimal gym-like space object."""
    def __init__(self, shape=None, n=None):
        self.shape = shape
        self.n = n


class AFDEnv:
    """Sequential active-fault-diagnosis environment."""

    def __init__(self,
                 pair_scenarios: List[PairScenario],
                 nu_gap_matrix: np.ndarray,
                 cfg: AFDEnvConfig):
        self.cfg = cfg
        self.scenarios = pair_scenarios
        self.M = len(pair_scenarios)
        self.H = pair_scenarios[0].y_clean.shape[0]
        self.T = pair_scenarios[0].y_clean.shape[1]
        self.D = nu_gap_matrix.astype(np.float32)

        self.y_clean = np.stack([s.y_clean for s in pair_scenarios], axis=0)  # (M, H, T, 1)
        self.y_norm = np.stack([s.y_norm for s in pair_scenarios], axis=0)    # (M, H)

        self.observation_space = _Space(shape=(self.H,))
        self.action_space = _Space(n=self.M)

        self.belief = None
        self.k = 0
        self.true_idx = 0
        self.rng = np.random.default_rng()

    # ---- Episode lifecycle ----

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.belief = np.ones(self.H, dtype=np.float32) / self.H
        self.k = 0
        self.true_idx = int(self.rng.integers(0, self.H))

        if self.cfg.randomize_snr:
            snr_db = self.rng.uniform(self.cfg.snr_db_low, self.cfg.snr_db_high)
            self.cfg.sigma_meas = float(
                self.cfg.ref_rms_for_snr * 10.0 ** (-snr_db / 20.0)
            )
        return self.belief.copy(), {"true_idx": self.true_idx}

    def step(self, action):
        a = int(action)
        sigma = self.cfg.sigma_meas

        # ---- Measurement under the chosen pair ----
        y_clean_true = self.y_clean[a, self.true_idx]
        noise = self.rng.standard_normal(y_clean_true.shape).astype(np.float32) * sigma
        y_meas = y_clean_true + noise

        # ---- Bayesian posterior over hypotheses ----
        diff = y_meas[None, ...] - self.y_clean[a]          # (H, T, 1)
        J = (diff ** 2).sum(axis=(1, 2))                    # (H,)
        log_lik = -J / (2.0 * sigma ** 2 * self.cfg.likelihood_temp)
        log_post = np.log(self.belief + 1e-30) + log_lik
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()

        belief_prev = self.belief
        self.belief = post.astype(np.float32)

        # ---- Reward ----
        eps = 1e-12
        H_prev = -float((belief_prev * np.log(belief_prev + eps)).sum())
        H_curr = -float((self.belief * np.log(self.belief + eps)).sum())
        info_gain = self.cfg.alpha_info * (H_prev - H_curr)

        top2 = np.argsort(self.belief)[-2:]
        i_top, j_top = int(top2[-1]), int(top2[-2])
        ambig = float(self.belief[i_top] * self.belief[j_top]
                      * (1.0 - self.D[i_top, j_top]))
        sep_pen = -self.cfg.beta_amb * ambig

        reward = info_gain + sep_pen - self.cfg.step_cost

        # ---- Termination ----
        self.k += 1
        decided = int(np.argmax(self.belief))
        is_conf = float(self.belief.max()) > self.cfg.belief_threshold
        timed_out = self.k >= self.cfg.K_max
        terminated = bool(is_conf or timed_out)
        truncated = False

        if terminated:
            reward += self.cfg.R_succ if decided == self.true_idx else self.cfg.R_fail

        info = {
            "true_idx": self.true_idx,
            "decided": decided,
            "correct": bool(decided == self.true_idx),
            "k": self.k,
            "max_belief": float(self.belief.max()),
            "entropy": H_curr,
            "info_gain": info_gain,
            "sep_pen": sep_pen,
            "action": a,
            "predicted_idx": decided,
            "true_fault_idx": self.true_idx,
        }
        return self.belief.copy(), float(reward), terminated, truncated, info
