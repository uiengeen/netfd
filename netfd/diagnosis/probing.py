"""
ν-gap-informed probe design.

Two ingredients:
  - `find_top_peaks`         : extract the most informative peaks of psi(omega)
  - `design_multi_sine_probe`: build a multi-sine probe combining all peaks
                               across hypotheses, weighted by inverse
                               detectability (so harder-to-distinguish faults
                               get more excitation energy)

These two functions are also re-used by `netfd.sequential.env` when it
builds per-(injection, observation)-pair offline scenarios.
"""

from typing import List, Tuple

import numpy as np
from scipy.signal import find_peaks

from netfd.systems.synthesis import GlobalSystem
from netfd.systems.nu_gap import psi_omega


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K_PEAKS: int = 3
DEFAULT_PEAK_PROMINENCE: float = 1e-4
DEFAULT_PEAK_REL_THRESH: float = 0.05
DEFAULT_INV_DETECT_FLOOR: float = 0.05


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------

def find_top_peaks(psi: np.ndarray,
                   omega: np.ndarray,
                   top_k: int = DEFAULT_TOP_K_PEAKS,
                   prominence: float = DEFAULT_PEAK_PROMINENCE,
                   rel_thresh: float = DEFAULT_PEAK_REL_THRESH,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Return up to `top_k` peaks of psi(omega), sorted by amplitude.

    A peak qualifies if its height is at least `rel_thresh * max(psi)`. If
    no peak qualifies, fall back to the single argmax.
    """
    pks, _ = find_peaks(psi, prominence=prominence)
    if len(pks) == 0:
        return (np.array([omega[int(np.argmax(psi))]]),
                np.array([float(np.max(psi))]))

    psi_max = float(np.max(psi))
    pks = [p for p in pks if psi[p] >= rel_thresh * psi_max]
    if len(pks) == 0:
        return (np.array([omega[int(np.argmax(psi))]]),
                np.array([float(np.max(psi))]))

    pks = sorted(pks, key=lambda p: -psi[p])[:top_k]
    return omega[np.array(pks)], psi[np.array(pks)]


# ---------------------------------------------------------------------------
# Multi-sine probe
# ---------------------------------------------------------------------------

def design_multi_sine_probe(peak_freqs_per_hyp: List[np.ndarray],
                            peak_psi_per_hyp: List[np.ndarray],
                            t: np.ndarray,
                            amp_ref: float,
                            inv_detect_floor: float = DEFAULT_INV_DETECT_FLOOR,
                            ) -> np.ndarray:
    """Compose a multi-sine signal from per-hypothesis peak descriptors.

    For each hypothesis k:
      - inter-hypothesis weight = 1 / max(strongest peak, floor),
        normalized so max weight = 1 (harder faults get more energy)
      - intra-hypothesis relative weight = peak_psi / max(peak_psi),
        so the dominant peak of each hypothesis carries the most energy

    Returns
    -------
    v : (T, 1) probe signal
    """
    H = len(peak_freqs_per_hyp)
    hyp_weights = np.zeros(H)
    for k in range(H):
        if len(peak_psi_per_hyp[k]) > 0:
            strongest = float(max(peak_psi_per_hyp[k]))
        else:
            strongest = inv_detect_floor
        hyp_weights[k] = 1.0 / max(strongest, inv_detect_floor)
    if hyp_weights.max() > 0:
        hyp_weights /= hyp_weights.max()

    v = np.zeros(len(t))
    for k in range(H):
        ws = peak_freqs_per_hyp[k]
        ps = peak_psi_per_hyp[k]
        if len(ws) == 0:
            continue
        rel = ps / max(ps.max(), 1e-12)
        for w_m, r_m in zip(ws, rel):
            v += amp_ref * hyp_weights[k] * r_m * np.sin(w_m * t)
    return v.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Convenience: peaks for an entire hypothesis set
# ---------------------------------------------------------------------------

def get_peaks_for_hypotheses(P_nom: GlobalSystem,
                             sys_hyps: List[GlobalSystem],
                             omega: np.ndarray,
                             top_k: int = DEFAULT_TOP_K_PEAKS,
                             prominence: float = DEFAULT_PEAK_PROMINENCE,
                             rel_thresh: float = DEFAULT_PEAK_REL_THRESH,
                             ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """For each hypothesis system, return its top peaks of psi vs the nominal."""
    peak_freqs, peak_psi = [], []
    for sysH in sys_hyps:
        psi = psi_omega(P_nom, sysH, omega)
        f, p = find_top_peaks(psi, omega, top_k=top_k,
                              prominence=prominence, rel_thresh=rel_thresh)
        peak_freqs.append(f)
        peak_psi.append(p)
    return peak_freqs, peak_psi
