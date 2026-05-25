"""
Frequency-weighted residual classifier.

Given a measured time-domain response `y_meas` and per-hypothesis clean
predictions `y_pred_list`, compute a residual cost at each peak frequency
that the hypothesis set considers informative, weight it by the maximum
detectability across hypotheses, and pick the hypothesis with minimum cost.

This is the diagnostic used by every single-shot experiment in the paper.
"""

from typing import List, Tuple

import numpy as np

from netfd.diagnosis.probing import DEFAULT_INV_DETECT_FLOOR


def diag_freq_weighted(y_meas: np.ndarray,
                       y_pred_list: List[np.ndarray],
                       peak_freqs_per_hyp: List[np.ndarray],
                       peak_psi_per_hyp: List[np.ndarray],
                       t: np.ndarray,
                       inv_detect_floor: float = DEFAULT_INV_DETECT_FLOOR,
                       ) -> Tuple[int, np.ndarray]:
    """Frequency-weighted residual diagnostic.

    Evaluate the residual at the UNION of all hypothesis peak frequencies.
    The weight at frequency w is the maximum psi value across all
    hypotheses that listed w as a peak, floored at `inv_detect_floor`.

    Returns
    -------
    decision : int  (argmin over hypotheses)
    costs    : (H,) per-hypothesis cost J_k
    """
    dt = float(t[1] - t[0])
    H = len(y_pred_list)

    # Build the union frequency set and per-frequency weights.
    freq_to_weight = {}
    for ws, ps in zip(peak_freqs_per_hyp, peak_psi_per_hyp):
        for w, p in zip(ws, ps):
            w_key = float(w)
            if w_key not in freq_to_weight or freq_to_weight[w_key] < p:
                freq_to_weight[w_key] = float(p)
    union_freqs = np.array(sorted(freq_to_weight.keys()))
    union_weights = np.array([freq_to_weight[w] for w in union_freqs])
    union_weights = np.maximum(union_weights, inv_detect_floor)

    # DFT basis on the union frequency set.
    cos_basis = np.stack([np.cos(w * t) for w in union_freqs], axis=0)  # (M, T)
    sin_basis = np.stack([np.sin(w * t) for w in union_freqs], axis=0)

    J = np.zeros(H)
    for k in range(H):
        diff = y_meas - y_pred_list[k]              # (T, ny)
        Re = cos_basis @ diff * dt                  # (M, ny)
        Im = sin_basis @ diff * dt
        energy = (Re ** 2 + Im ** 2).sum(axis=1)    # (M,)
        J[k] = float((union_weights ** 2 * energy).sum())
    return int(np.argmin(J)), J
