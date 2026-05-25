"""
Vinnicombe ν-gap metric for `GlobalSystem` pairs.

The pointwise chordal distance is
    psi(omega) = sigma_max[ (I + P1 P1*)^(-1/2) (P0 - P1) (I + P0* P0)^(-1/2) ]
and the ν-gap is
    delta_nu(P0, P1) = sup_omega psi(omega)    if the winding-number
                                                condition holds
                     = 1                        otherwise
"""

from typing import Tuple

import numpy as np

from netfd.systems.synthesis import GlobalSystem


# ---------------------------------------------------------------------------
# Linear-algebra helper
# ---------------------------------------------------------------------------

def _hermitian_inv_sqrt(M: np.ndarray) -> np.ndarray:
    """Inverse principal square root of a Hermitian positive-definite matrix."""
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    w_clipped = np.clip(w, 1e-14, None)
    return (V * (w_clipped ** -0.5)) @ V.conj().T


# ---------------------------------------------------------------------------
# Pointwise chordal distance
# ---------------------------------------------------------------------------

def psi_omega(sys0: GlobalSystem, sys1: GlobalSystem,
              omega: np.ndarray) -> np.ndarray:
    """Pointwise chordal distance psi(omega) between two systems."""
    if sys0.ny != sys1.ny or sys0.nu != sys1.nu:
        raise ValueError("psi_omega: systems must share input/output dimensions")

    H0 = sys0.freqresp(omega)
    H1 = sys1.freqresp(omega)
    Nw, ny, nu = H0.shape
    psi = np.zeros(Nw)
    Iy = np.eye(ny)
    Iu = np.eye(nu)

    for k in range(Nw):
        P0, P1 = H0[k], H1[k]
        L = _hermitian_inv_sqrt(Iy + P1 @ P1.conj().T)
        R = _hermitian_inv_sqrt(Iu + P0.conj().T @ P0)
        sv = np.linalg.svd(L @ (P0 - P1) @ R, compute_uv=False)
        psi[k] = float(sv[0])
    return psi


# ---------------------------------------------------------------------------
# Winding-number check
# ---------------------------------------------------------------------------

def winding_number_ok(sys0: GlobalSystem, sys1: GlobalSystem,
                      omega: np.ndarray = None,
                      verbose: bool = False) -> bool:
    """Approximate the MIMO Vinnicombe winding-number condition on a grid.

    For stable nominal/faulty pairs (the typical case here), both systems
    have no RHP poles, so the required winding is 0.
    """
    if omega is None:
        omega = np.logspace(-3, 3, 4001)

    H0 = sys0.freqresp(omega)
    H1 = sys1.freqresp(omega)
    nu = sys0.nu
    Iu = np.eye(nu)

    det_path = np.zeros(len(omega), dtype=complex)
    for k in range(len(omega)):
        det_path[k] = np.linalg.det(Iu + H1[k].conj().T @ H0[k])

    if np.any(np.abs(det_path) < 1e-9):
        if verbose:
            print("  [winding] det(I + P1* P0) ~ 0 on grid")
        return False

    phase = np.unwrap(np.angle(det_path))
    winding_approx = (phase[-1] - phase[0]) / (2.0 * np.pi)

    p0_unstable = int(np.sum(np.real(np.linalg.eigvals(sys0.A)) > 1e-9))
    p1_unstable = int(np.sum(np.real(np.linalg.eigvals(sys1.A)) > 1e-9))
    required = p0_unstable - p1_unstable

    ok = abs(winding_approx - required) < 0.5
    if verbose:
        print(f"  [winding] approx={winding_approx:.3f} required={required} ok={ok}")
    return ok


# ---------------------------------------------------------------------------
# Scalar ν-gap
# ---------------------------------------------------------------------------

def nu_gap(sys0: GlobalSystem, sys1: GlobalSystem,
           omega: np.ndarray = None,
           check_winding: bool = True,
           verbose: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the ν-gap.

    Returns
    -------
    delta_nu : float in [0, 1]
    omega    : the frequency grid used
    psi      : the pointwise chordal distance at each omega
    """
    if omega is None:
        omega = np.logspace(-2, 2, 401)

    if check_winding and not winding_number_ok(sys0, sys1, omega, verbose=verbose):
        if verbose:
            print("  [nu_gap] winding condition failed -> delta_nu = 1")
        psi = psi_omega(sys0, sys1, omega)
        return 1.0, omega, psi

    psi = psi_omega(sys0, sys1, omega)
    return float(np.max(psi)), omega, psi
