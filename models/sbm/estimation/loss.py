from __future__ import annotations

import numpy as np

def joint_loss(
    JmI_vec_free: np.ndarray,
    Xbar: np.ndarray,
    Sigma: np.ndarray,
    Sigma_eta: np.ndarray,
    wR: float,
    wX: float,
    u: np.ndarray,
    M: np.ndarray,
    J_ref: np.ndarray | None = None,
    wJ: float = 0.0,
    w_stab: float = 10.0,
    eps: float = 1e-3,
    x_weights: np.ndarray | None = None,
    normalize_x_by_counts: bool = True,
    normalize_r_by_counts: bool = True,
    component_scales: dict[str, float] | None = None,
    norm_eps: float = 1e-8,
    return_components: bool = False,
) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, dict[str, float]]:
    """
    Combined loss: covariance + drift + shrinkage + stability.
    Note: Discrete
    Returns (loss, gradient w.r.t. free parameters).
    """
    J = _rebuild_J(JmI_vec_free, M)

    LR, gradR = _covariance_loss(J, Sigma, Sigma_eta, normalize_r_by_counts)
    LX, gradX = _mean_drift_loss(J, Xbar, u, x_weights, normalize_x_by_counts)
    LJ, gradJ = _shrinkage_loss(J, J_ref)
    L_stab, grad_stab = _stability_penalty(J, w_stab, eps)

    scales = component_scales or {}
    sX = float(scales.get("LX", 1.0))
    sR = float(scales.get("LR", 1.0))
    sJ = float(scales.get("LJ", 1.0))
    sS = float(scales.get("L_stab", 1.0))

    LXn = LX / (sX + norm_eps)
    LRn = LR / (sR + norm_eps)
    LJn = LJ / (sJ + norm_eps)
    L_stab_n = L_stab / (sS + norm_eps)

    # Apply user weights at final combine step.
    L = wR * LRn + wX * LXn + wJ * LJn + L_stab_n
    grad_full = (
        wR * gradR / (sR + norm_eps)
        + wX * gradX / (sX + norm_eps)
        + wJ * gradJ / (sJ + norm_eps)
        + grad_stab / (sS + norm_eps)
    )

    grad_free = grad_full[M == 1]
    if not return_components:
        return L, grad_free

    components = {
        "LX": float(LX),
        "LR": float(LR),
        "LJ": float(LJ),
        "L_stab": float(L_stab),
        "LX_norm": float(LXn),
        "LR_norm": float(LRn),
        "LJ_norm": float(LJn),
        "L_stab_norm": float(L_stab_n),
        "LX_weighted": float(wX * LXn),
        "LR_weighted": float(wR * LRn),
        "LJ_weighted": float(wJ * LJn),
        "L_stab_weighted": float(L_stab_n),
        "L_total": float(L),
    }
    return L, grad_free, components

def _rebuild_J(JmI_vec_free: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Reconstruct J from free parameters."""
    m = M.shape[0]
    JmI = np.zeros((m, m))      # (J-I) matrix
    JmI[M == 1] = JmI_vec_free
    J = np.eye(m) + JmI         # J = I + (J-I)

    return J

def _covariance_loss(
    J,
    Sigma,
    Sigma_eta,
    normalize_by_counts: bool = True,
) -> tuple[float, np.ndarray]:
        """Lyapunov loss: ||Sigma - J Sigma J' - Sigma_eta||^2, with gradient."""

        Sigma_eta_diag = np.diag(np.diag(Sigma_eta))

        ER = -Sigma + J @ Sigma @ J.T + Sigma_eta_diag
        m = J.shape[0]
        scale = (m * m) if normalize_by_counts else 1.0
        gradR = (4.0 / scale) * ER @ J @ Sigma

        LR = (1.0 / scale) * np.sum(ER ** 2)

        return LR, gradR

def _mean_drift_loss(
    J,
    Xbar,
    u,
    x_weights: np.ndarray | None = None,
    normalize_by_counts: bool = True,
) -> tuple[float, np.ndarray]:
    """Mean drift loss: ||Xbar[:, t] - J Xbar[:, t-1] - u||^2, with gradient."""
    m, T = Xbar.shape
    LX = 0.0
    gradX = np.zeros((m, m))
    wvec = np.ones(m, dtype=float) if x_weights is None else np.asarray(x_weights, dtype=float).ravel()
    if wvec.size != m:
        raise ValueError(f"x_weights length mismatch: got {wvec.size}, expected {m}")
    W = wvec[:, None]
    n_steps = max(T - 1, 1)
    scale = (n_steps * m) if normalize_by_counts else 1.0

    for t in range(1, T):
        EX = Xbar[:, t] - J @ Xbar[:, t-1] - u
        WEX = wvec * EX
        LX += (1.0 / scale) * float(EX.T @ WEX)
        gradX -= (2.0 / scale) * (W * EX[:, None]) @ Xbar[:, t-1][None, :]

    return LX, gradX

def _shrinkage_loss(
    J: np.ndarray,
    J_ref: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Shrinkage toward reference J."""
    if J_ref is None:
        gradJ_shrink = np.zeros_like(J)
        LJ = 0.0
    else:
        EJ = J - J_ref
        gradJ_shrink = 2.0 * EJ
        LJ = float(np.sum(EJ ** 2))

    return LJ, gradJ_shrink


def _stability_penalty(
    J: np.ndarray,
    w_stab: float,
    eps: float,
) -> tuple[float, np.ndarray]:
    """Penalize unstable eigenvalues."""

    eigvals = np.linalg.eigvals(J)
    rho = np.max(np.real(eigvals))

    if rho > 1 - eps:
        penalty = w_stab * (rho - (1 - eps)) ** 2
        # approximate gradient 
        grad = 2 * w_stab * (rho - (1 - eps)) * np.eye(J.shape[0])
        return penalty, grad

    return 0.0, np.zeros_like(J)
