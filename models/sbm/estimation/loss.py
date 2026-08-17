from __future__ import annotations

import numpy as np


def joint_loss(
    theta_free: np.ndarray,
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
    structural_prior_mean: np.ndarray | None = None,
    structural_prior_tau: np.ndarray | None = None,
    drift_enabled: np.ndarray | None = None,
    drift_prior_mean: np.ndarray | None = None,
    drift_prior_tau: np.ndarray | None = None,
) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, dict[str, float]]:
    """Combined loss: covariance + trend-with-drift + shrinkage + priors + stability.

    The parameter vector is [free J-I entries, enabled drift entries]. If no
    drift entries are enabled, this reduces to the previous fixed-u model.
    """
    n_j = int(np.sum(M))
    j_free = np.asarray(theta_free[:n_j], dtype=float)
    c_free = np.asarray(theta_free[n_j:], dtype=float)
    J = _rebuild_J(j_free, M)
    c = _rebuild_drift(c_free, np.asarray(u, dtype=float), drift_enabled)

    LR, gradR = _covariance_loss(J, Sigma, Sigma_eta, normalize_r_by_counts)
    LX, gradX, gradc = _mean_drift_loss(J, Xbar, c, x_weights, normalize_x_by_counts)
    LJ, gradJ = _shrinkage_loss(J, J_ref)
    L_prior_J, grad_prior_J = _structural_prior_loss(J, structural_prior_mean, structural_prior_tau, M)
    L_prior_c, grad_prior_c = _drift_prior_loss(c_free, drift_enabled, drift_prior_mean, drift_prior_tau)
    L_stab, grad_stab = _stability_penalty(J, w_stab, eps)

    scales = component_scales or {}
    sX = float(scales.get("LX", 1.0))
    sR = float(scales.get("LR", 1.0))
    sJ = float(scales.get("LJ", 1.0))
    sPJ = float(scales.get("L_prior_J", 1.0))
    sPC = float(scales.get("L_prior_c", 1.0))
    sS = float(scales.get("L_stab", 1.0))

    LXn = LX / (sX + norm_eps)
    LRn = LR / (sR + norm_eps)
    LJn = LJ / (sJ + norm_eps)
    LPJn = L_prior_J / (sPJ + norm_eps)
    LPCn = L_prior_c / (sPC + norm_eps)
    L_stab_n = L_stab / (sS + norm_eps)

    L = wR * LRn + wX * LXn + wJ * LJn + LPJn + LPCn + L_stab_n
    grad_full = (
        wR * gradR / (sR + norm_eps)
        + wX * gradX / (sX + norm_eps)
        + wJ * gradJ / (sJ + norm_eps)
        + grad_prior_J / (sPJ + norm_eps)
        + grad_stab / (sS + norm_eps)
    )
    grad_j_free = grad_full[M == 1]
    if c_free.size:
        enabled = np.asarray(drift_enabled, dtype=bool)
        grad_c_free = wX * gradc[enabled] / (sX + norm_eps) + grad_prior_c / (sPC + norm_eps)
        grad_free = np.concatenate([grad_j_free, grad_c_free])
    else:
        grad_free = grad_j_free

    if not return_components:
        return L, grad_free

    components = {
        "LX": float(LX),
        "LR": float(LR),
        "LJ": float(LJ),
        "L_prior_J": float(L_prior_J),
        "L_prior_c": float(L_prior_c),
        "L_stab": float(L_stab),
        "LX_norm": float(LXn),
        "LR_norm": float(LRn),
        "LJ_norm": float(LJn),
        "L_prior_J_norm": float(LPJn),
        "L_prior_c_norm": float(LPCn),
        "L_stab_norm": float(L_stab_n),
        "LX_weighted": float(wX * LXn),
        "LR_weighted": float(wR * LRn),
        "LJ_weighted": float(wJ * LJn),
        "L_prior_J_weighted": float(LPJn),
        "L_prior_c_weighted": float(LPCn),
        "L_stab_weighted": float(L_stab_n),
        "L_total": float(L),
    }
    return L, grad_free, components


def split_theta(theta_free: np.ndarray, M: np.ndarray, drift_enabled: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    n_j = int(np.sum(M))
    return np.asarray(theta_free[:n_j], dtype=float), np.asarray(theta_free[n_j:], dtype=float)


def rebuild_from_theta(theta_free: np.ndarray, M: np.ndarray, u: np.ndarray, drift_enabled: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    j_free, c_free = split_theta(theta_free, M, drift_enabled)
    return _rebuild_J(j_free, M), _rebuild_drift(c_free, u, drift_enabled)


def _rebuild_J(JmI_vec_free: np.ndarray, M: np.ndarray) -> np.ndarray:
    m = M.shape[0]
    JmI = np.zeros((m, m), dtype=float)
    JmI[M == 1] = JmI_vec_free
    return np.eye(m) + JmI


def _rebuild_drift(c_free: np.ndarray, u: np.ndarray, drift_enabled: np.ndarray | None) -> np.ndarray:
    c = np.asarray(u, dtype=float).copy()
    if drift_enabled is None or c_free.size == 0:
        return c
    enabled = np.asarray(drift_enabled, dtype=bool)
    if int(enabled.sum()) != c_free.size:
        raise ValueError(f"Drift free-parameter mismatch: {c_free.size} vs {enabled.sum()}")
    c[enabled] = c_free
    return c


def _covariance_loss(J, Sigma, Sigma_eta, normalize_by_counts: bool = True) -> tuple[float, np.ndarray]:
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
    c,
    x_weights: np.ndarray | None = None,
    normalize_by_counts: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    m, T = Xbar.shape
    LX = 0.0
    gradX = np.zeros((m, m), dtype=float)
    gradc = np.zeros(m, dtype=float)
    wvec = np.ones(m, dtype=float) if x_weights is None else np.asarray(x_weights, dtype=float).ravel()
    if wvec.size != m:
        raise ValueError(f"x_weights length mismatch: got {wvec.size}, expected {m}")
    W = wvec[:, None]
    n_steps = max(T - 1, 1)
    scale = (n_steps * m) if normalize_by_counts else 1.0

    for t in range(1, T):
        EX = Xbar[:, t] - J @ Xbar[:, t - 1] - c
        WEX = wvec * EX
        LX += (1.0 / scale) * float(EX.T @ WEX)
        gradX -= (2.0 / scale) * (W * EX[:, None]) @ Xbar[:, t - 1][None, :]
        gradc -= (2.0 / scale) * WEX

    return LX, gradX, gradc


def _shrinkage_loss(J: np.ndarray, J_ref: np.ndarray | None) -> tuple[float, np.ndarray]:
    if J_ref is None:
        return 0.0, np.zeros_like(J)
    EJ = J - J_ref
    return float(np.sum(EJ ** 2)), 2.0 * EJ


def _structural_prior_loss(
    J: np.ndarray,
    prior_mean: np.ndarray | None,
    prior_tau: np.ndarray | None,
    M: np.ndarray,
) -> tuple[float, np.ndarray]:
    grad = np.zeros_like(J)
    if prior_mean is None or prior_tau is None:
        return 0.0, grad
    mean = np.asarray(prior_mean, dtype=float)
    tau = np.asarray(prior_tau, dtype=float)
    mask = (M == 1) & np.isfinite(mean) & np.isfinite(tau) & (tau > 0)
    if not np.any(mask):
        return 0.0, grad
    residual = np.zeros_like(J, dtype=float)
    residual[mask] = (J[mask] - mean[mask]) / tau[mask]
    loss = float(np.sum(residual[mask] ** 2))
    grad[mask] = 2.0 * residual[mask] / tau[mask]
    return loss, grad


def _drift_prior_loss(
    c_free: np.ndarray,
    drift_enabled: np.ndarray | None,
    prior_mean: np.ndarray | None,
    prior_tau: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    if drift_enabled is None or c_free.size == 0 or prior_mean is None or prior_tau is None:
        return 0.0, np.zeros_like(c_free, dtype=float)
    enabled = np.asarray(drift_enabled, dtype=bool)
    mean = np.asarray(prior_mean, dtype=float)[enabled]
    tau = np.asarray(prior_tau, dtype=float)[enabled]
    valid = np.isfinite(tau) & (tau > 0)
    grad = np.zeros_like(c_free, dtype=float)
    if not np.any(valid):
        return 0.0, grad
    residual = (c_free[valid] - mean[valid]) / tau[valid]
    loss = float(np.sum(residual ** 2))
    grad[valid] = 2.0 * residual / tau[valid]
    return loss, grad


def _stability_penalty(J: np.ndarray, w_stab: float, eps: float) -> tuple[float, np.ndarray]:
    eigvals = np.linalg.eigvals(J)
    rho = np.max(np.real(eigvals))
    if rho > 1 - eps:
        penalty = w_stab * (rho - (1 - eps)) ** 2
        grad = 2 * w_stab * (rho - (1 - eps)) * np.eye(J.shape[0])
        return float(penalty), grad
    return 0.0, np.zeros_like(J)
