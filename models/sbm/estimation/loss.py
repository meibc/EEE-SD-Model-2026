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
    stability_fd_eps: float = 1e-7,
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
    L_stab, grad_stab = _stability_penalty(J, w_stab, eps, stability_fd_eps)

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


def _stability_penalty(
    J: np.ndarray,
    w_stab: float,
    eps: float,
    fd_eps: float = 1e-7,
) -> tuple[float, np.ndarray]:
    """Penalize discrete-time instability using the spectral radius.

    The stability criterion for the SEM transition matrix is max(abs(lambda)) < 1.
    The gradient uses the analytic left/right eigenvector derivative for the
    dominant eigenvalue, with finite-difference fallback when the dominant mode
    is nearly tied or poorly conditioned.
    """
    eigvals, right_vecs = np.linalg.eig(J)
    abs_vals = np.abs(eigvals)
    dom_idx = int(np.argmax(abs_vals))
    rho = float(abs_vals[dom_idx])
    threshold = 1.0 - float(eps)
    if rho <= threshold:
        return 0.0, np.zeros_like(J)

    excess = rho - threshold
    penalty = float(w_stab) * excess**2
    d_rho = _spectral_radius_gradient_analytic(J, eigvals, right_vecs, dom_idx, fd_eps)
    grad = 2.0 * float(w_stab) * excess * d_rho
    return penalty, grad


def _spectral_radius_gradient_analytic(
    J: np.ndarray,
    eigvals: np.ndarray,
    right_vecs: np.ndarray,
    dom_idx: int,
    fd_eps: float = 1e-7,
) -> np.ndarray:
    """Gradient of max(abs(lambda)) with fallback for degenerate spectra."""
    lam = eigvals[dom_idx]
    rho = float(abs(lam))
    if rho <= 0:
        return _spectral_radius_gradient_fd(J, fd_eps)

    abs_vals = np.abs(eigvals)
    if abs_vals.size > 1:
        ordered = np.sort(abs_vals)
        gap = float(ordered[-1] - ordered[-2])
        # Near ties make the spectral-radius subgradient non-unique.
        if gap < 1e-6:
            return _spectral_radius_gradient_fd(J, fd_eps)

    left_vals, left_vecs = np.linalg.eig(J.T)
    left_idx = int(np.argmin(np.abs(left_vals - lam)))
    v = right_vecs[:, dom_idx]
    w = left_vecs[:, left_idx]
    denom = np.vdot(w, v)  # conjugate(w) dot v
    if abs(denom) < 1e-10 or not np.isfinite(denom):
        return _spectral_radius_gradient_fd(J, fd_eps)

    # For real perturbations dJ, d lambda = conjugate(w)^T dJ v / conjugate(w)^T v.
    # d |lambda| = Re(conjugate(lambda) / |lambda| * d lambda).
    grad_lam = np.outer(np.conjugate(w), v) / denom
    grad_rho = np.real((np.conjugate(lam) / rho) * grad_lam)
    if not np.all(np.isfinite(grad_rho)):
        return _spectral_radius_gradient_fd(J, fd_eps)
    return grad_rho.astype(float, copy=False)


def _spectral_radius_gradient_fd(J: np.ndarray, fd_eps: float = 1e-7) -> np.ndarray:
    """Finite-difference fallback for spectral-radius gradient."""
    rho = float(np.max(np.abs(np.linalg.eigvals(J))))
    grad = np.zeros_like(J, dtype=float)
    h = float(fd_eps) if fd_eps > 0 else 1e-7
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            J_perturb = J.copy()
            J_perturb[i, j] += h
            rho_perturb = float(np.max(np.abs(np.linalg.eigvals(J_perturb))))
            grad[i, j] = (rho_perturb - rho) / h
    return grad
