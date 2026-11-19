from __future__ import annotations

import numpy as np


def joint_loss(
    theta_free: np.ndarray,
    Xbar: np.ndarray,
    wX: float,
    u: np.ndarray,
    M: np.ndarray,
    J_ref: np.ndarray | None,
    wJ: float,
    *,
    w_stab: float,
    eps: float,
    stability_fd_eps: float,
    return_components: bool = False,
    structural_prior_mean: np.ndarray | None = None,
    structural_prior_tau: np.ndarray | None = None,
    drift_enabled: np.ndarray | None = None,
    drift_prior_mean: np.ndarray | None = None,
    drift_prior_tau: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, dict[str, float]]:
    """Combined loss and its gradient:

        L = wX * LX + wJ * LJ + L_prior_J + L_prior_c + L_stab

    LX: trajectory-with-drift misfit; LJ: shrinkage toward J_ref; L_prior_J / L_prior_c: structural
    (pathway and persistence) and drift priors; L_stab: spectral-radius penalty. ``offset`` (measure x
    period, optional) is a known input added to each step, x(t+1) = J x(t) + c + offset(t): the held
    PrEP -> AHS feedback acting on population PrEP (pipeline/dynamic_fit.py). The parameter vector is [free J-I
    entries, enabled drift entries]. Weights and the stability threshold (1 - eps)
    come from config/optimization.py; J_ref is None (and wJ 0) for the nation.
    """
    n_j = int(np.sum(M))
    j_free = np.asarray(theta_free[:n_j], dtype=float)
    c_free = np.asarray(theta_free[n_j:], dtype=float)
    J = _rebuild_J(j_free, M)
    c = _rebuild_drift(c_free, np.asarray(u, dtype=float), drift_enabled)

    LX, gradX, gradc = _mean_drift_loss(J, Xbar, c, offset)
    LJ, gradJ = _shrinkage_loss(J, J_ref)
    L_prior_J, grad_prior_J = _structural_prior_loss(J, structural_prior_mean, structural_prior_tau, M)
    L_prior_c, grad_prior_c = _drift_prior_loss(c_free, drift_enabled, drift_prior_mean, drift_prior_tau)
    L_stab, grad_stab = _stability_penalty(J, w_stab, eps, stability_fd_eps)

    L = wX * LX + wJ * LJ + L_prior_J + L_prior_c + L_stab
    grad_full = wX * gradX + wJ * gradJ + grad_prior_J + grad_stab
    grad_free = grad_full[M == 1]
    if c_free.size:
        enabled = np.asarray(drift_enabled, dtype=bool)
        grad_free = np.concatenate([grad_free, wX * gradc[enabled] + grad_prior_c])

    if not return_components:
        return L, grad_free
    components = {"LX": float(LX), "LJ": float(LJ), "L_prior_J": float(L_prior_J),
                  "L_prior_c": float(L_prior_c), "L_stab": float(L_stab),
                  "L_total": float(L)}
    return L, grad_free, components


def rebuild_from_theta(theta_free: np.ndarray, M: np.ndarray, u: np.ndarray, drift_enabled: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    n_j = int(np.sum(M))
    return _rebuild_J(theta_free[:n_j], M), _rebuild_drift(np.asarray(theta_free[n_j:], dtype=float), u, drift_enabled)


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


def _mean_drift_loss(J, Xbar, c, offset=None) -> tuple[float, np.ndarray, np.ndarray]:
    m, T = Xbar.shape
    if offset is None:
        offset = np.zeros_like(Xbar)
    LX = 0.0
    gradX = np.zeros((m, m), dtype=float)
    gradc = np.zeros(m, dtype=float)
    for t in range(1, T):
        EX = Xbar[:, t] - J @ Xbar[:, t - 1] - c - offset[:, t - 1]
        LX += float(EX @ EX)
        gradX -= 2.0 * np.outer(EX, Xbar[:, t - 1])
        gradc -= 2.0 * EX
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

    The stability criterion for the transition matrix is max(abs(lambda)) < 1.
    The gradient uses the analytic left/right eigenvector derivative for the
    dominant eigenvalue, with finite-difference fallback when the dominant mode
    is nearly tied or poorly conditioned.

    A row that is a pure integrator (persistence 1 and no inputs: logistic diffusion, pipeline/dynamic_fit.py)
    has eigenvalue 1 by construction, and J's other eigenvalues are those of J without that row and column;
    the penalty then applies to the rest of J.
    """
    off = J - np.diag(np.diag(J))
    integrator = (np.abs(np.diag(J) - 1.0) < 1e-12) & (np.abs(off).sum(axis=1) < 1e-12)
    if integrator.any():
        keep = np.ix_(~integrator, ~integrator)
        penalty, grad_rest = _stability_penalty(J[keep], w_stab, eps, fd_eps)
        grad = np.zeros_like(J)
        grad[keep] = grad_rest
        return penalty, grad
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
