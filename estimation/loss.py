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
) -> tuple[float, np.ndarray]:
    """
    Combined loss: covariance + drift + shrinkage + stability.
    Note: Discrete
    Returns (loss, gradient w.r.t. free parameters).
    """
    J = _rebuild_J(JmI_vec_free, M)

    LR, gradR = _covariance_loss(J, Sigma, Sigma_eta, wR)
    LX, gradX = _mean_drift_loss(J, Xbar, u, wX)
    LJ, gradJ = _shrinkage_loss(J, J_ref, wJ)
    L_stab, grad_stab = _stability_penalty(J, w_stab, eps)

    L = LR + LX + LJ + L_stab
    grad_full = gradR + gradX + gradJ + grad_stab

    return L, grad_full[M == 1]

def _rebuild_J(JmI_vec_free: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Reconstruct J from free parameters."""
    m = M.shape[0]
    JmI = np.zeros((m, m))      # (J-I) matrix
    JmI[M == 1] = JmI_vec_free
    J = np.eye(m) + JmI         # J = I + (J-I)

    return J

def _covariance_loss(J, Sigma, Sigma_eta, wR) -> tuple[float, np.ndarray]:
        """Lyapunov loss: ||Sigma - J Sigma J' - Sigma_eta||^2, with gradient."""

        Sigma_eta_diag = np.diag(np.diag(Sigma_eta))

        ER = -Sigma + J @ Sigma @ J.T + Sigma_eta_diag
        gradR = 4 * wR * ER @ J @ Sigma   # NO Delta_t

        LR = wR * np.sum(ER ** 2)

        return LR, gradR

def _mean_drift_loss(J, Xbar, u, wX) -> tuple[float, np.ndarray]:
    """Mean drift loss: ||Xbar[:, t] - J Xbar[:, t-1] - u||^2, with gradient."""
    m, T = Xbar.shape
    LX = 0
    gradX = np.zeros((m, m))

    for t in range(1, T):
        EX = Xbar[:, t] - J @ Xbar[:, t-1] - u
        LX += wX * (EX.T @ EX)
        gradX -= 2 * wX * np.outer(EX, Xbar[:, t-1])

    return LX, gradX

def _shrinkage_loss(
    J: np.ndarray,
    J_ref: np.ndarray | None,
    wJ: float,
) -> tuple[float, np.ndarray]:
    """Shrinkage toward reference J."""
    if J_ref is None or wJ == 0:
        gradJ_shrink = np.zeros_like(J)
        LJ = 0
    else:
        EJ = J - J_ref
        gradJ_shrink = 2 * wJ * EJ
        LJ = wJ * np.sum(EJ ** 2)

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