"""Transition-matrix (J) and drift estimation for one geographic unit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from config.optimization import OptimConfig
from .loss import joint_loss, rebuild_from_theta


def get_bounds(sign_mask, M, bounds_range):
    """Bounds on the free entries of J - I (the optimized parameters).

    Off-diagonal entries (= J): ``bounds_range``, with sign-constrained pathways bounded at 0 on the
    other side. Diagonal entries: J - I in [-1, bounds_range[1]], i.e. persistence J in [0, 1 + bounds_range[1]]
    ([0, 3] with the default (-2, 2)): non-negative (no oscillation) before the stability penalty acts.
    """
    n_free = int(np.sum(M))
    sign_free = sign_mask[M == 1]
    diag_free = np.eye(M.shape[0], dtype=bool)[M == 1]

    LB = np.full(n_free, bounds_range[0], dtype=float)
    UB = np.full(n_free, bounds_range[1], dtype=float)

    LB[sign_free > 0] = 0
    UB[sign_free < 0] = 0
    LB[diag_free] = np.maximum(LB[diag_free], -1.0)

    return LB, UB


class CachedLossFunction:
    """Cache loss/gradient evaluations at the same optimizer point."""

    def __init__(self, loss_fn):
        self._loss_fn = loss_fn
        self._cache_key: bytes | None = None
        self._cache_value = None

    def __call__(self, theta: np.ndarray):
        key = np.asarray(theta, dtype=float).tobytes()
        if key != self._cache_key:
            self._cache_key = key
            self._cache_value = self._loss_fn(theta)
        return self._cache_value

    def loss(self, theta: np.ndarray) -> float:
        return self(theta)[0]

    def grad(self, theta: np.ndarray) -> np.ndarray:
        return self(theta)[1]


@dataclass(frozen=True)
class JacobianFit:
    """Result of one unit's fit: transition matrix, drift (all measures; zero where not enabled) and
    optimizer diagnostics."""
    J: np.ndarray
    drift: np.ndarray
    diagnostics: dict


class JacobianEstimator:
    """Estimates the transition matrix J and the enabled drift terms c for one unit.

    Minimizes the joint loss (models/dynamic/estimation/loss.py) over all observed
    periods with L-BFGS-B; sign constraints enter as bounds on J - I. The loss can
    have more than one local minimum, so the fit is run from several deterministic
    starting points and the lowest loss is kept (``_starting_points``).
    """

    def __init__(self, optconfig: OptimConfig):
        self.opt = optconfig

    def fit(
        self,
        Xbar: np.ndarray,
        SigmaX: np.ndarray,
        M: np.ndarray,
        sign_mask: np.ndarray,
        u: np.ndarray,
        J_ref: np.ndarray | None = None,
        wJ: float | None = None,
        structural_prior_mean: np.ndarray | None = None,
        structural_prior_tau: np.ndarray | None = None,
        drift_enabled: np.ndarray | None = None,
        drift_prior_mean: np.ndarray | None = None,
        drift_prior_tau: np.ndarray | None = None,
        evidence_target: dict | None = None,
    ) -> JacobianFit:
        """Fit one unit; return its J (m x m), drift and diagnostics."""
        m = M.shape[0]
        Sigma_eta = (self.opt.sigma_eta_scale ** 2) * np.diag(np.diag(SigmaX))
        drift_enabled = np.zeros(m, dtype=bool) if drift_enabled is None else np.asarray(drift_enabled, dtype=bool)
        if drift_prior_mean is None:
            drift_prior_mean = np.asarray(u, dtype=float)
        if drift_prior_tau is None:
            drift_prior_tau = np.full(m, np.nan, dtype=float)

        def loss_fn(theta_free):
            return joint_loss(
                theta_free, Xbar, SigmaX, Sigma_eta,
                self.opt.wR, self.opt.wX, u, M,
                J_ref, wJ if wJ is not None else 0.0,
                w_stab=self.opt.w_stab,
                eps=self.opt.stability_eps,
                stability_fd_eps=self.opt.stability_fd_eps,
                return_components=True,
                structural_prior_mean=structural_prior_mean,
                structural_prior_tau=structural_prior_tau,
                drift_enabled=drift_enabled,
                drift_prior_mean=drift_prior_mean,
                drift_prior_tau=drift_prior_tau,
                evidence_target=evidence_target,
            )

        LB, UB = get_bounds(sign_mask, M, self.opt.bounds)
        n_drift = int(drift_enabled.sum())
        lower = np.concatenate([LB, np.full(n_drift, -self.opt.drift_bounds)])
        upper = np.concatenate([UB, np.full(n_drift, self.opt.drift_bounds)])
        drift0 = np.asarray(drift_prior_mean, dtype=float)[drift_enabled]
        cached_loss = CachedLossFunction(loss_fn)
        results = []
        for JmI0 in self._starting_points(m, sign_mask, J_ref, structural_prior_mean):
            theta0 = np.clip(np.concatenate([JmI0[M == 1], drift0]), lower, upper)
            results.append(minimize(
                cached_loss.loss, theta0, method="L-BFGS-B", jac=cached_loss.grad, bounds=list(zip(lower, upper)),
                options={"maxiter": self.opt.maxiter, "ftol": self.opt.ftol, "gtol": self.opt.gtol,
                         "maxls": self.opt.maxls},
            ))
        best = int(np.argmin([r.fun for r in results]))
        result = results[best]

        J_est, c_est = rebuild_from_theta(result.x, M, u, drift_enabled)
        final_loss, _, final_components = cached_loss(result.x)
        diagnostics = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "n_iter": int(result.nit),
            "final_components": final_components,
            "final_loss": float(final_loss),
            "best_start": best,
            "start_losses": [float(r.fun) for r in results],
        }
        return JacobianFit(J=J_est, drift=c_est, diagnostics=diagnostics)

    def _starting_points(self, m: int, sign_mask: np.ndarray, J_ref: np.ndarray | None,
                         prior_mean: np.ndarray | None) -> list[np.ndarray]:
        """J - I starting points: the default start, the prior means, and the parent unit's J if any."""
        default = self._initialize_JmI0(m, sign_mask)
        prior = default.copy()
        if prior_mean is not None:
            finite = np.isfinite(prior_mean)
            prior[finite] = (np.asarray(prior_mean, dtype=float) - np.eye(m))[finite]
        starts = [default, prior]
        if J_ref is not None:
            starts.append(np.asarray(J_ref, dtype=float) - np.eye(m))
        return starts

    def _initialize_JmI0(self, m: int, sign_mask: np.ndarray) -> np.ndarray:
        """Deterministic start: diagonal J in (0.5, 1], off-diagonal entries small with the permitted sign."""
        rng = np.random.default_rng(seed=0)
        JmI0 = -self.opt.B0_diag * np.diag(rng.random(m))
        n_pos = int(np.sum(sign_mask == 1))
        n_neg = int(np.sum(sign_mask == -1))
        if n_pos > 0:
            JmI0[sign_mask == 1] = self.opt.B0_off_diag * rng.random(n_pos)
        if n_neg > 0:
            JmI0[sign_mask == -1] = -self.opt.B0_off_diag * rng.random(n_neg)
        return JmI0
