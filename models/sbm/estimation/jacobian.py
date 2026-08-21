from __future__ import annotations

"""Jacobian matrix estimation."""

import numpy as np
from scipy.optimize import minimize

from config.optimization import OptimConfig
from .loss import joint_loss, rebuild_from_theta
from .constraints import get_bounds

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



class JacobianEstimator:
    """Estimates J matrices and optional selected drift vector c."""

    def __init__(self, optconfig: OptimConfig):
        self.opt = optconfig
        self.last_fit_history: list[dict] = []
        self.last_drift_sequence: np.ndarray | None = None

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
    ) -> np.ndarray:
        """Estimate rolling J matrix sequence; fitted drift is in last_drift_sequence."""
        m, T = Xbar.shape
        J_estimates = np.repeat(np.eye(m)[:, :, None], T, axis=2)
        drift_estimates = np.repeat(np.asarray(u, dtype=float)[:, None], T, axis=1)
        JmI0 = None
        self.last_fit_history = []
        self.last_drift_sequence = drift_estimates.copy()

        for t in range(2, T + 1):
            if self.opt.flag_reinitialize_JmI0:
                JmI0 = None
            try:
                J_est, c_est, diagnostics = self._optimize(
                    Xbar[:, :t],
                    SigmaX,
                    M,
                    sign_mask,
                    u,
                    JmI0,
                    J_ref,
                    wJ,
                    structural_prior_mean,
                    structural_prior_tau,
                    drift_enabled,
                    drift_prior_mean,
                    drift_prior_tau,
                )
            except Exception as e:
                print(f"Optimization failed: {e}")
                JmI0 = None
                continue

            J_estimates[:, :, t - 1] = J_est
            drift_estimates[:, t - 1] = c_est
            JmI0 = J_est - np.eye(m)
            self.last_fit_history.append(
                {
                    "t_end": int(t),
                    "n_obs_steps": int(t - 1),
                    "diagnostics": diagnostics,
                }
            )

        self.last_drift_sequence = drift_estimates
        return J_estimates

    def _optimize(
        self,
        Xbar: np.ndarray,
        SigmaX: np.ndarray,
        M: np.ndarray,
        sign_mask: np.ndarray,
        u: np.ndarray,
        JmI0: np.ndarray | None,
        J_ref: np.ndarray | None,
        wJ: float | None,
        structural_prior_mean: np.ndarray | None,
        structural_prior_tau: np.ndarray | None,
        drift_enabled: np.ndarray | None,
        drift_prior_mean: np.ndarray | None,
        drift_prior_tau: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        m = M.shape[0]
        Sigma_eta = (self.opt.sigma_eta_scale ** 2) * np.diag(np.diag(SigmaX))

        if JmI0 is None:
            JmI0 = self._initialize_JmI0(m, sign_mask)

        JmI_vec0 = JmI0[M == 1]
        drift_enabled_use = np.zeros(m, dtype=bool) if drift_enabled is None else np.asarray(drift_enabled, dtype=bool)
        if drift_prior_mean is None:
            drift_prior_mean = np.asarray(u, dtype=float)
        if drift_prior_tau is None:
            drift_prior_tau = np.full(m, np.nan, dtype=float)
        c_vec0 = np.asarray(drift_prior_mean, dtype=float)[drift_enabled_use]
        theta0 = np.concatenate([JmI_vec0, c_vec0])

        x_weights = None
        if self.opt.use_diag_whitening_x:
            diag = np.diag(SigmaX).astype(float)
            x_weights = 1.0 / np.maximum(diag, self.opt.norm_eps)
        wJ_use = wJ if wJ is not None else 0.0

        _, _, base_components = joint_loss(
            theta0, Xbar, SigmaX, Sigma_eta,
            self.opt.wR, self.opt.wX, u, M,
            J_ref, wJ_use,
            w_stab=self.opt.w_stab,
            eps=self.opt.stability_eps,
            stability_fd_eps=self.opt.stability_fd_eps,
            x_weights=x_weights,
            normalize_x_by_counts=self.opt.normalize_x_by_counts,
            normalize_r_by_counts=self.opt.normalize_r_by_counts,
            component_scales=None,
            norm_eps=self.opt.norm_eps,
            return_components=True,
            structural_prior_mean=structural_prior_mean,
            structural_prior_tau=structural_prior_tau,
            drift_enabled=drift_enabled_use,
            drift_prior_mean=drift_prior_mean,
            drift_prior_tau=drift_prior_tau,
        )
        component_scales = None
        if self.opt.normalize_components_by_baseline:
            component_scales = {
                "LX": max(base_components["LX"], self.opt.norm_eps),
                "LR": max(base_components["LR"], self.opt.norm_eps),
                "LJ": max(base_components["LJ"], self.opt.norm_eps),
                "L_prior_J": max(base_components["L_prior_J"], self.opt.norm_eps),
                "L_prior_c": max(base_components["L_prior_c"], self.opt.norm_eps),
                "L_stab": max(base_components["L_stab"], self.opt.norm_eps),
            }

        history: list[dict[str, float]] = []

        def loss_fn(theta_free):
            return joint_loss(
                theta_free, Xbar, SigmaX, Sigma_eta,
                self.opt.wR, self.opt.wX, u, M,
                J_ref, wJ_use,
                w_stab=self.opt.w_stab,
                eps=self.opt.stability_eps,
                stability_fd_eps=self.opt.stability_fd_eps,
                x_weights=x_weights,
                normalize_x_by_counts=self.opt.normalize_x_by_counts,
                normalize_r_by_counts=self.opt.normalize_r_by_counts,
                component_scales=component_scales,
                norm_eps=self.opt.norm_eps,
                return_components=True,
                structural_prior_mean=structural_prior_mean,
                structural_prior_tau=structural_prior_tau,
                drift_enabled=drift_enabled_use,
                drift_prior_mean=drift_prior_mean,
                drift_prior_tau=drift_prior_tau,
            )

        def callback(xk):
            Lk, _, ck = cached_loss(xk)
            row = dict(ck)
            row["iter"] = float(len(history))
            row["L_total_callback"] = float(Lk)
            history.append(row)

        LB, UB = get_bounds(sign_mask, M, self.opt.bounds)
        bounds = list(zip(LB, UB))
        drift_bound = float(getattr(self.opt, "drift_bounds", 1.5))
        bounds.extend([(-drift_bound, drift_bound)] * int(drift_enabled_use.sum()))

        cached_loss = CachedLossFunction(loss_fn)
        result = minimize(
            cached_loss.loss,
            theta0,
            method="L-BFGS-B",
            jac=cached_loss.grad,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": self.opt.maxiter,
                "ftol": self.opt.ftol,
                "gtol": self.opt.gtol,
                "maxls": self.opt.maxls,
            },
        )

        J_est, c_est = rebuild_from_theta(result.x, M, u, drift_enabled_use)
        final_loss, _, final_components = cached_loss(result.x)
        diagnostics = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "n_iter": int(result.nit),
            "n_func_eval": int(result.nfev),
            "n_grad_eval": int(getattr(result, "njev", -1)),
            "history": history,
            "final_components": final_components,
            "final_loss": float(final_loss),
            "component_scales": component_scales or {},
            "drift": c_est.tolist(),
        }
        return J_est, c_est, diagnostics

    def _initialize_JmI0(self, m, sign_mask):
        rng = np.random.default_rng(seed=0)
        JmI0 = -self.opt.B0_diag * np.diag(rng.random(m))
        n_pos = int(np.sum(sign_mask == 1))
        n_neg = int(np.sum(sign_mask == -1))
        if n_pos > 0:
            JmI0[sign_mask == 1] = self.opt.B0_off_diag * rng.random(n_pos)
        if n_neg > 0:
            JmI0[sign_mask == -1] = -self.opt.B0_off_diag * rng.random(n_neg)
        return JmI0
