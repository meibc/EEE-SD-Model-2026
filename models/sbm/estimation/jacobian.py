"""Jacobian matrix estimation."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from config.optimization import OptimConfig
from .loss import joint_loss
from .constraints import get_bounds

class JacobianEstimator:
    """Estimates J matrices from observed trajectories."""

    def __init__(self, optconfig: OptimConfig):
        self.opt = optconfig
        self.last_fit_history: list[dict] = []


    def fit(
        self,
        Xbar: np.ndarray,
        SigmaX: np.ndarray,
        M: np.ndarray,
        sign_mask: np.ndarray,
        u: np.ndarray,
        J_ref: np.ndarray | None = None,
        wJ: float | None = None,
    ) -> np.ndarray:
        """
        Estimate J matrix sequence using rolling optimization

        Returns
        -------
        J_estimates : (m, m, T) array
        """
        m, T = Xbar.shape
        J_estimates = np.repeat(np.eye(m)[:, :, None], T, axis=2)  # Identity repeated T times
        JmI0 = None
        self.last_fit_history = []

        for t in range(2, T + 1):
            if self.opt.flag_reinitialize_JmI0:
                JmI0 = None  # re-initialize JmI0 for each t if flag is set                
            try:
                J_est, diagnostics = self._optimize(
                    Xbar[:, :t], SigmaX, M, sign_mask, u, JmI0, J_ref, wJ
                )
            except Exception as e:
                print(f"Optimization failed: {e}")
                JmI0 = None  # reset JmI0 on failure 
                continue

            J_estimates[:, :, t-1] = J_est
            JmI0 = J_est - np.eye(m)  # warm start for next t
            self.last_fit_history.append(
                {
                    "t_end": int(t),
                    "n_obs_steps": int(t - 1),
                    "diagnostics": diagnostics,
                }
            )

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
    ) -> tuple[np.ndarray, dict]:
        """Single optimization run."""
        m = M.shape[0]

        # establish Sigma_eta as scaled diagonal of SigmaX
        Sigma_eta = (self.opt.sigma_eta_scale ** 2) * np.diag(np.diag(SigmaX))

        # Initialize J0 if not provided
        if JmI0 is None:
            JmI0 = self._initialize_JmI0(m, sign_mask)

        # initial free parameters as vector
        JmI_vec0 = JmI0[M == 1]
        x_weights = None
        if self.opt.use_diag_whitening_x:
            diag = np.diag(SigmaX).astype(float)
            x_weights = 1.0 / np.maximum(diag, self.opt.norm_eps)
        # Shrinkage weights are owned by ShrinkageConfig and should be passed in.
        # If no shrinkage is requested for this unit, use zero penalty.
        wJ_use = wJ if wJ is not None else 0.0

        _, _, base_components = joint_loss(
            JmI_vec0, Xbar, SigmaX, Sigma_eta,
            self.opt.wR, self.opt.wX, u, M,
            J_ref, wJ_use,
            x_weights=x_weights,
            normalize_x_by_counts=self.opt.normalize_x_by_counts,
            normalize_r_by_counts=self.opt.normalize_r_by_counts,
            component_scales=None,
            norm_eps=self.opt.norm_eps,
            return_components=True,
        )
        component_scales = None
        if self.opt.normalize_components_by_baseline:
            component_scales = {
                "LX": max(base_components["LX"], self.opt.norm_eps),
                "LR": max(base_components["LR"], self.opt.norm_eps),
                "LJ": max(base_components["LJ"], self.opt.norm_eps),
                "L_stab": max(base_components["L_stab"], self.opt.norm_eps),
            }

        # Define the loss function for optimization
        history: list[dict[str, float]] = []

        def loss_fn(JmI_vec_free):
            return joint_loss(
                JmI_vec_free, Xbar, SigmaX, Sigma_eta,
                self.opt.wR, self.opt.wX, u, M,
                J_ref, wJ_use,
                x_weights=x_weights,
                normalize_x_by_counts=self.opt.normalize_x_by_counts,
                normalize_r_by_counts=self.opt.normalize_r_by_counts,
                component_scales=component_scales,
                norm_eps=self.opt.norm_eps,
                return_components=True,
            )

        def callback(xk):
            Lk, _, ck = loss_fn(xk)
            row = dict(ck)
            row["iter"] = float(len(history))
            row["L_total_callback"] = float(Lk)
            history.append(row)

        # Get bounds for free parameters based on sign constraints
        LB, UB = get_bounds(sign_mask, M, self.opt.bounds)
        bounds = list(zip(LB, UB))

        # normal optimization with L-BFGS-B
        result = minimize(
            lambda x: loss_fn(x)[0],
            JmI_vec0,
            method="L-BFGS-B",
            jac=lambda x: loss_fn(x)[1],
            bounds=bounds,
            callback=callback,
            options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6},
        )

        # rebuild J matrix from optimized free parameters
        JmI = np.zeros((m, m))
        JmI[M == 1] = result.x
        J_est = np.eye(m) + JmI

        final_loss, _, final_components = loss_fn(result.x)
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
        }

        return J_est, diagnostics
    
    def _initialize_JmI0(self, m, sign_mask):
        """Initialize JmI0 based on sign constraints."""
        rng = np.random.default_rng(seed=0) # fixed seed for reproducibility

        # Start with a negative diagonal for stability        
        JmI0 = -self.opt.B0_diag * np.diag(rng.random(m))

        
        n_pos = int(np.sum(sign_mask == 1))
        n_neg = int(np.sum(sign_mask == -1))

        # fills off-diagonal entries respecting sign constraints with random values scaled by configured B0_off_diag
        if n_pos > 0:
            JmI0[sign_mask == 1] = self.opt.B0_off_diag * rng.random(n_pos)
        if n_neg > 0:
            JmI0[sign_mask == -1] = -self.opt.B0_off_diag * rng.random(n_neg)

        return JmI0
    
