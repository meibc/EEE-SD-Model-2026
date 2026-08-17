"""Prediction functions."""
import numpy as np
from core.math.transforms import Transforms


class Predictor:
    def __init__(self, reference_probs=None):
        self.transforms = Transforms()
        self.reference_probs = None if reference_probs is None else np.asarray(reference_probs, dtype=float)
        self.reference_logits = None
        if self.reference_probs is not None:
            self.reference_logits = self.transforms.logit(self.reference_probs)

    def _inverse_x(self, X):
        arr = np.asarray(X, dtype=float)
        if self.reference_logits is None:
            return self.transforms.inverse_logit(arr)
        ref = self.reference_logits.reshape((self.reference_logits.size,) + (1,) * (arr.ndim - 1))
        return self.transforms.inverse_logit(arr + ref)

    def predict_next(self, J, X_current, u):
        return J @ X_current + u

    def predict_rolling(self, J_sequence, Xbar, u) -> tuple[np.ndarray, np.ndarray]:
        m, T = Xbar.shape
        Xpred = np.full((m, T + 1), np.nan)
        t_start = 2
        Xpred[:, :t_start] = Xbar[:, :t_start]
        for t in range(2, T + 1):
            J_t = J_sequence[:, :, t - 1]
            Xpred[:, t] = self.predict_next(J_t, Xbar[:, t - 1], u)
        return self._inverse_x(Xpred), Xpred

    def predict_trajectory(
        self,
        J,
        X0,
        u,
        n_steps: int,
        state_interventions=None,
        rel_interventions=None,
        baseline_probs=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        m = X0.shape[0]
        Xpred = np.zeros((m, n_steps), dtype=float)
        Xpred[:, 0] = X0
        J_base = J.copy()

        for t in range(1, n_steps):
            J_t = J_base.copy()
            if rel_interventions:
                for iv in rel_interventions:
                    J_t = iv.apply(J_t, t)
            X_next = J_t @ Xpred[:, t - 1] + u
            if state_interventions:
                for iv in state_interventions:
                    baseline_p_t = None if baseline_probs is None else baseline_probs[:, t]
                    X_next = iv.apply(X_next, Xpred[:, t - 1], t, predictor=self, baseline_p=baseline_p_t)
            Xpred[:, t] = X_next

        return self._inverse_x(Xpred), Xpred
