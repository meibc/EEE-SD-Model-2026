"""Prediction functions."""
import numpy as np
from core.math.transforms import Transforms


class Predictor:
    def __init__(self, reference_probs=None, reference_logits=None):
        self.transforms = Transforms()
        self.reference_probs = None if reference_probs is None else np.asarray(reference_probs, dtype=float)
        self.reference_logits = None
        self.current_J = None   # transition matrix of the step being taken (read by shift interventions)
        if reference_logits is not None:
            self.reference_logits = np.asarray(reference_logits, dtype=float)
        elif self.reference_probs is not None:
            self.reference_logits = self.transforms.logit(self.reference_probs)

    def _inverse_x(self, X):
        arr = np.asarray(X, dtype=float)
        if self.reference_logits is None:
            return self.transforms.inverse_logit(arr)
        ref = self.reference_logits.reshape((self.reference_logits.size,) + (1,) * (arr.ndim - 1))
        return self.transforms.inverse_logit(arr + ref)

    def predict_next(self, J, X_current, u):
        return J @ X_current + u

    def predict_trajectory(
        self,
        J,
        X0,
        u,
        n_steps: int,
        state_interventions=None,
        rel_interventions=None,
        baseline_probs=None,
        fixed_prefix=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate X[t+1] = J X[t] + u from X0.

        If ``fixed_prefix`` (m x k, centered logits) is given, the first k
        columns are held at those values and propagation starts from column k-1.
        """
        m = X0.shape[0]
        Xpred = np.zeros((m, n_steps), dtype=float)
        Xpred[:, 0] = X0
        t_start = 1
        if fixed_prefix is not None:
            prefix = np.asarray(fixed_prefix, dtype=float)
            t_start = min(prefix.shape[1], n_steps)
            Xpred[:, :t_start] = prefix[:, :t_start]
        J_base = J.copy()

        for t in range(t_start, n_steps):
            J_t = J_base.copy()
            if rel_interventions:
                for iv in rel_interventions:
                    J_t = iv.apply(J_t, t)
            X_next = J_t @ Xpred[:, t - 1] + u
            self.current_J = J_t
            if state_interventions:
                for iv in state_interventions:
                    baseline_p_t = None if baseline_probs is None else baseline_probs[:, t]
                    X_next = iv.apply(X_next, Xpred[:, t - 1], t, predictor=self, baseline_p=baseline_p_t)
            Xpred[:, t] = X_next

        return self._inverse_x(Xpred), Xpred
