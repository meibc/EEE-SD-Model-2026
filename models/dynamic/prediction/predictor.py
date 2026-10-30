"""Prediction functions."""
import numpy as np
from core.transforms import LOGIT_EPS, inverse_logit, logit


class Predictor:
    def __init__(self, reference_probs=None, reference_logits=None):
        self.reference_probs = None if reference_probs is None else np.asarray(reference_probs, dtype=float)
        self.reference_logits = None
        if reference_logits is not None:
            self.reference_logits = np.asarray(reference_logits, dtype=float)
        elif self.reference_probs is not None:
            self.reference_logits = logit(self.reference_probs, LOGIT_EPS)

    def inverse_x(self, X):
        """Probabilities from centred logits."""
        arr = np.asarray(X, dtype=float)
        if self.reference_logits is None:
            return inverse_logit(arr)
        ref = self.reference_logits.reshape((self.reference_logits.size,) + (1,) * (arr.ndim - 1))
        return inverse_logit(arr + ref)

    def predict_next(self, J, X_current, u):
        return J @ X_current + u

    def predict_trajectory(
        self,
        J,
        X0,
        u,
        n_steps: int,
        state_interventions=None,
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
        for t in range(t_start, n_steps):
            X_next = J @ Xpred[:, t - 1] + u
            if state_interventions:
                for iv in state_interventions:
                    baseline_p_t = None if baseline_probs is None else baseline_probs[:, t]
                    X_next = iv.apply(X_next, t, predictor=self, baseline_p=baseline_p_t)
            Xpred[:, t] = X_next

        return self.inverse_x(Xpred), Xpred
