import numpy as np
from core.transforms import LOGIT_EPS, inverse_logit, logit, logit_jacobian


class FeatureTransformer:
    """Probability-space to centered-logit-space transformer.

    Forward transform:
        X = logit(Y) - reference_logit

    The reference is either given as probabilities or directly as logits; the
    state-anchored model uses each unit's own mean logit over the observed periods
    (``state_anchor``). The covariance Jacobian is unchanged by centering because
    subtracting a constant does not change derivatives.
    """

    def __init__(self, reference_probs=None, eps=LOGIT_EPS, reference_logits=None):
        self.eps = eps
        self.reference_probs = None if reference_probs is None else np.asarray(reference_probs, dtype=float)
        if reference_logits is not None:
            self.reference_logits = np.asarray(reference_logits, dtype=float)
            if self.reference_logits.ndim != 1:
                raise ValueError("reference_logits must be 1D")
        elif self.reference_probs is not None:
            if self.reference_probs.ndim != 1:
                raise ValueError("reference_probs must be 1D")
            if not np.all((self.reference_probs > 0) & (self.reference_probs < 1)):
                raise ValueError("reference_probs must be in (0, 1)")
            self.reference_logits = logit(self.reference_probs, self.eps)
        else:
            self.reference_logits = None

    @staticmethod
    def state_anchor(Ybar, eps=LOGIT_EPS):
        """A unit's anchor: the mean logit of each measure over its observed periods (m,)."""
        Y = np.clip(np.asarray(Ybar, dtype=float), eps, 1 - eps)
        return logit(Y, eps).mean(axis=1)

    def _ref_logits_for(self, Ybar):
        if self.reference_logits is None:
            return np.zeros(Ybar.shape[0], dtype=float)
        if Ybar.shape[0] != self.reference_logits.size:
            raise ValueError(f"reference_probs length mismatch: {self.reference_logits.size} vs {Ybar.shape[0]}")
        return self.reference_logits

    def transform(self, Ybar, SigmaY=None):
        Ybar = np.clip(Ybar, self.eps, 1 - self.eps)
        ref = self._ref_logits_for(Ybar)
        Xbar = logit(Ybar, self.eps) - ref[:, None]

        if SigmaY is None:
            return Xbar, None

        p = np.mean(Ybar, axis=1)
        Jlogit = logit_jacobian(p, self.eps)
        SigmaX = Jlogit @ SigmaY @ Jlogit
        return Xbar, SigmaX

    def inverse_transform(self, X):
        arr = np.asarray(X, dtype=float)
        ref = self.reference_logits
        if ref is None:
            ref = np.zeros(arr.shape[0], dtype=float)
        shifted = arr + ref.reshape((ref.size,) + (1,) * (arr.ndim - 1))
        return inverse_logit(shifted)
