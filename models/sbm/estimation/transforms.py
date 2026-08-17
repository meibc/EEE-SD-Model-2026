import numpy as np
from core.math.transforms import Transforms


class FeatureTransformer:
    """Probability-space to centered-logit-space transformer.

    Forward transform:
        X = logit(Y) - logit(reference_probability)

    With reference_probability=0.5 this reduces to the old raw logit. The
    covariance Jacobian is unchanged by centering because subtracting a constant
    does not change derivatives.
    """

    def __init__(self, reference_probs=None, eps=1e-2):
        self.transforms = Transforms()
        self.eps = eps
        self.reference_probs = None if reference_probs is None else np.asarray(reference_probs, dtype=float)
        if self.reference_probs is not None:
            if self.reference_probs.ndim != 1:
                raise ValueError("reference_probs must be 1D")
            if not np.all((self.reference_probs > 0) & (self.reference_probs < 1)):
                raise ValueError("reference_probs must be in (0, 1)")
            self.reference_logits = self.transforms.logit(self.reference_probs, self.eps)
        else:
            self.reference_logits = None

    def _ref_logits_for(self, Ybar):
        if self.reference_logits is None:
            return np.zeros(Ybar.shape[0], dtype=float)
        if Ybar.shape[0] != self.reference_logits.size:
            raise ValueError(f"reference_probs length mismatch: {self.reference_logits.size} vs {Ybar.shape[0]}")
        return self.reference_logits

    def transform(self, Ybar, SigmaY=None):
        Ybar = np.clip(Ybar, self.eps, 1 - self.eps)
        ref = self._ref_logits_for(Ybar)
        Xbar = self.transforms.logit(Ybar, self.eps) - ref[:, None]

        if SigmaY is None:
            return Xbar, None

        p = np.mean(Ybar, axis=1)
        Jlogit = self.transforms.logit_jacobian(p, self.eps)
        SigmaX = Jlogit @ SigmaY @ Jlogit
        return Xbar, SigmaX

    def inverse_transform(self, X):
        arr = np.asarray(X, dtype=float)
        ref = self.reference_logits
        if ref is None:
            ref = np.zeros(arr.shape[0], dtype=float)
        shifted = arr + ref.reshape((ref.size,) + (1,) * (arr.ndim - 1))
        return self.transforms.inverse_logit(shifted)
