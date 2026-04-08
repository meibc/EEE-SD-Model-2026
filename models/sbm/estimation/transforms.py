from core.math.transforms import Transforms
import numpy as np


class FeatureTransformer:
    def __init__(self, eps=1e-2):
        self.transforms = Transforms()
        self.eps = eps

    def transform(self, Ybar, SigmaY):
        Ybar = np.clip(Ybar, self.eps, 1 - self.eps)

        Xbar = self.transforms.logit(Ybar, self.eps)

        p = np.mean(Ybar, axis=1)
        Jlogit = self.transforms.logit_jacobian(p, self.eps)

        SigmaX = Jlogit @ SigmaY @ Jlogit

        return Xbar, SigmaX
