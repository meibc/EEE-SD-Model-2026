"""Probability transforms and the two clipping bounds shared across the workflow.

``LOGIT_EPS`` is the substantive model-scale bound applied to survey proportions before fitting or
projection. ``PROB_EPS`` is only a numerical bound that keeps stored and modified probabilities inside
the open interval (0, 1).
"""
import numpy as np
from scipy.special import expit

LOGIT_EPS = 1e-2
PROB_EPS = 1e-6


def logit(p, eps):
    """Return logit(p) after clipping p to [eps, 1 - eps]."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def inverse_logit(x):
    """Return the logistic inverse of x."""
    return expit(x)


def logit_jacobian(p, eps):
    """Return the diagonal Jacobian of logit at p, after clipping."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.diag(1 / p / (1 - p))
