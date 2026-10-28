"""Logit transforms shared by the model, and the two clipping bounds it uses.

LOGIT_EPS  model scale: observed proportions are clipped to [0.01, 0.99] before they are put on the centred
           logit scale (fit, projection start), so a near-0 or near-1 survey estimate cannot dominate.
PROB_EPS   numerical bound: probabilities are kept strictly inside (0, 1) where they are stored or changed
           (input draws, projected probabilities, scenario changes), only to keep logits finite.
"""
import numpy as np
from scipy.special import expit

LOGIT_EPS = 1e-2
PROB_EPS = 1e-6


def logit(p, eps):
    """logit(p) after clipping p to [eps, 1 - eps]."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def inverse_logit(x):
    return expit(x)


def logit_jacobian(p, eps):
    """Diagonal Jacobian of logit at p (clipped to [eps, 1 - eps])."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.diag(1 / p / (1 - p))
