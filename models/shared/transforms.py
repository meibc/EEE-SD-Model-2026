# models/shared/transforms.py
"""Transformations between model spaces."""

import numpy as np


def hazard_proxy(
    p_test: np.ndarray,
    eps: float = 1e-6,
    gamma_tau: float = 1.0
) -> np.ndarray:
    """
    Transform testing probability to hazard proxy.
    
    tau = gamma_tau * -log(1 - p_test)
    """
    p = np.clip(np.asarray(p_test, dtype=float), 0.0, 1.0 - eps)
    return gamma_tau * (-np.log1p(-p))