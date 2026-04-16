# models/shared/transforms.py
"""Transformations between model spaces."""

import numpy as np


def hazard_proxy(
    p_test: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Transform testing probability to hazard proxy.
    
    tau = -log(1 - p_test)
    Note: could add gamma_tau scaling later
    """
    p = np.clip(np.asarray(p_test, dtype=float), 0.0, 1.0 - eps)
    return -np.log1p(-p)