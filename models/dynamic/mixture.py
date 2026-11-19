"""The PrEP mixture, shared by the fit (pipeline/dynamic_fit.py) and the projection (pipeline/projection.py).

Population PrEP is a mixture over men with and without AHS: P = P0 (1 - A (1 - r)), with P0 PrEP use among men
without AHS, A the AHS prevalence and r the unit's access-adjusted AHS-PrEP prevalence ratio. The model's PrEP row
is P0 / ceiling. The PrEP -> AHS feedback b acts on population PrEP P, outside J.
"""
from __future__ import annotations

import numpy as np

from core.transforms import LOGIT_EPS, logit


def prep_without_ahs(prep: np.ndarray, ahs: np.ndarray, ratio: float) -> np.ndarray:
    """P0, PrEP use among men without AHS, from population PrEP P = P0 (1 - A (1 - r))."""
    return np.clip(np.asarray(prep, dtype=float) / (1 - np.asarray(ahs, dtype=float) * (1 - ratio)), 0.0, 0.999)


def prep_logit(prep: np.ndarray) -> np.ndarray:
    """Logit of population PrEP, clipped as in the fit (FeatureTransformer)."""
    return logit(prep, LOGIT_EPS)


def feedback_input(prep: np.ndarray, observed_prep: np.ndarray, b: float) -> np.ndarray:
    """The PrEP -> AHS feedback's input to AHS (centred logit) for population PrEP ``prep``: b x its centred logit,
    centred at the unit's mean logit of observed population PrEP (the state anchor, as for every measure). A known
    input in the fit (DynamicFitRunner.feedback_offsets) and applied each year in the projection."""
    return b * (prep_logit(prep) - prep_logit(observed_prep).mean())
