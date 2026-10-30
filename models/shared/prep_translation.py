"""Scenario PrEP from a change in anticipated healthcare stigma (AHS).

The dynamic model projects AHS under baseline and scenario; these functions set scenario PrEP from that AHS
change (pipeline/projection.py). Arrays are probabilities over the projection years.
"""
from __future__ import annotations

import numpy as np

from core.transforms import PROB_EPS, inverse_logit, logit


def prevalence_ratio_translation(baseline_prep: np.ndarray, baseline_ahs: np.ndarray, scenario_ahs: np.ndarray,
                                 ratio: float) -> np.ndarray:
    """Primary translation. Population PrEP is P = P0 (1 - A (1 - r)), with P0 PrEP among men without AHS
    and r the respondent-level prevalence ratio. Holding P0 at baseline and changing A gives
    P_s = P_b (1 - A_s (1 - r)) / (1 - A_b (1 - r))."""
    gap = 1 - ratio
    factor = (1 - scenario_ahs * gap) / (1 - baseline_ahs * gap)
    return np.clip(baseline_prep * factor, PROB_EPS, 1 - PROB_EPS)


def contextual_translation(baseline_prep: np.ndarray, baseline_ahs: np.ndarray, scenario_ahs: np.ndarray,
                           climate_ratio: float) -> np.ndarray:
    """Contextual upper bound: everyone's PrEP changes by the between-state climate prevalence ratio per
    10 percentage points of AHS, P_s = P_b x ratio ** ((A_s - A_b) / 0.10)."""
    factor = climate_ratio ** ((scenario_ahs - baseline_ahs) / 0.10)
    return np.clip(baseline_prep * factor, PROB_EPS, 1 - PROB_EPS)


def retain_prep_fraction(baseline_probs: np.ndarray, scenario_probs: np.ndarray, prep_idx: int,
                         fraction: float) -> np.ndarray:
    """Resilient delivery: keep ``fraction`` of the scenario's PrEP change (on the logit scale) and set every
    other measure, AHS included, to baseline. Arrays are measure x year."""
    lb = logit(baseline_probs[prep_idx], PROB_EPS)
    out = np.array(baseline_probs, dtype=float, copy=True)
    out[prep_idx] = inverse_logit(lb + fraction * (logit(scenario_probs[prep_idx], PROB_EPS) - lb))
    return out
