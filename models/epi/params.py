from __future__ import annotations

"""Epidemic model parameters, inputs and outputs."""

from dataclasses import dataclass

import numpy as np

POST_YEAR = 2021        # the diagnosis multiplier (post_multiplier) applies from this year


@dataclass
class EpiParams:
    """Epidemic-model (hierarchical calibration) parameters for one geography and draw."""

    beta: float
    kdx: float
    U0: float
    alpha: float
    risk0: float = 1.0
    post_multiplier: float = 1.0


@dataclass
class EpiInputs:
    """Annual inputs: HIV testing, PrEP use and risk behaviour from the dynamic model; PrEP-eligible and not
    virally suppressed counts from CDC (models/epi/input_adapter.py)."""
    years: np.ndarray
    hiv_testing: np.ndarray
    prep_on: np.ndarray
    N_elig: np.ndarray
    risk_behavior: np.ndarray
    no_vs: np.ndarray


@dataclass
class EpiOutput:
    """Epidemic-model output for one geography and draw."""
    unit_id: str
    years: np.ndarray
    prep_on_count: np.ndarray
    incidence: np.ndarray
    diagnosed: np.ndarray
    undiagnosed: np.ndarray
