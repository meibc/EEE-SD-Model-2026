from __future__ import annotations

"""Containers for model inputs and simulation outputs."""

import numpy as np
from dataclasses import dataclass


@dataclass
class PreparedData:
    """Inputs for estimation."""
    units: list
    sign_mask: np.ndarray
    SigmaY: np.ndarray
    M: np.ndarray
    ts: np.ndarray
    v_names: list[str]


@dataclass
class CDCInputs:
    years: np.ndarray
    tau: np.ndarray
    prep_on: np.ndarray
    N_elig: np.ndarray
    risk_behavior: np.ndarray
    no_vs: np.ndarray


@dataclass
class CDCOutput:
    """Output from CDC model."""
    unit_id: str
    years: np.ndarray
    prep_on_count: np.ndarray
    incidence: np.ndarray
    diagnosed: np.ndarray
    undiagnosed: np.ndarray


@dataclass
class UncertaintySample:
    """Single MC sample."""
    sem_idx: int
    cdc_idx: int
    sem_trajectory: np.ndarray
    cdc_output: CDCOutput


@dataclass
class UncertaintyResult:
    """MC results for one unit."""
    unit_id: str
    samples: list[UncertaintySample]
    years: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def get_stack(self, var: str) -> np.ndarray:
        return np.array([getattr(s.cdc_output, var) for s in self.samples])

    def get_quantiles(self, var: str, q: tuple[float, ...] = (0.025, 0.5, 0.975)) -> dict[float, np.ndarray]:
        arr = self.get_stack(var)
        return {qi: np.quantile(arr, qi, axis=0) for qi in q}


@dataclass
class UncertaintyOutput:
    """MC results for all units."""
    results: dict[str, UncertaintyResult]
    years: np.ndarray
    v_names: list[str]


