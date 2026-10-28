from __future__ import annotations

"""Containers for the Monte Carlo projection results (pipeline/projection.py)."""

import numpy as np
from dataclasses import dataclass

from models.epi.params import EpiOutput


@dataclass
class ProjectionSample:
    """Single MC sample."""
    dynamic_idx: int
    epi_idx: int
    dynamic_trajectory: np.ndarray
    epi_output: EpiOutput


@dataclass
class ProjectionResult:
    """MC results for one unit."""
    unit_id: str
    samples: list[ProjectionSample]
    years: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def dynamic_stack(self) -> np.ndarray:
        """Projected measures, sample x measure x dynamic-model year."""
        return np.asarray([s.dynamic_trajectory for s in self.samples], dtype=float)

    def epi_stack(self, var: str) -> np.ndarray:
        """One epidemic output, sample x year."""
        return np.asarray([getattr(s.epi_output, var) for s in self.samples], dtype=float)


@dataclass
class ProjectionOutput:
    """MC results for all units."""
    results: dict[str, ProjectionResult]
    years: np.ndarray
    v_names: list[str]


