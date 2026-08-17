from __future__ import annotations

"""SEM/SBM model parameter schema."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SEMParams:
    """SEM parameters for one geography."""

    J: np.ndarray
    drift: np.ndarray | None = None
    reference_probs: np.ndarray | None = None
