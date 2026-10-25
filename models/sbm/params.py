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
    # State anchors (mean logit per measure) for this draw; when set, they define the
    # centred-logit space instead of reference_probs.
    reference_logits: np.ndarray | None = None
    # Observed probability-scale inputs (variables x observed periods) for this
    # draw; projections hold these fixed and start from the last period.
    observed: np.ndarray | None = None
