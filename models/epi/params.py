from __future__ import annotations

"""Epidemiologic model parameter schema."""

from dataclasses import dataclass


@dataclass
class CDCParams:
    """CDC/hierarchical parameters for a single geography."""

    beta: float
    kdx: float
    U0: float
    kappa_prep: float
    alpha: float
    risk0: float = 1.0
    post_multiplier: float = 1.0
