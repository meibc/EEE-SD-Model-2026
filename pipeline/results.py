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
class FitResult:
    """Fit result for single unit."""
    unit_id: str
    J: np.ndarray
    Ybar: np.ndarray
    Xbar: np.ndarray


@dataclass
class FitResults:
    """Fit results for all units."""
    results: dict[str, FitResult]
    ts: np.ndarray
    v_names: list[str]


@dataclass
class PredictionResult:
    unit_id: str
    Ypred_rolling: np.ndarray      # (m, T+1) - one-step-ahead, Y space
    Xpred_rolling: np.ndarray      # (m, T+1) - one-step-ahead, X space
    Ypred_trajectory: np.ndarray   # (m, T+n_forecast) - recursive sim, Y space
    Xpred_trajectory: np.ndarray   # (m, T+n_forecast) - recursive sim, X space


@dataclass
class PredictionResults:
    """Prediction results for all units."""
    results: dict[str, PredictionResult]
    ts: np.ndarray
    v_names: list[str]


@dataclass
class RunOutput:
    """Complete output from SBRunner."""
    fit: FitResults
    predictions: PredictionResults | None
