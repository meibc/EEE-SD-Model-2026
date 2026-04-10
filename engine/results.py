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
    ts_cdc: np.ndarray | None = None
    cdc_names: list[str] | None = None


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
    inputs: PreparedData
    fit: FitResults
    predictions: PredictionResults | None

# Add to existing results.py

@dataclass
class CDCInputs:
    """Inputs to CDC model (constructed by connector)."""
    years: np.ndarray       # (T,)
    tau: np.ndarray         # (T,) from SEM
    prep_on: np.ndarray     # (T,) from SEM  
    N_elig: np.ndarray      # (T,) from CDC data


@dataclass
class CDCOutput:
    """Output from CDC model."""
    unit_id: str
    years: np.ndarray
    prep_on_count: np.ndarray   # (T,) people on PrEP
    incidence: np.ndarray       # (T,) new infections
    diagnosed: np.ndarray       # (T,) newly diagnosed (flow)
    undiagnosed: np.ndarray     # (T,) undiagnosed pool (stock)


@dataclass
class JointResult:
    """Joint result for one unit."""
    unit_id: str
    sem_trajectory: np.ndarray
    cdc_inputs: CDCInputs
    cdc_output: CDCOutput


@dataclass
class JointOutput:
    """Complete joint pipeline output."""
    results: dict[str, JointResult]
    sem_years: np.ndarray
    cdc_years: np.ndarray
    v_names: list[str]

@dataclass
class SEMSamples:
    """Pre-computed SEM posterior samples."""
    samples: np.ndarray     # (S, G, T, m)
    unit_order: list[str]
    v_names: list[str]
    ts: np.ndarray
    
    def get_unit_samples(self, unit_id: str) -> np.ndarray:
        """Get samples for one unit: (S, m, T)"""
        idx = self.unit_order.index(unit_id)
        return self.samples[:, idx, :, :].transpose(0, 2, 1)
    
    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]


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
        """Stack CDC output: (S, T)"""
        return np.array([getattr(s.cdc_output, var) for s in self.samples])
    
    def get_quantiles(
        self, 
        var: str, 
        q: list[float] = [0.025, 0.5, 0.975],
    ) -> dict[float, np.ndarray]:
        arr = self.get_stack(var)
        return {qi: np.quantile(arr, qi, axis=0) for qi in q}
    
