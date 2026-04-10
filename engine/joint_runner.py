# engine/joint_runner.py
"""Joint SEM → CDC pipeline."""

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

from data.unit import Unit
from data.params_cdc import CDCParamsLoader
from models.shared.alignment import align_to_years
from models.shared.transforms import hazard_proxy
from models.epi.prediction.predictor import CDCPredictor
from engine.results import (
    RunOutput,
    CDCInputs,
    JointResult,
    JointOutput,
    SEMSamples,
    UncertaintySample,
    UncertaintyResult,
)


CDC_YEARS = np.array([2017, 2018, 2019, 2020, 2021, 2022])


# =============================================================================
# Deterministic Runner
# =============================================================================

class JointRunner:
    """
    Joint SEM → CDC prediction (deterministic).
    
    Uses posterior means from both models.
    """
    
    def __init__(
        self,
        sem_output: RunOutput,
        cdc_params_loader: CDCParamsLoader,
        units: dict[str, Unit],
        cdc_years: np.ndarray = CDC_YEARS,
        hivtest_var: str = 'hivtest12',
        prep_var: str = 'prep_used',
        n_elig_var: str = 'PrEP Eligible',
        gamma_tau: float = 1.0,
    ):
        self.sem_output = sem_output
        self.cdc_loader = cdc_params_loader
        self.units = units
        self.cdc_years = cdc_years
        self.hivtest_var = hivtest_var
        self.prep_var = prep_var
        self.n_elig_var = n_elig_var
        self.gamma_tau = gamma_tau
        
        # Cache from SEM output
        self._v_names = sem_output.predictions.v_names
        self._sem_years = sem_output.predictions.ts
        self._hivtest_idx = self._v_names.index(hivtest_var)
        self._prep_idx = self._v_names.index(prep_var)
    
    def _build_cdc_inputs(self, unit_id: str, sem_traj: np.ndarray) -> CDCInputs:
        """Transform SEM trajectory → CDC inputs."""
        # Extract variables
        hivtest = sem_traj[self._hivtest_idx, :]
        prep_on = sem_traj[self._prep_idx, :]
        
        # Align to CDC years
        hivtest = align_to_years(self._sem_years, hivtest, self.cdc_years)
        prep_on = align_to_years(self._sem_years, prep_on, self.cdc_years)
        
        # Transform hivtest → tau
        tau = hazard_proxy(hivtest, gamma_tau=self.gamma_tau)
        
        # Get N_elig from unit
        unit = self.units[unit_id]
        n_elig = unit.get_cdc(self.n_elig_var)
        if n_elig is None:
            raise ValueError(f"'{self.n_elig_var}' not found for '{unit_id}'")
        if len(n_elig) != len(self.cdc_years):
            n_elig = align_to_years(unit.cdc_years, n_elig, self.cdc_years)
        
        return CDCInputs(
            years=self.cdc_years,
            tau=tau,
            prep_on=prep_on,
            N_elig=n_elig,
        )
    
    def predict(self, unit_id: str) -> JointResult:
        """Run for one unit."""
        sem_traj = self.sem_output.predictions.results[unit_id].Ypred_trajectory
        cdc_inputs = self._build_cdc_inputs(unit_id, sem_traj)
        
        cdc_params = self.cdc_loader.load_point_estimates(unit_id)
        cdc_output = CDCPredictor(cdc_params).predict(cdc_inputs, unit_id)
        
        return JointResult(
            unit_id=unit_id,
            sem_trajectory=sem_traj,
            cdc_inputs=cdc_inputs,
            cdc_output=cdc_output,
        )
    
    def run(self, unit_ids: list[str] | None = None) -> JointOutput:
        """Run for all units."""
        if unit_ids is None:
            unit_ids = list(self.sem_output.predictions.results.keys())
        
            # Filter to units with CDC params
        available = set(self.cdc_loader.geo_names)
        valid_ids = [uid for uid in unit_ids if uid in available]
        skipped = set(unit_ids) - set(valid_ids)
        
        if skipped:
            print(f"Skipping {len(skipped)} units without CDC params: {skipped}")
        
        results = {uid: self.predict(uid) for uid in unit_ids}
        
        return JointOutput(
            results=results,
            sem_years=self._sem_years,
            cdc_years=self.cdc_years,
            v_names=self._v_names,
        )


# =============================================================================
# Uncertainty Runner
# =============================================================================

class UncertaintyRunner:
    """
    Monte Carlo uncertainty propagation.
    
    Samples from SEM and CDC posteriors jointly.
    """
    
    def __init__(
        self,
        sem_samples: SEMSamples,
        cdc_params_loader: CDCParamsLoader,
        units: dict[str, Unit],
        cdc_years: np.ndarray = CDC_YEARS,
        hivtest_var: str = 'hivtest12',
        prep_var: str = 'prep_used',
        n_elig_var: str = 'PrEP Eligible',
        gamma_tau: float = 1.0,
    ):
        self.sem_samples = sem_samples
        self.cdc_loader = cdc_params_loader
        self.units = units
        self.cdc_years = cdc_years
        self.hivtest_var = hivtest_var
        self.prep_var = prep_var
        self.n_elig_var = n_elig_var
        self.gamma_tau = gamma_tau
        
        # Cache indices
        self._v_names = sem_samples.v_names
        self._sem_years = sem_samples.ts
        self._hivtest_idx = self._v_names.index(hivtest_var)
        self._prep_idx = self._v_names.index(prep_var)
        
        self.S_sem = sem_samples.n_samples
        self.S_cdc = cdc_params_loader.n_samples
    
    def _build_cdc_inputs(self, unit_id: str, sem_traj: np.ndarray) -> CDCInputs:
        """Transform SEM trajectory → CDC inputs."""
        hivtest = sem_traj[self._hivtest_idx, :]
        prep_on = sem_traj[self._prep_idx, :]
        
        hivtest = align_to_years(self._sem_years, hivtest, self.cdc_years)
        prep_on = align_to_years(self._sem_years, prep_on, self.cdc_years)
        
        tau = hazard_proxy(hivtest, gamma_tau=self.gamma_tau)
        
        unit = self.units[unit_id]
        n_elig = unit.get_cdc(self.n_elig_var)
        if n_elig is None:
            raise ValueError(f"'{self.n_elig_var}' not found for '{unit_id}'")
        if len(n_elig) != len(self.cdc_years):
            n_elig = align_to_years(unit.cdc_years, n_elig, self.cdc_years)
        
        return CDCInputs(
            years=self.cdc_years,
            tau=tau,
            prep_on=prep_on,
            N_elig=n_elig,
        )
    
    def predict_sample(
        self,
        unit_id: str,
        sem_idx: int,
        cdc_idx: int,
    ) -> UncertaintySample:
        """Single MC sample."""
        sem_traj = self.sem_samples.get_unit_samples(unit_id)[sem_idx]
        cdc_inputs = self._build_cdc_inputs(unit_id, sem_traj)
        
        cdc_params = self.cdc_loader.load_sample(cdc_idx, unit_id)
        cdc_output = CDCPredictor(cdc_params).predict(cdc_inputs, unit_id)
        
        return UncertaintySample(
            sem_idx=sem_idx,
            cdc_idx=cdc_idx,
            sem_trajectory=sem_traj,
            cdc_output=cdc_output,
        )
    
    def run(
        self,
        unit_id: str,
        n_samples: int = 1000,
        seed: int = 123,
        show_progress: bool = True,
    ) -> UncertaintyResult:
        """Run MC for one unit."""
        rng = np.random.default_rng(seed)
        
        idx_sem = rng.choice(self.S_sem, size=n_samples, replace=True)
        idx_cdc = rng.choice(self.S_cdc, size=n_samples, replace=True)
        
        samples = []
        iterator = zip(idx_sem, idx_cdc)
        if show_progress:
            iterator = tqdm(list(iterator), desc=f"MC {unit_id}")
        
        for s_sem, s_cdc in iterator:
            sample = self.predict_sample(unit_id, int(s_sem), int(s_cdc))
            samples.append(sample)
        
        return UncertaintyResult(
            unit_id=unit_id,
            samples=samples,
            years=self.cdc_years,
        )
    
    def run_all(
        self,
        unit_ids: list[str] | None = None,
        n_samples: int = 1000,
        seed: int = 123,
        show_progress: bool = True,
    ) -> dict[str, UncertaintyResult]:
        """Run MC for multiple units."""
        if unit_ids is None:
            unit_ids = self.sem_samples.unit_order
        
        results = {}
        for i, uid in enumerate(unit_ids):
            results[uid] = self.run(uid, n_samples, seed + i, show_progress)
        
        return results


# =============================================================================
# Loaders & Helpers
# =============================================================================

def load_sem_output(path: Path) -> RunOutput:
    """Load SEM output from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_sem_samples(path: Path) -> SEMSamples:
    """Load SEM samples from npz."""
    data = np.load(path, allow_pickle=True)
    return SEMSamples(
        samples=data['Ypreds_stack'],
        unit_order=list(data['unit_order']),
        v_names=list(data['v_names']),
        ts=data['ts'],
    )


def build_units_dict(units_list: list[Unit]) -> dict[str, Unit]:
    """Convert list of Units to dict."""
    return {u.id: u for u in units_list}


# =============================================================================
# Convenience Functions
# =============================================================================

def run_joint(
    sem_output: RunOutput,
    cdc_params_loader: CDCParamsLoader,
    units: dict[str, Unit],
    unit_ids: list[str] | None = None,
    **kwargs,
) -> JointOutput:
    """Run deterministic joint pipeline."""
    runner = JointRunner(sem_output, cdc_params_loader, units, **kwargs)
    return runner.run(unit_ids)


def run_uncertainty(
    sem_samples: SEMSamples,
    cdc_params_loader: CDCParamsLoader,
    units: dict[str, Unit],
    unit_ids: list[str] | None = None,
    n_samples: int = 1000,
    seed: int = 123,
    **kwargs,
) -> dict[str, UncertaintyResult]:
    """Run uncertainty pipeline."""
    runner = UncertaintyRunner(sem_samples, cdc_params_loader, units, **kwargs)
    return runner.run_all(unit_ids, n_samples, seed)
