from __future__ import annotations

"""Joint SEM → CDC pipeline."""

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

from data.unit import Unit
from data.params_cdc import CDCParamsLoader
from data.params_sem import SEMParamsLoader
from models.shared.alignment import build_cdc_inputs_from_sem, extend_years, extend_to_end_year
from models.shared.intervention import (
    build_relationship_interventions,
    build_state_interventions,
)
from models.epi.prediction.predictor import CDCPredictor
from models.sbm.prediction.predictor import Predictor
from core.math.transforms import Transforms
from pipeline.results import (
    RunOutput,
    CDCInputs,
    JointResult,
    JointOutput,
    UncertaintySample,
    UncertaintyResult,
    UncertaintyOutput,
)

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
        model_years: np.ndarray | None = None,
        state_intervention_codes: list[str] | None = None,
        relationship_intervention_codes: list[str] | None = None,
        intervention_duration_steps: int = 1,
        hivtest_var: str = 'hivtest12',
        prep_var: str = 'prep_used',
        risk_var: str = 'risk_behavior',
        n_elig_var: str = 'PrEP Eligible',
        prevalence_var: str = 'Estimated HIV prevalence (MSM)',
        viral_suppression_var: str = 'HIV viral suppression',
    ):
        self.sem_output = sem_output
        self.cdc_loader = cdc_params_loader
        self.units = units
        self.hivtest_var = hivtest_var
        self.prep_var = prep_var
        self.risk_var = risk_var
        self.n_elig_var = n_elig_var
        self.prevalence_var = prevalence_var
        self.viral_suppression_var = viral_suppression_var
        self.state_intervention_codes = list(state_intervention_codes or [])
        self.relationship_intervention_codes = list(relationship_intervention_codes or [])
        self.intervention_duration_steps = int(intervention_duration_steps)
        self.predictor = Predictor()
        self.transforms = Transforms()
        
        # Cache from SEM output
        self._v_names = sem_output.predictions.v_names
        self.model_years = (
            np.asarray(model_years, dtype=int)
            if model_years is not None
            else np.asarray(sem_output.predictions.ts, dtype=int)
        )
        self._sem_years = extend_to_end_year(
            sem_output.predictions.ts,
            target_end_year=int(self.model_years[-1]) if self.model_years.size > 0 else None,
        )
        self._hivtest_idx = self._v_names.index(hivtest_var)
        self._prep_idx = self._v_names.index(prep_var)
        self._risk_idx = self._v_names.index(risk_var)

    def _build_sem_trajectory(self, unit_id: str) -> np.ndarray:
        unit = self.units[unit_id]
        fit = self.sem_output.fit.results[unit_id]
        J_fit = np.asarray(fit.J, dtype=float)
        J = J_fit[:, :, -1] if J_fit.ndim == 3 else J_fit

        y0 = np.asarray(unit.amis_values[:, 0], dtype=float)
        x0 = self.transforms.logit(y0)
        u = np.zeros(J.shape[0], dtype=float)

        if (
            self.sem_output.predictions is not None
            and unit_id in self.sem_output.predictions.results
        ):
            n_steps = self.sem_output.predictions.results[unit_id].Ypred_trajectory.shape[1]
        else:
            n_steps = len(self._sem_years)

        state_iv = build_state_interventions(
            unit,
            self._sem_years,
            self._v_names,
            codes=self.state_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )
        rel_iv = build_relationship_interventions(
            unit,
            self._sem_years,
            self._v_names,
            codes=self.relationship_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )

        ypred, _ = self.predictor.predict_trajectory(
            J,
            x0,
            u,
            n_steps,
            state_interventions=state_iv,
            rel_interventions=rel_iv,
        )
        return ypred
    
    def predict(self, unit_id: str) -> JointResult:
        """Run for one unit."""
        if (
            self.state_intervention_codes
            or self.relationship_intervention_codes
            or self.sem_output.predictions is None
        ):
            sem_traj = self._build_sem_trajectory(unit_id)
        else:
            sem_traj = self.sem_output.predictions.results[unit_id].Ypred_trajectory
        sem_years = extend_years(self._sem_years, sem_traj.shape[1])
        tau, prep_on, n_elig, risk_behavior, no_vs = build_cdc_inputs_from_sem(
            sem_traj=sem_traj,
            unit=self.units[unit_id],
            hivtest_idx=self._hivtest_idx,
            prep_idx=self._prep_idx,
            risk_idx=self._risk_idx,
            sem_years=sem_years,
            model_years=self.model_years,
            n_elig_var=self.n_elig_var,
            prevalence_var=self.prevalence_var,
            viral_suppression_var=self.viral_suppression_var,
        )
        cdc_inputs = CDCInputs(
            years=self.model_years,
            tau=tau,
            prep_on=prep_on,
            N_elig=n_elig,
            risk_behavior=risk_behavior,
            no_vs=no_vs,
        )
        
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
            if self.sem_output.predictions is not None:
                unit_ids = list(self.sem_output.predictions.results.keys())
            else:
                unit_ids = list(self.sem_output.fit.results.keys())
        
            # Filter to units with CDC params
        available = set(self.cdc_loader.geo_names)
        valid_ids = [uid for uid in unit_ids if uid in available]
        skipped = set(unit_ids) - set(valid_ids)
        
        if skipped:
            print(f"Skipping {len(skipped)} units without CDC params: {skipped}")
        
        results = {uid: self.predict(uid) for uid in valid_ids}
        
        return JointOutput(
            results=results,
            sem_years=self._sem_years,
            cdc_years=self.model_years,
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
        sem_loader: SEMParamsLoader,
        cdc_params_loader: CDCParamsLoader,
        units: dict[str, Unit],
        model_years: np.ndarray | None = None,
        state_intervention_codes: list[str] | None = None,
        relationship_intervention_codes: list[str] | None = None,
        intervention_duration_steps: int = 1,
        v_names: list[str] | None = None,
        hivtest_var: str = 'hivtest12',
        prep_var: str = 'prep_used',
        risk_var: str = 'risk_behavior',
        n_elig_var: str = 'PrEP Eligible',
        prevalence_var: str = 'Estimated HIV prevalence (MSM)',
        viral_suppression_var: str = 'HIV viral suppression',
    ):
        self.sem_loader = sem_loader
        self.cdc_loader = cdc_params_loader
        self.units = units
        self.hivtest_var = hivtest_var
        self.prep_var = prep_var
        self.risk_var = risk_var
        self.n_elig_var = n_elig_var
        self.prevalence_var = prevalence_var
        self.viral_suppression_var = viral_suppression_var
        self.state_intervention_codes = list(state_intervention_codes or [])
        self.relationship_intervention_codes = list(relationship_intervention_codes or [])
        self.intervention_duration_steps = int(intervention_duration_steps)
        self.predictor = Predictor()
        self.transforms = Transforms()

        self._v_names = list(v_names if v_names is not None else sem_loader.v_names)
        self.model_years = np.asarray(
            model_years if model_years is not None else sem_loader.ts,
            dtype=int,
        )
        self._sem_years = extend_to_end_year(
            sem_loader.ts,
            target_end_year=int(self.model_years[-1]) if self.model_years.size > 0 else None,
        )
        self.S_sem = sem_loader.n_samples
        self._unit_order = list(sem_loader.geo_names)

        self._hivtest_idx = self._v_names.index(hivtest_var)
        self._prep_idx = self._v_names.index(prep_var)
        self._risk_idx = self._v_names.index(risk_var)
        self.S_cdc = cdc_params_loader.n_samples

    def _build_sem_trajectory(self, unit_id: str, sem_idx: int) -> np.ndarray:
        sem_params = self.sem_loader.load_sample(sem_idx, unit_id)
        J = np.asarray(sem_params.J, dtype=float)

        unit = self.units[unit_id]

        y0 = np.asarray(unit.amis_values[:, 0], dtype=float)
        x0 = self.transforms.logit(y0)

        u = np.zeros(J.shape[0], dtype=float)
        n_steps = len(self._sem_years)

        state_iv = build_state_interventions(
            unit,
            self._sem_years,
            self._v_names,
            codes=self.state_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )

        rel_iv = build_relationship_interventions(
            unit,
            self._sem_years,
            self._v_names,
            codes=self.relationship_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )

        ypred, _ = self.predictor.predict_trajectory(
            J,
            x0,
            u,
            n_steps,
            state_interventions=state_iv,
            rel_interventions=rel_iv,
        )

        return ypred
    
    def predict_sample(
        self,
        unit_id: str,
        sem_idx: int,
        cdc_idx: int,
    ) -> UncertaintySample:
        """Single MC sample."""
        sem_traj = self._build_sem_trajectory(unit_id, sem_idx)
        sem_years = extend_years(self._sem_years, sem_traj.shape[1])
        tau, prep_on, n_elig, risk_behavior, no_vs = build_cdc_inputs_from_sem(
            sem_traj=sem_traj,
            unit=self.units[unit_id],
            hivtest_idx=self._hivtest_idx,
            prep_idx=self._prep_idx,
            risk_idx=self._risk_idx,
            sem_years=sem_years,
            model_years=self.model_years,
            n_elig_var=self.n_elig_var,
            prevalence_var=self.prevalence_var,
            viral_suppression_var=self.viral_suppression_var,
        )
        cdc_inputs = CDCInputs(
            years=self.model_years,
            tau=tau,
            prep_on=prep_on,
            N_elig=n_elig,
            risk_behavior=risk_behavior,
            no_vs=no_vs,
        )
        
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
            iterator = tqdm(zip(idx_sem, idx_cdc), total=n_samples, desc=f"MC {unit_id}")
        
        for s_sem, s_cdc in iterator:
            sample = self.predict_sample(unit_id, int(s_sem), int(s_cdc))
            samples.append(sample)
        
        return UncertaintyResult(
            unit_id=unit_id,
            samples=samples,
            years=self.model_years,
        )
    
    def run_all(
        self,
        unit_ids: list[str] | None = None,
        n_samples: int = 1000,
        seed: int = 123,
        show_progress: bool = True,
    ) -> UncertaintyOutput:
        """Run MC for multiple units."""
        if unit_ids is None:
            unit_ids = list(self._unit_order)

        # Keep only units available in all required sources.
        available_cdc = set(self.cdc_loader.geo_names)
        available_units = set(self.units.keys())
        available_sem = set(self._unit_order)
        unit_ids = [
            uid for uid in unit_ids
            if uid in available_cdc and uid in available_units and uid in available_sem
        ]
        
        results = {}
        for i, uid in enumerate(unit_ids):
            results[uid] = self.run(uid, n_samples, seed + i, show_progress)
        
        return UncertaintyOutput(
            results=results,
            years=self.model_years,
            v_names=self._v_names,
        )


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
    sem_loader: SEMParamsLoader,
    cdc_params_loader: CDCParamsLoader,
    units: dict[str, Unit],
    unit_ids: list[str] | None = None,
    n_samples: int = 1000,
    seed: int = 123,
    show_progress: bool = True,
    **kwargs,
) -> UncertaintyOutput:
    """Run uncertainty pipeline."""
    runner = UncertaintyRunner(sem_loader, cdc_params_loader, units, **kwargs)
    return runner.run_all(unit_ids, n_samples, seed, show_progress=show_progress)
