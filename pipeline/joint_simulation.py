from __future__ import annotations

"""Monte Carlo projection: fitted state model -> CDC/epidemic model, with interventions.

Each sample pairs one state-model draw (transition matrix, drift, observed inputs)
with one epidemic-model posterior draw. Draws are taken without replacement when
there are enough of them, and the random stream depends only on the seed and the
state code, so a state gets the same pairs whichever other states are run.
Baseline and scenario runs with the same seed use the same pairs, so their
contrasts are paired.
"""

from typing import Any
import numpy as np
from tqdm import tqdm

from data.unit import Unit
from models.shared.alignment import build_cdc_inputs_from_sem, extend_years, extend_to_end_year
from models.shared.intervention import (
    _logit,
    build_relationship_interventions,
    build_state_interventions,
)
from models.epi.prediction.predictor import CDCPredictor
from models.sbm.estimation.transforms import FeatureTransformer
from models.sbm.prediction.predictor import Predictor
from core.math.transforms import Transforms
from pipeline.results import (
    CDCInputs,
    UncertaintySample,
    UncertaintyResult,
    UncertaintyOutput,
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
        sem_loader: Any,
        cdc_params_loader: Any,
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
        projection_start: str = 'observed',
        contextual_prep_pr: float | None = None,
        constant_pr_prep: bool = False,
        prep_only_fraction: float | None = None,
        zero_projection_edges: tuple[tuple[str, str], ...] = (),
    ):
        if projection_start not in ('observed', 'reconstructed'):
            raise ValueError("projection_start must be 'observed' or 'reconstructed'")
        if contextual_prep_pr is not None and (constant_pr_prep or prep_only_fraction is not None):
            raise ValueError("contextual_prep_pr cannot be combined with constant_pr_prep or prep_only_fraction")
        if zero_projection_edges and projection_start != 'observed':
            raise ValueError("zero_projection_edges applies to projections from the observed periods only")
        self.projection_start = projection_start
        # Resilient delivery (derived policy comparison): only this fraction of the scenario's change in
        # PrEP (centered logit) is kept, and every other measure, AHS included, stays at baseline.
        self.prep_only_fraction = prep_only_fraction
        # Diagnostic: pathways (source, target) set to zero in the projection, baseline and scenario
        # alike (e.g. PrEP -> AHS, to see how much projected AHS growth relies on that feedback).
        self.zero_projection_edges = tuple(zero_projection_edges)
        # Contextual upper bound (sensitivity): scenario PrEP is the baseline PrEP multiplied by
        # contextual_prep_pr ** (change in AHS / 0.10), the between-state association of AHS climate
        # with everyone's PrEP use, assumed causal; the counterpart of the primary translation below
        # with the climate ratio in place of the individual one.
        self.contextual_prep_pr = contextual_prep_pr
        # Primary translation: scenario PrEP is the baseline PrEP moved by the draw's
        # respondent-level prevalence ratio r held constant at the projected levels,
        # P_s = P_b (1 - A_s (1 - r)) / (1 - A_b (1 - r)), in place of the fitted PrEP response.
        self.constant_pr_prep = constant_pr_prep
        self.sem_loader = sem_loader
        self.cdc_loader = cdc_params_loader
        self.units = units
        self.hivtest_var = hivtest_var
        self.prep_var = prep_var
        self.risk_var = risk_var
        self.n_elig_var = n_elig_var
        self.state_intervention_codes = list(state_intervention_codes or [])
        self.relationship_intervention_codes = list(relationship_intervention_codes or [])
        self.intervention_duration_steps = int(intervention_duration_steps)
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
        J = np.array(sem_params.J, dtype=float)
        for source, target in self.zero_projection_edges:
            J[self._v_names.index(target), self._v_names.index(source)] = 0.0
        unit = self.units[unit_id]

        # Draw-specific observed inputs when the bundle carries them; otherwise
        # the point estimates.
        observed = sem_params.observed if sem_params.observed is not None else unit.amis_values
        observed = np.asarray(observed, dtype=float)
        predictor = Predictor(reference_probs=sem_params.reference_probs, reference_logits=sem_params.reference_logits)
        # Same transform as the fit (FeatureTransformer: probabilities clipped to [0.01, 0.99]).
        x_observed = FeatureTransformer(reference_logits=predictor.reference_logits).transform(observed)[0]
        x0 = x_observed[:, 0]
        # "observed": hold observed periods fixed and project from the last one.
        # "reconstructed": propagate the fitted system from the first period.
        fixed_prefix = x_observed if self.projection_start == "observed" else None

        u = np.zeros(J.shape[0], dtype=float) if sem_params.drift is None else np.asarray(sem_params.drift, dtype=float)
        n_steps = len(self._sem_years)
        baseline_probs, _ = predictor.predict_trajectory(J, x0, u, n_steps, fixed_prefix=fixed_prefix)

        state_iv = build_state_interventions(
            unit, self._sem_years, self._v_names,
            codes=self.state_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )
        rel_iv = build_relationship_interventions(
            unit, self._sem_years, self._v_names,
            codes=self.relationship_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )
        for iv in state_iv:
            # Cuts defined relative to the last observed period use this draw's own value there.
            iv.anchor_p = baseline_probs[:, iv.start_t - 1]

        ypred, _ = predictor.predict_trajectory(
            J, x0, u, n_steps,
            state_interventions=state_iv,
            rel_interventions=rel_iv,
            baseline_probs=baseline_probs,
            fixed_prefix=fixed_prefix,
        )
        if self.contextual_prep_pr is not None:
            ahs = self._v_names.index("stigma_ahs")
            factor = self.contextual_prep_pr ** ((ypred[ahs] - baseline_probs[ahs]) / 0.10)
            ypred[self._prep_idx] = np.clip(baseline_probs[self._prep_idx] * factor, 1e-6, 1 - 1e-6)
        if self.constant_pr_prep:
            ahs = self._v_names.index("stigma_ahs")
            gap = 1 - self.sem_loader.ahs_prep_apr(sem_idx, unit_id)
            factor = (1 - ypred[ahs] * gap) / (1 - baseline_probs[ahs] * gap)
            ypred[self._prep_idx] = np.clip(baseline_probs[self._prep_idx] * factor, 1e-6, 1 - 1e-6)
        if self.prep_only_fraction is not None:
            p = self._prep_idx
            lb = _logit(baseline_probs[p])
            prep = 1 / (1 + np.exp(-(lb + self.prep_only_fraction * (_logit(ypred[p]) - lb))))
            ypred = baseline_probs.copy()
            ypred[p] = prep
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
        rng = np.random.default_rng([seed, *unit_id.encode()])
        idx_sem = rng.choice(self.S_sem, size=n_samples, replace=n_samples > self.S_sem)
        idx_cdc = rng.choice(self.S_cdc, size=n_samples, replace=n_samples > self.S_cdc)
        
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
        
        results = {uid: self.run(uid, n_samples, seed, show_progress) for uid in unit_ids}
        
        return UncertaintyOutput(
            results=results,
            years=self.model_years,
            v_names=self._v_names,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_uncertainty(
    sem_loader: Any,
    cdc_params_loader: Any,
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
