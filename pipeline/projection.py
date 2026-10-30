"""Monte Carlo projection: fitted dynamic model -> epidemic model, with interventions.

Each sample pairs one dynamic-model draw (transition matrix, drift, observed inputs)
with one epidemic-model posterior draw. Draws are taken without replacement when
there are enough of them, and the random stream depends only on the seed: sample k
uses the same dynamic-model draw (bootstrap replicate) and the same epidemic posterior
draw in every state, so sums over states within a sample are joint realizations and
pooled intervals carry uncertainty shared across states. Baseline and scenario runs
with the same seed use the same pairs, so their contrasts are paired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from tqdm import tqdm

from data.unit import Unit
from models.epi.input_adapter import epi_inputs_from_dynamic
from core.years import extend_to_end_year
from models.scenarios.intervention import build_measure_interventions
from models.scenarios.prep_translation import contextual_translation, prevalence_ratio_translation, retain_prep_fraction
from models.epi.params import EpiOutput
from models.epi.prediction.predictor import EpiPredictor
from models.dynamic.estimation.transforms import FeatureTransformer
from models.dynamic.prediction.predictor import Predictor


# =============================================================================
# Projection outputs
# =============================================================================

@dataclass
class ProjectionSample:
    """One paired dynamic-model and epidemic-model Monte Carlo sample."""
    dynamic_idx: int
    epi_idx: int
    dynamic_trajectory: np.ndarray
    epi_output: EpiOutput


@dataclass
class ProjectionResult:
    """All Monte Carlo samples for one unit."""
    unit_id: str
    samples: list[ProjectionSample]
    years: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def dynamic_stack(self) -> np.ndarray:
        """Projected measures, sample x measure x dynamic-model year."""
        return np.asarray([sample.dynamic_trajectory for sample in self.samples], dtype=float)

    def epi_stack(self, variable: str) -> np.ndarray:
        """One epidemic output, sample x year."""
        return np.asarray([getattr(sample.epi_output, variable) for sample in self.samples], dtype=float)


@dataclass
class ProjectionOutput:
    """Monte Carlo results for all requested units."""
    results: dict[str, ProjectionResult]
    years: np.ndarray
    v_names: list[str]

# =============================================================================
# Projection runner
# =============================================================================

class ProjectionRunner:
    """Monte Carlo projection: each sample pairs a dynamic-model draw with an epidemic-model posterior draw."""
    
    def __init__(
        self,
        dynamic_loader: Any,
        epi_loader: Any,
        units: dict[str, Unit],
        model_years: np.ndarray | None = None,
        intervention_codes: list[str] | None = None,
        intervention_duration_steps: int = 1,
        v_names: list[str] | None = None,
        hivtest_var: str = 'hivtest12',
        prep_var: str = 'prep_used',
        risk_var: str = 'risk_behavior',
        n_elig_var: str = 'PrEP Eligible',
        prep_translation: str = "none",
        contextual_prep_pr: float | None = None,
        prep_only_fraction: float | None = None,
        zero_projection_edges: tuple[tuple[str, str], ...] = (),
    ):
        if prep_translation not in ("none", "prevalence_ratio", "contextual"):
            raise ValueError(f"Unknown PrEP translation {prep_translation!r}")
        if prep_translation == "contextual" and (contextual_prep_pr is None or prep_only_fraction is not None):
            raise ValueError("The contextual translation needs contextual_prep_pr and excludes prep_only_fraction")
        # Scenario PrEP from the AHS change (models/scenarios/prep_translation.py): "none" keeps the fitted
        # dynamic response (the baseline run, where AHS does not change), "prevalence_ratio" is the primary
        # translation and "contextual" the upper bound.
        self.prep_translation = prep_translation
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
        self.dynamic_loader = dynamic_loader
        self.epi_loader = epi_loader
        self.units = units
        self.hivtest_var = hivtest_var
        self.prep_var = prep_var
        self.risk_var = risk_var
        self.n_elig_var = n_elig_var
        self.intervention_codes = list(intervention_codes or [])
        self.intervention_duration_steps = int(intervention_duration_steps)

        self._v_names = list(v_names if v_names is not None else dynamic_loader.v_names)
        self.model_years = np.asarray(
            model_years if model_years is not None else dynamic_loader.ts,
            dtype=int,
        )
        self._dynamic_years = extend_to_end_year(
            dynamic_loader.ts,
            target_end_year=int(self.model_years[-1]) if self.model_years.size > 0 else None,
        )
        self.S_dynamic = dynamic_loader.n_samples
        self._unit_order = list(dynamic_loader.geo_names)

        self._hivtest_idx = self._v_names.index(hivtest_var)
        self._prep_idx = self._v_names.index(prep_var)
        self._risk_idx = self._v_names.index(risk_var)
        self.S_epi = epi_loader.n_samples

    def _build_dynamic_trajectory(self, unit_id: str, dynamic_idx: int) -> np.ndarray:
        dynamic_params = self.dynamic_loader.load_sample(dynamic_idx, unit_id)
        J = np.array(dynamic_params.J, dtype=float)
        for source, target in self.zero_projection_edges:
            J[self._v_names.index(target), self._v_names.index(source)] = 0.0
        unit = self.units[unit_id]

        # This draw's own observed inputs (the bundle's sem_obs_samples).
        observed = np.asarray(dynamic_params.observed, dtype=float)
        predictor = Predictor(reference_probs=dynamic_params.reference_probs, reference_logits=dynamic_params.reference_logits)
        # Same transform as the fit (FeatureTransformer: probabilities clipped to [0.01, 0.99]).
        x_observed = FeatureTransformer(reference_logits=predictor.reference_logits).transform(observed)[0]
        x0 = x_observed[:, 0]
        fixed_prefix = x_observed   # observed periods held; projection starts from the last one

        u = np.zeros(J.shape[0], dtype=float) if dynamic_params.drift is None else np.asarray(dynamic_params.drift, dtype=float)
        n_steps = len(self._dynamic_years)
        baseline_probs, _ = predictor.predict_trajectory(J, x0, u, n_steps, fixed_prefix=fixed_prefix)

        state_iv = build_measure_interventions(
            unit, self._dynamic_years, self._v_names,
            codes=self.intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )
        for iv in state_iv:
            # Cuts defined relative to the last observed period use this draw's own value there.
            iv.anchor_p = baseline_probs[:, iv.start_t - 1]

        ypred, _ = predictor.predict_trajectory(
            J, x0, u, n_steps,
            state_interventions=state_iv,
            baseline_probs=baseline_probs,
            fixed_prefix=fixed_prefix,
        )
        # Scenario PrEP from the AHS change (models/scenarios/prep_translation.py)
        ahs, prep = self._v_names.index("stigma_ahs"), self._prep_idx
        if self.prep_translation == "contextual":
            ypred[prep] = contextual_translation(baseline_probs[prep], baseline_probs[ahs], ypred[ahs],
                                                 self.contextual_prep_pr)
        if self.prep_translation == "prevalence_ratio":
            ypred[prep] = prevalence_ratio_translation(baseline_probs[prep], baseline_probs[ahs], ypred[ahs],
                                                       self.dynamic_loader.ahs_prep_apr(dynamic_idx, unit_id))
        if self.prep_only_fraction is not None:
            ypred = retain_prep_fraction(baseline_probs, ypred, prep, self.prep_only_fraction)
        return ypred

    def predict_sample(
        self,
        unit_id: str,
        dynamic_idx: int,
        epi_idx: int,
    ) -> ProjectionSample:
        """Single MC sample."""
        dynamic_traj = self._build_dynamic_trajectory(unit_id, dynamic_idx)
        epi_inputs = epi_inputs_from_dynamic(
            dynamic_traj=dynamic_traj,
            unit=self.units[unit_id],
            hivtest_idx=self._hivtest_idx,
            prep_idx=self._prep_idx,
            risk_idx=self._risk_idx,
            dynamic_years=self._dynamic_years,
            model_years=self.model_years,
            n_elig_var=self.n_elig_var,
        )
        epi_params = self.epi_loader.load_sample(epi_idx, unit_id)
        epi_output = EpiPredictor(epi_params).predict(epi_inputs, unit_id)
        
        return ProjectionSample(
            dynamic_idx=dynamic_idx,
            epi_idx=epi_idx,
            dynamic_trajectory=dynamic_traj,
            epi_output=epi_output,
        )
    
    def run(
        self,
        unit_id: str,
        n_samples: int = 1000,
        seed: int = 123,
        show_progress: bool = True,
    ) -> ProjectionResult:
        """Run MC for one unit."""
        rng = np.random.default_rng(seed)    # same pairs in every state (joint across states)
        idx_dynamic = rng.choice(self.S_dynamic, size=n_samples, replace=n_samples > self.S_dynamic)
        idx_epi = rng.choice(self.S_epi, size=n_samples, replace=n_samples > self.S_epi)
        
        samples = []
        iterator = zip(idx_dynamic, idx_epi)
        if show_progress:
            iterator = tqdm(zip(idx_dynamic, idx_epi), total=n_samples, desc=f"MC {unit_id}")
        
        for s_dynamic, s_epi in iterator:
            sample = self.predict_sample(unit_id, int(s_dynamic), int(s_epi))
            samples.append(sample)
        
        return ProjectionResult(
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
    ) -> ProjectionOutput:
        """Run MC for multiple units."""
        if unit_ids is None:
            unit_ids = list(self._unit_order)

        # Keep only units available in all required sources.
        available_epi = set(self.epi_loader.geo_names)
        available_units = set(self.units.keys())
        available_dynamic = set(self._unit_order)
        unit_ids = [
            uid for uid in unit_ids
            if uid in available_epi and uid in available_units and uid in available_dynamic
        ]
        
        results = {uid: self.run(uid, n_samples, seed, show_progress) for uid in unit_ids}
        
        return ProjectionOutput(
            results=results,
            years=self.model_years,
            v_names=self._v_names,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_projection(
    dynamic_loader: Any,
    epi_loader: Any,
    units: dict[str, Unit],
    unit_ids: list[str] | None = None,
    n_samples: int = 1000,
    seed: int = 123,
    show_progress: bool = True,
    **kwargs,
) -> ProjectionOutput:
    """Run the projection for the given units."""
    runner = ProjectionRunner(dynamic_loader, epi_loader, units, **kwargs)
    return runner.run_all(unit_ids, n_samples, seed, show_progress=show_progress)
