"""Monte Carlo projection: fitted dynamic model -> epidemic model, with interventions.

The model (``project``): the fit is the mixture model (pipeline/dynamic_fit.py), whose PrEP row is P0 / ceiling
(P0: PrEP among men without AHS). Each year after 2024

    X[t]       = J X[t-1] + u                                         (centred logits)
    X[AHS, t] += b x (logit P[t-1] - mean logit of observed P)         (PrEP -> AHS feedback, as in the fit)
    interventions then hold the intervened measures on their targets  (the feedback cannot undo them)

with population PrEP P = ceiling x P0-share x (1 - AHS x (1 - r)) (``population_prep``). The baseline is the same
run without interventions.

Monte Carlo (``ProjectionRunner``): each sample pairs one dynamic-model draw (transition matrix, drift, observed
inputs) with one epidemic-model posterior draw. Draws are taken without replacement when there are enough of
them, and the random stream depends only on the seed: sample k uses the same dynamic-model draw (bootstrap
replicate) and the same epidemic posterior draw in every state, so sums over states within a sample are joint
realizations and pooled intervals carry uncertainty shared across states. Baseline and scenario runs with the
same seed use the same pairs, so their contrasts are paired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from core.years import extend_to_end_year
from data.unit import Unit
from models.dynamic.transforms import FeatureTransformer
from models.epi.input_adapter import epi_inputs_from_dynamic
from models.epi.params import EpiOutput
from models.epi.prediction.predictor import EpiPredictor
from models.scenarios.intervention import build_measure_interventions, intervention_window, ramp
from models.dynamic.mixture import feedback_input, prep_without_ahs

AHS, PREP, HIVTEST, RISK = "stigma_ahs", "prep_used", "hivtest12", "risk_behavior"
N_ELIGIBLE = "PrEP Eligible"


# =============================================================================
# The model
# =============================================================================

@dataclass
class DrawInputs:
    """One draw's model inputs for one unit."""
    J: np.ndarray                 # transition matrix (measure x measure)
    u: np.ndarray                 # drift
    x_observed: np.ndarray        # observed periods, centred logits (PrEP row: P0 / ceiling)
    prep_observed: np.ndarray     # observed population PrEP, by period
    ceiling: float                # PrEP ceiling
    ratios: np.ndarray            # AHS-PrEP prevalence ratio r, by year
    b: float                      # PrEP -> AHS feedback
    transformer: FeatureTransformer   # probabilities <-> centred logits (the unit's anchor)
    ahs: int                      # row of AHS
    prep: int                     # row of PrEP


def population_prep(d: DrawInputs, y: np.ndarray, t: int) -> float:
    """Population PrEP in year t from the model's measures y (probabilities); observed in observed periods."""
    if t < d.x_observed.shape[1]:
        return d.prep_observed[t]
    return d.ceiling * y[d.prep] * (1 - y[d.ahs] * (1 - d.ratios[t]))


def project(d: DrawInputs, interventions=(), baseline: np.ndarray | None = None) -> np.ndarray:
    """Projected measures (measure x year), probabilities on the model's scale (PrEP row: P0 / ceiling).
    ``baseline`` (the run without interventions) caps the intervention targets."""
    k, n = d.x_observed.shape[1], len(d.ratios)
    X = np.zeros((d.J.shape[0], n), dtype=float)
    X[:, :k] = d.x_observed       # observed periods held; the projection starts from the last one
    for t in range(k, n):
        X[:, t] = d.J @ X[:, t - 1] + d.u
        X[d.ahs, t] += feedback_input(population_prep(d, d.transformer.inverse_transform(X[:, t - 1]), t - 1),
                                      d.prep_observed, d.b)
        for iv in interventions:
            X[:, t] = iv.apply(X[:, t], t, transformer=d.transformer, baseline_p=baseline[:, t])
    return d.transformer.inverse_transform(X)


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
    """Monte Carlo projection: each sample pairs a dynamic-model draw with an epidemic-model posterior draw.

    intervention_codes       config/interventions.py; ramped in over intervention_duration_steps two-year steps
    ratio_gap_closed         fraction of the PrEP gap for men with AHS (1 - r) closed, on the same ramp
                             (stigma-resilient services)
    zero_projection_edges    diagnostic: (source, target) pathways set to zero, baseline and scenario alike;
                             ("prep_used", "stigma_ahs") sets the feedback b to 0
    """

    def __init__(
        self,
        dynamic_loader: Any,
        epi_loader: Any,
        units: dict[str, Unit],
        model_years: np.ndarray | None = None,
        intervention_codes: list[str] | None = None,
        intervention_duration_steps: int = 1,
        ratio_gap_closed: float | None = None,
        zero_projection_edges: tuple[tuple[str, str], ...] = (),
    ):
        if getattr(dynamic_loader, "mixture_prep_to_ahs", None) is None:
            raise ValueError("Projection needs a mixture fit (mixture_feedback); v2 fits are no longer projected")
        self.dynamic_loader = dynamic_loader
        self.epi_loader = epi_loader
        self.units = units
        self.intervention_codes = list(intervention_codes or [])
        self.intervention_duration_steps = int(intervention_duration_steps)
        self.ratio_gap_closed = ratio_gap_closed
        self.zero_projection_edges = tuple(zero_projection_edges)

        self.v_names = list(dynamic_loader.v_names)
        self.idx = {v: self.v_names.index(v) for v in (AHS, PREP, HIVTEST, RISK)}
        self.model_years = np.asarray(model_years if model_years is not None else dynamic_loader.ts, dtype=int)
        self.dynamic_years = extend_to_end_year(
            dynamic_loader.ts, target_end_year=int(self.model_years[-1]) if self.model_years.size > 0 else None)

    def _draw_inputs(self, unit_id: str, dynamic_idx: int) -> DrawInputs:
        """Load one draw and put it on the fit's scale."""
        ahs, prep = self.idx[AHS], self.idx[PREP]
        params = self.dynamic_loader.load_sample(dynamic_idx, unit_id)
        J = np.array(params.J, dtype=float)
        for source, target in self.zero_projection_edges:
            J[self.v_names.index(target), self.v_names.index(source)] = 0.0
        if J[ahs, prep] != 0:
            raise ValueError("Mixture fit with a PrEP -> AHS entry in J: refit (the feedback now acts on population PrEP)")
        b = 0.0 if (PREP, AHS) in self.zero_projection_edges else self.dynamic_loader.mixture_prep_to_ahs

        # This draw's own observed inputs (the bundle's sem_obs_samples), PrEP row as in the fit: P0 / ceiling.
        observed = np.array(params.observed, dtype=float)
        ratio = self.dynamic_loader.ahs_prep_apr(dynamic_idx, unit_id)
        ceiling = self.dynamic_loader.mixture_prep_ceiling(unit_id)
        prep_observed = observed[prep].copy()
        observed[prep] = np.clip(prep_without_ahs(prep_observed, observed[ahs], ratio) / ceiling, 0.0, 0.999)
        # Same transform as the fit (probabilities clipped to [0.01, 0.99]), centred at the unit's anchor.
        transformer = FeatureTransformer(reference_probs=params.reference_probs, reference_logits=params.reference_logits)
        x_observed = transformer.transform(observed)

        ratios = np.full(len(self.dynamic_years), ratio)
        if self.ratio_gap_closed is not None:
            start, end = intervention_window(self.units[unit_id], self.dynamic_years, self.intervention_duration_steps)
            for t in range(len(ratios)):
                ratios[t] = ratio + ramp(t, start, end) * self.ratio_gap_closed * (1 - ratio)

        u = np.zeros(J.shape[0]) if params.drift is None else np.asarray(params.drift, dtype=float)
        return DrawInputs(J=J, u=u, x_observed=x_observed, prep_observed=prep_observed, ceiling=ceiling,
                          ratios=ratios, b=b, transformer=transformer, ahs=ahs, prep=prep)

    def _trajectory(self, unit_id: str, dynamic_idx: int) -> np.ndarray:
        """One draw's projected measures (measure x dynamic-model year, probabilities; PrEP row: population PrEP),
        the baseline or, with interventions, the scenario."""
        d = self._draw_inputs(unit_id, dynamic_idx)
        y = project(d)
        interventions = build_measure_interventions(self.units[unit_id], self.dynamic_years, self.v_names,
                                                    codes=self.intervention_codes,
                                                    duration_steps=self.intervention_duration_steps)
        if interventions:
            for iv in interventions:
                iv.anchor_p = y[:, iv.start_t - 1]     # targets relative to 2024 use this draw's own 2024 value
            y = project(d, interventions, baseline=y)
        y[d.prep] = [population_prep(d, y[:, t], t) for t in range(y.shape[1])]
        return y

    def predict_sample(self, unit_id: str, dynamic_idx: int, epi_idx: int) -> ProjectionSample:
        """Single MC sample."""
        trajectory = self._trajectory(unit_id, dynamic_idx)
        epi_inputs = epi_inputs_from_dynamic(
            dynamic_traj=trajectory, unit=self.units[unit_id],
            hivtest_idx=self.idx[HIVTEST], prep_idx=self.idx[PREP], risk_idx=self.idx[RISK],
            dynamic_years=self.dynamic_years, model_years=self.model_years, n_elig_var=N_ELIGIBLE,
        )
        epi_output = EpiPredictor(self.epi_loader.load_sample(epi_idx, unit_id)).predict(epi_inputs, unit_id)
        return ProjectionSample(dynamic_idx=dynamic_idx, epi_idx=epi_idx, dynamic_trajectory=trajectory,
                                epi_output=epi_output)

    def run(self, unit_id: str, n_samples: int = 1000, seed: int = 123, show_progress: bool = True) -> ProjectionResult:
        """Run MC for one unit."""
        rng = np.random.default_rng(seed)    # same pairs in every state (joint across states)
        S_dynamic, S_epi = self.dynamic_loader.n_samples, self.epi_loader.n_samples
        pairs = zip(rng.choice(S_dynamic, size=n_samples, replace=n_samples > S_dynamic),
                    rng.choice(S_epi, size=n_samples, replace=n_samples > S_epi))
        if show_progress:
            pairs = tqdm(pairs, total=n_samples, desc=f"MC {unit_id}")
        samples = [self.predict_sample(unit_id, int(s_dynamic), int(s_epi)) for s_dynamic, s_epi in pairs]
        return ProjectionResult(unit_id=unit_id, samples=samples, years=self.model_years)

    def run_all(self, unit_ids: list[str] | None = None, n_samples: int = 1000, seed: int = 123,
                show_progress: bool = True) -> ProjectionOutput:
        """Run MC for the units available in every source (all of them by default)."""
        available = set(self.dynamic_loader.geo_names) & set(self.epi_loader.geo_names) & set(self.units)
        unit_ids = [u for u in (unit_ids or self.dynamic_loader.geo_names) if u in available]
        results = {u: self.run(u, n_samples, seed, show_progress) for u in unit_ids}
        return ProjectionOutput(results=results, years=self.model_years, v_names=self.v_names)


def run_projection(dynamic_loader: Any, epi_loader: Any, units: dict[str, Unit], unit_ids: list[str] | None = None,
                   n_samples: int = 1000, seed: int = 123, show_progress: bool = True, **kwargs) -> ProjectionOutput:
    """Run the projection for the given units."""
    runner = ProjectionRunner(dynamic_loader, epi_loader, units, **kwargs)
    return runner.run_all(unit_ids, n_samples, seed, show_progress=show_progress)
