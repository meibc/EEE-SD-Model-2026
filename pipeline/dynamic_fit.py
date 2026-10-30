"""Hierarchical dynamic-model fits across input draws, for each model specification.

For each input draw, the nation, the nine Census divisions and the selected
states are fitted in that order; each division is shrunk toward the national
transition matrix and each state toward its division (``shrinkage_reference``).
Each draw is fitted by ``fit_hierarchy`` (models/dynamic/estimation/hierarchy.py), every unit by
``JacobianEstimator``, in state-anchored logit space: each
measure is centred at the unit's own mean logit over the observed periods, so the
pathways act on departures from a unit's usual level and fixed differences between
units (access, local climate) stay in the anchor. The fit uses the unit's
trajectory, the pooled covariance target, the structural and diagonal priors, the
drift priors, the stability penalty and an evidence target: the unit's total
AHS -> PrEP response over the projection horizon (every route and feedback acting)
should match its access-adjusted individual-level association (scripts/2_evidence/01;
models/dynamic/estimation/calibration.py). The association is estimated on the old AHS wording
(2017-2021 cycles, the scale of the wording-adjusted AHS) and each draw takes its
own prevalence ratio from the pooled estimate and its posterior SE (log scale), so
the target carries the evidence's uncertainty. The theory priors decide how that
total is split across the direct, disclosure, healthcare-contact and testing routes.

Model specifications (config/specifications.py) each change one component of the primary model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from config.optimization import OptimConfig
from config.dynamic import DynamicConfig
from config.shrinkage import ShrinkageConfig
from config.specifications import EVIDENCE_SEED, PERSISTENCE_CAP, SPECIFICATIONS
from data.state_inputs import StateInputs
from models.dynamic.estimation.calibration import evidence_slope, sustained_shift_response
from models.dynamic.estimation.hierarchy import FitSettings, UnitFit, fit_hierarchy

EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "results/02_inputs/ahs_prep_evidence.csv"
EVIDENCE_ALL_YEARS_PATH = EVIDENCE_PATH.with_name("ahs_prep_evidence_all_years.csv")
LOSS_COMPONENTS = ("LX", "LR", "LJ", "L_prior_J", "L_prior_c", "L_stab", "L_evidence")


# =============================================================================
# Specifications
# =============================================================================

@dataclass
class Specification:
    """One model specification: its fit settings, the measures whose periods are permuted (placebo),
    and the evidence file for the AHS -> PrEP target."""
    name: str
    fit: FitSettings
    permuted: tuple[str, ...]
    evidence_path: Path


def specification(name: str, config: DynamicConfig, v_names: list[str],
                  evidence_path: Path = EVIDENCE_PATH) -> Specification:
    """The primary settings with this specification's one change applied."""
    if name not in SPECIFICATIONS:
        raise ValueError(f"Unknown specification {name!r}; expected one of {list(SPECIFICATIONS)}")
    change = SPECIFICATIONS[name]
    opt, shrink = OptimConfig(), ShrinkageConfig()
    sign, prior_mean, prior_tau = config.structural_prior_arrays()
    drift_enabled, drift_mean, drift_tau = config.drift_prior_arrays()

    if "wR" in change:
        opt.wR = change["wR"]
    if change.get("shrinkage_off"):
        shrink.wJ = 0.0
        shrink.wJ_state = 0.0
    for measure, value in change.get("drift_mean", {}).items():
        drift_mean[v_names.index(measure)] = value
    if "persistence_shift" in change:
        idx = np.diag_indices(len(v_names))
        targeted = np.isfinite(prior_mean[idx])
        shifted = np.minimum(prior_mean[idx] + change["persistence_shift"], PERSISTENCE_CAP)
        prior_mean[idx] = np.where(targeted, shifted, prior_mean[idx])
    fit = FitSettings(opt=opt, shrink=shrink, sign=sign, prior_mean=prior_mean, prior_tau=prior_tau,
                      drift_enabled=drift_enabled, drift_mean=drift_mean, drift_tau=drift_tau)
    return Specification(name=name, fit=fit, permuted=tuple(change.get("permuted", ())),
                         evidence_path=EVIDENCE_ALL_YEARS_PATH if change.get("evidence") == "all_years" else evidence_path)


def evidence_slopes(units, config: DynamicConfig, v_names: list[str], draw_indices: np.ndarray,
                    evidence_path: Path = EVIDENCE_PATH) -> tuple[np.ndarray, np.ndarray]:
    """AHS -> PrEP response implied by each unit's access-adjusted prevalence ratio, per draw.

    States use their pooled (empirical-Bayes) ratio; divisions, the nation and any state
    without an estimate use the random-effects mean. Each draw takes the ratio from
    N(log pooled ratio, posterior SE) (seeded by the source draw index, so draws are paired
    across specifications). Levels are the unit's mean
    AHS and PrEP over the observed periods; the benchmark is the AHS reference probability.
    Returns (slopes, prevalence ratios), each draw x unit.
    """
    evidence = pd.read_csv(evidence_path).set_index("state")
    a, p = v_names.index("stigma_ahs"), v_names.index("prep_used")
    benchmark = float(config.reference_probs()[a])
    levels, log_apr, sd = [], [], []
    for unit in units:
        values = np.asarray(unit.amis_values, dtype=float)
        levels.append(tuple(float(np.nanmean(values[i])) for i in (a, p)))
        row = evidence.loc[str(unit.id)] if str(unit.id) in evidence.index else evidence.loc["POOLED"]
        log_apr.append(float(row["log_apr_pooled"]))
        sd.append(float(row["se_pooled"]))
    z = np.array([np.random.default_rng([EVIDENCE_SEED, int(d)]).standard_normal(len(units)) for d in draw_indices])
    apr = np.exp(np.asarray(log_apr) + np.asarray(sd) * z)
    slopes = np.array([[evidence_slope(ahs, prep, r, benchmark) for (ahs, prep), r in zip(levels, draw_aprs)]
                       for draw_aprs in apr])
    return slopes, apr


def _permute_periods(Y: np.ndarray, rows: list[int], rng: np.random.Generator) -> np.ndarray:
    """Placebo: reorder the periods of the given measures (never the identity order)."""
    Y = Y.copy()
    T = Y.shape[1]
    order = np.arange(T)
    while np.array_equal(order, np.arange(T)):
        order = rng.permutation(T)
    Y[rows] = Y[rows][:, order]
    return Y


# =============================================================================
# Runner
# =============================================================================

class DynamicFitRunner:
    """Fits one model specification to every selected input draw: the nation, the Census divisions and
    the selected states (models/dynamic/estimation/hierarchy.py), collecting the fits and their diagnostics."""

    def __init__(self, name: str, inputs: StateInputs, selected_states: list[str], draw_indices: np.ndarray,
                 evidence_path: Path = EVIDENCE_PATH):
        self.config = DynamicConfig()
        if inputs.v_names != list(self.config.v_names):
            raise ValueError("Model-input measures do not match the model")
        self.v_names, self.years = inputs.v_names, inputs.years
        self.selected_states = list(selected_states)
        self.draw_indices = np.asarray(draw_indices, dtype=int)
        self.units = self._select_units(inputs)
        self.unit_ids = [str(u.id) for u in self.units]
        self.SigmaY = inputs.covariance
        self.M = np.eye(len(self.v_names), dtype=int) + self.config.allowed_mask().astype(int)
        # Input draws for these units: draw x unit x measure x period (in fitting order)
        self.draws = np.asarray(inputs.observed_draws, dtype=float)[self.draw_indices][:, inputs.unit_index(self.unit_ids)]

        self.spec = specification(name, self.config, self.v_names, evidence_path)
        self.source, self.target = self.v_names.index("stigma_ahs"), self.v_names.index("prep_used")
        self.slopes, self.aprs = evidence_slopes(self.units, self.config, self.v_names, self.draw_indices,
                                                 self.spec.evidence_path)
        self.fits: list[list[UnitFit]] = []
        self.diagnostics: list[dict] = []

    def _select_units(self, inputs: StateInputs):
        """The nation, the divisions and the selected states, in the model-input order."""
        keep = set(self.selected_states)
        units = [u for u in inputs.units if u.kind != "state" or str(u.id) in keep]
        missing = keep - {str(u.id) for u in units if u.kind == "state"}
        if missing:
            raise ValueError(f"Selected states missing from the model inputs: {sorted(missing)}")
        # Each unit is shrunk toward its parent, so the parent must be fitted first: the nation before any
        # division, and a state's division before the state.
        fitted = set()
        for u in units:
            parent = {"division": "nation", "state": f"division {u.census_div}"}.get(u.kind)
            if parent is not None and parent not in fitted:
                raise ValueError(f"{u.kind} {u.id} comes before its parent ({parent})")
            fitted.add("nation" if u.kind == "nation" else f"division {u.census_div}" if u.kind == "division" else u.id)
        return units

    def unit_data(self, d: int) -> list[np.ndarray]:
        """Each unit's proportions (measure x period) in draw ``d``; for the placebo, the permuted
        measures' periods reordered."""
        rows = [self.v_names.index(v) for v in self.spec.permuted]
        unit_Y = []
        for k in range(len(self.units)):
            Y = self.draws[d, k]
            if rows:
                Y = _permute_periods(Y, rows, np.random.default_rng([int(self.draw_indices[d]), k]))
            unit_Y.append(Y)
        return unit_Y

    def fit_draw(self, d: int) -> list[UnitFit]:
        """Fit draw ``d`` through the hierarchy."""
        return fit_hierarchy(self.units, self.unit_data(d), self.slopes[d], self.spec.fit, self.SigmaY, self.M,
                             self.source, self.target)

    def run(self) -> "DynamicFitRunner":
        """Fit every draw, collecting the fits and one diagnostics row per draw and unit."""
        t0 = time.time()
        for d in range(len(self.draw_indices)):
            draw_fits = self.fit_draw(d)
            self.fits.append(draw_fits)
            self.diagnostics += [self._diagnostics_row(d, unit, fit) for unit, fit in zip(self.units, draw_fits)]
            success = np.mean([fit.diagnostics.get("success", False) for fit in draw_fits])
            print(f"{self.spec.name}: draw {d + 1}/{len(self.draw_indices)}; success={success:.3f}; "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
        return self

    def _diagnostics_row(self, d: int, unit, fit: UnitFit) -> dict:
        """Optimizer outcome and loss components for one unit in one draw."""
        diag = fit.diagnostics
        components = diag.get("final_components", {})
        return {
            "variant": self.spec.name, "draw": d, "source_draw": int(self.draw_indices[d]),
            "unit_id": str(unit.id), "unit_kind": str(unit.kind),
            "success": bool(diag.get("success", False)), "optimizer_status": diag.get("status"),
            "optimizer_message": diag.get("message"), "final_loss": float(diag.get("final_loss", np.nan)),
            "hierarchy_weight": np.nan if fit.hierarchy_weight is None else float(fit.hierarchy_weight),
            "best_start": diag.get("best_start"),
            "start_loss_spread": float(np.ptp(diag.get("start_losses", [np.nan]))),
            **{c: float(components.get(c, np.nan)) for c in LOSS_COMPONENTS},
        }

    def save(self, output_path: Path) -> None:
        """Write fitted arrays and diagnostics.

        ``J`` is draw x unit x target measure x source measure. ``drift`` and ``anchor_logits`` are
        draw x unit x measure; evidence quantities and success flags are draw x unit.
        """
        def stack(value, dtype=np.float32):
            return np.array([[value(fit) for fit in draw_fits] for draw_fits in self.fits], dtype=dtype)

        horizon = self.spec.fit.opt.evidence_horizon
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path, variant=np.array(self.spec.name), J=stack(lambda f: f.J),
            drift=stack(lambda f: f.drift), anchor_logits=stack(lambda f: f.anchor),
            success_stack=stack(lambda f: f.diagnostics.get("success", False), bool),
            unit_ids=np.asarray(self.unit_ids), v_names=np.asarray(self.v_names), years=self.years,
            input_draw_indices=self.draw_indices, selected_states=np.asarray(self.selected_states),
            ahs_prep_evidence_slope=self.slopes, ahs_prep_apr=self.aprs,              # draw x unit
            ahs_prep_evidence_path=np.asarray(str(self.spec.evidence_path)),
            ahs_prep_shift_response=stack(lambda f: sustained_shift_response(f.J, self.source, self.target, horizon)),
            evidence_horizon=np.array(horizon),
        )
        pd.DataFrame(self.diagnostics).to_csv(output_path.with_suffix(".loss_components.csv"), index=False)


def fit_specification(name: str, draw_indices: np.ndarray, selected_states: list[str], inputs: StateInputs,
                      output_path: Path, evidence_path: Path = EVIDENCE_PATH) -> None:
    """Fit one specification to the selected draws and save the results (scripts/3_model/03)."""
    DynamicFitRunner(name, inputs, selected_states, draw_indices, evidence_path).run().save(output_path)
