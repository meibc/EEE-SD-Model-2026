"""Hierarchical dynamic-model fits across input draws, for each model specification.

For each input draw, the nation, the nine Census divisions and the selected states are fitted in that order; each
division is shrunk toward the national transition matrix and each state toward its division (fit_hierarchy,
models/dynamic/estimation/hierarchy.py; every unit by JacobianEstimator). The fit is in state-anchored logit space:
each measure is centred at the unit's own mean logit over the observed periods, so the pathways act on departures
from a unit's usual level and fixed differences between units (access, local climate) stay in the anchor. The fit
uses the unit's trajectory, the pathway and persistence priors, the drift priors, shrinkage toward the parent unit
and the stability penalty.

The model's PrEP structure (``model_structure``; models/dynamic/mixture.py):
  - AHS enters PrEP through the respondent-level mixture P = P0 (1 - A (1 - r)). Each draw takes each unit's
    access-adjusted AHS-PrEP prevalence ratio r (scripts/2_evidence/01) from the pooled estimate and its posterior
    SE (log scale). r is a total effect (it includes the routes through disclosure and testing).
  - The PrEP row of the state is P0 / ceiling (each unit's willingness ceiling, scripts/2_evidence/04) and follows
    logistic diffusion: persistence held at 1, no inputs, so logit(P0 / ceiling) moves by the unit's PrEP drift
    (growth rate) each step; the growth-rate prior mean is the random-effects mean of the observed state slopes.
  - PrEP -> AHS is the held feedback b, acting on population PrEP outside J: a known input to AHS in the fit
    (feedback_offsets), applied each year in the projection (pipeline/projection.py).

Model specifications (config/specifications.py) set b and, for the sensitivity analyses, change one component.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from config.dynamic import DynamicConfig
from config.fit import EVIDENCE_SEED
from config.optimization import OptimConfig
from config.shrinkage import ShrinkageConfig
from config.specifications import PERSISTENCE_CAP, SPECIFICATIONS
from data.state_inputs import StateInputs
from models.dynamic.estimation.hierarchy import FitSettings, UnitFit, fit_hierarchy
from models.dynamic.mixture import feedback_input, prep_without_ahs

INPUTS_DIR = Path(__file__).resolve().parents[1] / "results/02_inputs"
EVIDENCE_PATH = INPUTS_DIR / "ahs_prep_evidence.csv"                       # scripts/2_evidence/01
EVIDENCE_ALL_YEARS_PATH = INPUTS_DIR / "ahs_prep_evidence_all_years.csv"   # scripts/2_evidence/01
PREP_CEILING_PATH = INPUTS_DIR / "prep_ceiling.csv"                        # scripts/2_evidence/04
PREP_GROWTH_PATH = INPUTS_DIR / "prep_growth.csv"                          # scripts/2_evidence/04
LOSS_COMPONENTS = ("LX", "LJ", "L_prior_J", "L_prior_c", "L_stab")


# =============================================================================
# Specifications and model structure
# =============================================================================

@dataclass
class Specification:
    """One model specification: its fit settings, the PrEP -> AHS feedback b and the AHS-PrEP evidence file."""
    name: str
    fit: FitSettings
    feedback: float
    evidence_path: Path


def model_structure(config: DynamicConfig, v_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The pathways estimated in J (M: 1 = estimated) and the entries held at a value (fixed: NaN = estimated).

    PrEP row (P0 / ceiling): persistence held at 1 and no inputs (logistic diffusion); AHS reaches PrEP through
    the mixture, so neither AHS -> PrEP nor the routes covered by r are in J. PrEP -> AHS is the feedback b,
    outside J.
    """
    ahs, prep = v_names.index("stigma_ahs"), v_names.index("prep_used")
    M = np.eye(len(v_names), dtype=int) + config.allowed_mask().astype(int)
    M[prep] = 0
    M[prep, prep] = 1
    M[ahs, prep] = 0
    fixed = np.full((len(v_names), len(v_names)), np.nan)
    fixed[prep, prep] = 1.0
    return M, fixed


def specification(name: str, config: DynamicConfig, v_names: list[str]) -> Specification:
    """The model's settings with this specification's feedback and change applied."""
    if name not in SPECIFICATIONS:
        raise ValueError(f"Unknown specification {name!r}; expected one of {list(SPECIFICATIONS)}")
    change = SPECIFICATIONS[name]
    opt, shrink = OptimConfig(), ShrinkageConfig()
    sign, prior_mean, prior_tau = config.structural_prior_arrays()
    drift_enabled, drift_mean, drift_tau = config.drift_prior_arrays()

    if change.get("shrinkage_off"):
        shrink.wJ = 0.0
        shrink.wJ_state = 0.0
    if "persistence_shift" in change:
        idx = np.diag_indices(len(v_names))
        targeted = np.isfinite(prior_mean[idx])
        shifted = np.minimum(prior_mean[idx] + change["persistence_shift"], PERSISTENCE_CAP)
        prior_mean[idx] = np.where(targeted, shifted, prior_mean[idx])

    _, fixed = model_structure(config, v_names)
    prior_mean = np.where(np.isfinite(fixed), np.nan, prior_mean)   # no prior on held entries
    # PrEP drift = growth rate of logit(P0 / ceiling) per step: prior mean from the observed state S-curves, prior
    # SD left weak (config) so each state's growth follows its own data
    drift_mean[v_names.index("prep_used")] = pd.read_csv(PREP_GROWTH_PATH).set_index("state").loc["prior", "slope"]

    fit = FitSettings(opt=opt, shrink=shrink, sign=sign, prior_mean=prior_mean, prior_tau=prior_tau,
                      drift_enabled=drift_enabled, drift_mean=drift_mean, drift_tau=drift_tau, fixed=fixed)
    return Specification(name=name, fit=fit, feedback=float(change["feedback"]),
                         evidence_path=EVIDENCE_ALL_YEARS_PATH if change.get("evidence") == "all_years" else EVIDENCE_PATH)


def prevalence_ratios(units, draw_indices: np.ndarray, evidence_path: Path = EVIDENCE_PATH) -> np.ndarray:
    """Each unit's access-adjusted AHS-PrEP prevalence ratio, per draw (draw x unit).

    States use their pooled (empirical-Bayes) ratio; divisions, the nation and any state
    without an estimate use the random-effects mean. Each draw takes the ratio from
    N(log pooled ratio, posterior SE) (seeded by the source draw index, so draws are paired
    across specifications).
    """
    evidence = pd.read_csv(evidence_path).set_index("state")
    rows = [evidence.loc[str(u.id)] if str(u.id) in evidence.index else evidence.loc["POOLED"] for u in units]
    log_apr = np.array([float(r["log_apr_pooled"]) for r in rows])
    sd = np.array([float(r["se_pooled"]) for r in rows])
    z = np.array([np.random.default_rng([EVIDENCE_SEED, int(d)]).standard_normal(len(units)) for d in draw_indices])
    return np.exp(log_apr + sd * z)


def prep_ceilings(unit_ids: list[str]) -> np.ndarray:
    """Each unit's PrEP ceiling (states shrunk toward the pooled value; scripts/2_evidence/04)."""
    table = pd.read_csv(PREP_CEILING_PATH).set_index("unit_id")["ceiling"]
    missing = sorted(set(unit_ids) - set(table.index))
    if missing:
        raise ValueError(f"No PrEP ceiling for units {missing} ({PREP_CEILING_PATH})")
    return table.loc[unit_ids].to_numpy(dtype=float)


# =============================================================================
# Runner
# =============================================================================

class DynamicFitRunner:
    """Fits one model specification to every selected input draw: the nation, the Census divisions and
    the selected states (models/dynamic/estimation/hierarchy.py), collecting the fits and their diagnostics."""

    def __init__(self, name: str, inputs: StateInputs, selected_states: list[str], draw_indices: np.ndarray):
        self.config = DynamicConfig()
        if inputs.v_names != list(self.config.v_names):
            raise ValueError("Model-input measures do not match the model")
        self.v_names, self.years = inputs.v_names, inputs.years
        self.ahs, self.prep = self.v_names.index("stigma_ahs"), self.v_names.index("prep_used")
        self.selected_states = list(selected_states)
        self.draw_indices = np.asarray(draw_indices, dtype=int)          # positions in the model-input file
        self.source_draw_ids = inputs.source_draw_ids[self.draw_indices]  # their bootstrap IDs (scripts/1_data/04)
        self.units = self._select_units(inputs)
        self.unit_ids = [str(u.id) for u in self.units]
        # Input draws for these units: draw x unit x measure x period (in fitting order)
        self.draws = np.asarray(inputs.observed_draws, dtype=float)[self.draw_indices][:, inputs.unit_index(self.unit_ids)]

        self.spec = specification(name, self.config, self.v_names)
        self.M, _ = model_structure(self.config, self.v_names)
        self.aprs = prevalence_ratios(self.units, self.draw_indices, self.spec.evidence_path)
        self.ceilings = prep_ceilings(self.unit_ids)
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
        """The proportions the fit models in draw ``d`` (measure x period, by unit): observed, with PrEP replaced by
        P0 / ceiling."""
        out = []
        for k in range(len(self.units)):
            Y = self.draws[d, k].copy()
            Y[self.prep] = np.clip(prep_without_ahs(Y[self.prep], Y[self.ahs], self.aprs[d, k]) / self.ceilings[k],
                                   0.0, 0.999)
            out.append(Y)
        return out

    def feedback_offsets(self, d: int) -> list[np.ndarray]:
        """Each unit's known input per step (measure x period): the held PrEP -> AHS feedback acting on observed
        population PrEP (feedback_input)."""
        offsets = []
        for k in range(len(self.units)):
            prep = self.draws[d, k, self.prep]
            offset = np.zeros_like(self.draws[d, k])
            offset[self.ahs] = feedback_input(prep, prep, self.spec.feedback)
            offsets.append(offset)
        return offsets

    def fit_draw(self, d: int) -> list[UnitFit]:
        """Fit draw ``d`` through the hierarchy."""
        return fit_hierarchy(self.units, self.unit_data(d), self.spec.fit, self.M, self.feedback_offsets(d))

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
            "variant": self.spec.name, "draw": d, "input_draw_index": int(self.draw_indices[d]),
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
        draw x unit x measure; prevalence ratios and success flags are draw x unit.
        """
        def stack(value, dtype=np.float32):
            return np.array([[value(fit) for fit in draw_fits] for draw_fits in self.fits], dtype=dtype)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path, variant=np.array(self.spec.name), J=stack(lambda f: f.J),
            drift=stack(lambda f: f.drift), anchor_logits=stack(lambda f: f.anchor),
            success_stack=stack(lambda f: f.diagnostics.get("success", False), bool),
            unit_ids=np.asarray(self.unit_ids), v_names=np.asarray(self.v_names), years=self.years,
            input_draw_indices=self.draw_indices, source_draw_ids=self.source_draw_ids,
            selected_states=np.asarray(self.selected_states),
            ahs_prep_apr=self.aprs,                                                     # draw x unit
            ahs_prep_evidence_path=np.asarray(str(self.spec.evidence_path)),
            mixture_prep_to_ahs=np.array(self.spec.feedback),                          # b
            mixture_prep_ceiling=self.ceilings,                                         # by unit
        )
        pd.DataFrame(self.diagnostics).to_csv(output_path.with_suffix(".loss_components.csv"), index=False)


def fit_specification(name: str, draw_indices: np.ndarray, selected_states: list[str], inputs: StateInputs,
                      output_path: Path) -> None:
    """Fit one specification to the selected draws and save the results (scripts/3_model/03)."""
    DynamicFitRunner(name, inputs, selected_states, draw_indices).run().save(output_path)
