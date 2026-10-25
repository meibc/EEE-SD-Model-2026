"""Hierarchical state-model fits across input draws, and their scenario projections.

For each input draw, the nation, the nine Census divisions and the selected
states are fitted in that order; each division is shrunk toward the national
transition matrix and each state toward its division (``ShrinkageCalculator``).
Every unit is fitted by ``JacobianEstimator`` in state-anchored logit space: each
measure is centred at the unit's own mean logit over the observed periods, so the
pathways act on departures from a unit's usual level and fixed differences between
units (access, local climate) stay in the anchor. The fit uses the unit's
trajectory, the pooled covariance target, the structural and diagonal priors, the
drift priors, the stability penalty and an evidence target: the unit's total
AHS -> PrEP response over the projection horizon (every route and feedback acting)
should match its access-adjusted individual-level association (scripts/08b;
models/sbm/calibration.py). The association is estimated on the old AHS wording
(2017-2021 cycles, the scale of the wording-adjusted AHS) and each draw takes its
own prevalence ratio from the pooled estimate and its posterior SE (log scale), so
the target carries the evidence's uncertainty. The theory priors decide how that
total is split across the direct, disclosure, healthcare-contact and testing routes.

Model specifications (supplement S7) change one component of the primary model:
    primary               the primary model
    trajectory_only       covariance-fit weight 0
    covariance_moderate   covariance-fit weight 0.10 (primary 0.01)
    covariance_balanced   covariance-fit weight 0.40
    sign_only             off-diagonal prior means and scales removed (signs and zeros kept)
    common_weak           every permitted off-diagonal pathway: prior mean 0, tau 0.70
    theory_prior          no evidence target on the total AHS -> PrEP response (theory priors only)
    no_hierarchy          geographic shrinkage weights 0
    half_prep_drift       PrEP drift prior mean halved
    zero_prep_drift       PrEP drift prior mean 0
    pooled_prep_drift     PrEP drift prior scale 0.35 (primary 1.0): states pooled toward one expansion
    loose_ahs_drift       AHS drift prior scale 1.0 (primary 0.35): each state's AHS can follow its own
                          trend, so a shared upward trend need not load onto PrEP -> AHS
    cluster_moderate      PrEP/risk/testing/disclosure/contact pathways centred at 0.10 (primary 0.05)
    persistence_low       targeted persistence priors 0.10 lower (PrEP, contact, GSS, family)
    persistence_high      targeted persistence priors 0.10 higher (capped at 0.98)
    stability_w10         stability-penalty weight 10 (primary 10,000)
    placebo_ahs           AHS periods permuted within each unit and draw (timing destroyed)
    placebo_engagement    disclosure and healthcare-contact periods permuted likewise
    evidence_all_years    evidence target from all AMIS cycles (both AHS wordings)
    evidence_fixed_apr    evidence target at the pooled prevalence ratio, no draw-to-draw uncertainty
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from config.interventions import PAPER_SCENARIOS, SCENARIO_CODEBOOK
from config.optimization import OptimConfig
from config.sem import SEMConfig
from config.shrinkage import ShrinkageConfig
from data.data_prep import DataPrep
from input.research_bundle import build_research_bundle
from input.standardized_runtime import StandardizedBundle, StandardizedCDCParamsLoader, StandardizedSEMParamsLoader
from models.sbm.calibration import evidence_slope, shift_response
from models.sbm.estimation.jacobian import JacobianEstimator
from models.sbm.estimation.shrinkage import ShrinkageCalculator
from models.sbm.estimation.transforms import FeatureTransformer
from models.shared.alignment import build_model_years
from pipeline.joint_simulation import run_uncertainty

SPECIFICATIONS = (
    "primary", "trajectory_only", "covariance_moderate", "covariance_balanced", "sign_only",
    "common_weak", "theory_prior", "no_hierarchy", "half_prep_drift", "zero_prep_drift",
    "pooled_prep_drift", "cluster_moderate", "persistence_low", "persistence_high", "stability_w10",
    "placebo_ahs", "placebo_engagement", "evidence_all_years", "evidence_fixed_apr", "loose_ahs_drift",
)
STIGMA = ("stigma_ahs", "stigma_gss", "stigma_family")
EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "results/02_inputs/ahs_prep_evidence.csv"
EVIDENCE_ALL_YEARS_PATH = EVIDENCE_PATH.with_name("ahs_prep_evidence_all_years.csv")
EVIDENCE_SEED = 20261023
PLACEBO_PERMUTED = {"placebo_ahs": ("stigma_ahs",), "placebo_engagement": ("out_gid", "seehcp")}
SPECIFICATION_SCENARIOS = PAPER_SCENARIOS
LOSS_COMPONENTS = ("LX", "LR", "LJ", "L_prior_J", "L_prior_c", "L_stab", "L_total_effect")


def specification_settings(name: str, config: SEMConfig, v_names: list[str]) -> dict:
    """Optimizer, shrinkage and prior settings for one model specification."""
    if name not in SPECIFICATIONS:
        raise ValueError(f"Unknown specification {name!r}; expected one of {SPECIFICATIONS}")
    opt, shrink = OptimConfig(), ShrinkageConfig()
    sign, prior_mean, prior_tau = config.structural_prior_arrays()
    drift_enabled, drift_mean, drift_tau = config.drift_prior_arrays()
    off_diagonal = ~np.eye(len(v_names), dtype=bool)
    prep = v_names.index("prep_used")

    if name == "trajectory_only":
        opt.wR = 0.0
    elif name == "covariance_moderate":
        opt.wR = 0.10
    elif name == "covariance_balanced":
        opt.wR = 0.40
    elif name == "sign_only":
        prior_mean[off_diagonal] = np.nan
        prior_tau[off_diagonal] = np.nan
    elif name == "common_weak":
        allowed = config.allowed_mask()
        prior_mean[allowed] = 0.0
        prior_tau[allowed] = 0.70
    elif name == "no_hierarchy":
        shrink.wJ = 0.0
        shrink.wJ_state = 0.0
    elif name == "half_prep_drift":
        drift_mean[prep] *= 0.5
    elif name == "zero_prep_drift":
        drift_mean[prep] = 0.0
    elif name == "pooled_prep_drift":
        drift_tau[prep] = 0.35
    elif name == "loose_ahs_drift":
        drift_tau[v_names.index("stigma_ahs")] = 1.0
    elif name == "cluster_moderate":
        stigma = np.isin(np.asarray(v_names), STIGMA)
        cluster = config.allowed_mask() & ~stigma[:, None] & ~stigma[None, :]
        prior_mean[cluster] = 0.10
    elif name in ("persistence_low", "persistence_high"):
        idx = np.diag_indices(len(v_names))
        targeted = np.isfinite(prior_mean[idx])
        shift = -0.10 if name == "persistence_low" else 0.10
        prior_mean[idx] = np.where(targeted, np.minimum(prior_mean[idx] + shift, 0.98), prior_mean[idx])
    elif name == "stability_w10":
        opt.w_stab = 10.0
    evidence_prior = name not in ("sign_only", "common_weak", "theory_prior")
    return dict(opt=opt, shrink=shrink, sign=sign, prior_mean=prior_mean, prior_tau=prior_tau,
                drift_enabled=drift_enabled, drift_mean=drift_mean, drift_tau=drift_tau,
                evidence_prior=evidence_prior, permuted=PLACEBO_PERMUTED.get(name, ()),
                evidence_all_years=name == "evidence_all_years", apr_uncertainty=name != "evidence_fixed_apr")


def evidence_slopes(units, config: SEMConfig, v_names: list[str], draw_indices: np.ndarray,
                    evidence_path: Path = EVIDENCE_PATH, uncertainty: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """AHS -> PrEP response implied by each unit's access-adjusted prevalence ratio, per draw.

    States use their pooled (empirical-Bayes) ratio; divisions, the nation and any state
    without an estimate use the random-effects mean. With ``uncertainty`` each draw takes the
    ratio from N(log pooled ratio, posterior SE) (seeded by the source draw index, so draws are
    paired across specifications); otherwise the pooled ratio itself. Levels are the unit's mean
    AHS and PrEP over the observed periods; the benchmark is the AHS reference probability.
    Returns (slopes, prevalence ratios), each draw x unit.
    """
    evidence = pd.read_csv(evidence_path).set_index("state")
    a, p = v_names.index("stigma_ahs"), v_names.index("prep_used")
    benchmark = float(config.reference_probs()[a])
    levels, log_apr, sd = [], [], []
    for unit in units:
        values = np.asarray(unit.amis_values, dtype=float)
        levels.append(tuple(float(np.mean(values[i][values[i] > 0])) for i in (a, p)))
        row = evidence.loc[str(unit.id)] if str(unit.id) in evidence.index else evidence.loc["POOLED"]
        log_apr.append(float(row["log_apr_pooled"]))
        sd.append(float(row["se_pooled"]) if uncertainty else 0.0)
    z = np.array([np.random.default_rng([EVIDENCE_SEED, int(d)]).standard_normal(len(units)) for d in draw_indices])
    apr = np.exp(np.asarray(log_apr) + np.asarray(sd) * z)
    slopes = np.array([[evidence_slope(ahs, prep, r, benchmark) for (ahs, prep), r in zip(levels, row)] for row in apr])
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


def fit_specification(name: str, draw_indices: np.ndarray, selected_states: list[str],
                      input_draws_path: Path, workbook_path: Path, output_path: Path,
                      evidence_path: Path = EVIDENCE_PATH) -> None:
    """Fit the hierarchy for every selected input draw and save the stacked results.

    Writes ``output_path`` (NPZ: Jmeans_stack, drift_stack, anchor_logits, success_stack, ...)
    and ``<output_path>.loss_components.csv`` (one row per draw and unit).
    """
    inputs = np.load(input_draws_path, allow_pickle=True)
    draw_indices = np.asarray(draw_indices, dtype=int)
    draws = np.asarray(inputs["Xbars_samples"], dtype=float)[draw_indices]
    input_units = inputs["unit_order"].astype(str).tolist()
    input_vars = inputs["v_names"].astype(str).tolist()
    input_years = np.asarray(inputs["ts"], dtype=int)

    config = SEMConfig(data_path=workbook_path)
    prepared = DataPrep(config).prepare_inputs()
    keep = set(selected_states)
    prepared.units = [u for u in prepared.units if u.kind != "state" or str(u.id) in keep]
    fitted_states = {str(u.id) for u in prepared.units if u.kind == "state"}
    if fitted_states != keep:
        raise ValueError(f"Selected states missing from prepared data: {sorted(keep - fitted_states)}")
    if input_vars != list(prepared.v_names) or not np.array_equal(input_years, prepared.ts):
        raise ValueError("Input-draw variables/years do not match the model")
    unit_order = [str(u.id) for u in prepared.units]
    draw_unit = [input_units.index(uid) for uid in unit_order]

    s = specification_settings(name, config, input_vars)
    a, p = input_vars.index("stigma_ahs"), input_vars.index("prep_used")
    if s["evidence_all_years"]:
        evidence_path = EVIDENCE_ALL_YEARS_PATH
    if s["evidence_prior"]:
        slopes, aprs = evidence_slopes(prepared.units, config, input_vars, draw_indices, evidence_path,
                                       uncertainty=s["apr_uncertainty"])
    else:
        slopes = aprs = np.full((len(draw_indices), len(unit_order)), np.nan)
    permuted = [input_vars.index(v) for v in s["permuted"]]
    shrink_calc = ShrinkageCalculator(s["shrink"])
    m = len(input_vars)
    J_stack = np.full((len(draw_indices), len(unit_order), m, m), np.nan, dtype=np.float32)
    drift_stack = np.full((len(draw_indices), len(unit_order), m), np.nan, dtype=np.float32)
    anchor_stack = np.full((len(draw_indices), len(unit_order), m), np.nan, dtype=np.float32)
    total_effect_stack = np.full((len(draw_indices), len(unit_order)), np.nan, dtype=np.float32)
    horizon = s["opt"].total_effect_horizon
    success_stack = np.zeros((len(draw_indices), len(unit_order)), dtype=bool)
    loss_rows = []
    t0 = time.time()

    def fit_unit(Y, J_ref, wJ, total_effect):
        anchor = FeatureTransformer.state_anchor(Y)
        Xbar, SigmaX = FeatureTransformer(reference_logits=anchor).transform(Y, prepared.SigmaY)
        estimator = JacobianEstimator(s["opt"])
        J = estimator.fit(
            Xbar=Xbar, SigmaX=SigmaX, M=prepared.M, sign_mask=s["sign"], u=np.zeros(m),
            J_ref=J_ref, wJ=wJ,
            structural_prior_mean=s["prior_mean"], structural_prior_tau=s["prior_tau"],
            drift_enabled=s["drift_enabled"], drift_prior_mean=s["drift_mean"],
            drift_prior_tau=s["drift_tau"], total_effect=total_effect,
        )
        return J, anchor, estimator

    for d in range(len(draw_indices)):
        us_J, division_J = None, {}
        unit_Y = []
        for k, unit in enumerate(prepared.units):
            Y = draws[d][:, :, draw_unit[k]]
            if permuted:
                Y = _permute_periods(Y, permuted, np.random.default_rng([int(draw_indices[d]), k]))
            unit_Y.append(Y)
        for k, unit in enumerate(prepared.units):
            Y = unit_Y[k]
            unit.amis_values = Y
            total_effect = None
            if s["evidence_prior"]:
                total_effect = {"source": a, "target": p, "horizon": horizon,
                                "value": float(slopes[d, k]), "tau": s["opt"].total_effect_tau}
            J_ref, wJ = shrink_calc.get_params(unit, us_J, division_J)
            J, anchor, estimator = fit_unit(Y, J_ref, wJ, total_effect)
            J_stack[d, k] = J
            anchor_stack[d, k] = anchor
            total_effect_stack[d, k] = shift_response(J, a, p, horizon)
            drift_stack[d, k] = estimator.last_drift
            diag = estimator.last_diagnostics
            components = diag.get("final_components", {})
            success_stack[d, k] = bool(diag.get("success", False))
            loss_rows.append({
                "variant": name, "draw": d, "unit_id": str(unit.id), "unit_kind": str(unit.kind),
                "success": success_stack[d, k], "optimizer_status": diag.get("status"),
                "optimizer_message": diag.get("message"), "final_loss": float(diag.get("final_loss", np.nan)),
                "hierarchy_weight": np.nan if wJ is None else float(wJ),
                "best_start": diag.get("best_start"),
                "start_loss_spread": float(np.ptp(diag.get("start_losses", [np.nan]))),
                **{c: float(components.get(c, np.nan)) for c in LOSS_COMPONENTS},
            })
            if unit.kind == "nation":
                us_J = J
            elif unit.kind == "division":
                division_J[unit.census_div] = J
        print(f"{name}: draw {d + 1}/{len(draw_indices)}; success={success_stack[d].mean():.3f}; "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, variant=np.array(name), Jmeans_stack=J_stack, drift_stack=drift_stack,
        anchor_logits=anchor_stack, success_stack=success_stack, unit_order=np.asarray(unit_order),
        v_names=np.asarray(input_vars), ts=input_years, source_draw_indices=draw_indices,
        selected_states=np.asarray(selected_states),
        ahs_prep_evidence_slope=slopes, ahs_prep_apr=aprs,                       # draw x unit
        ahs_prep_evidence_path=np.asarray(str(evidence_path) if s["evidence_prior"] else ""),
        ahs_prep_total_effect=total_effect_stack, total_effect_horizon=np.array(horizon),
    )
    pd.DataFrame(loss_rows).to_csv(output_path.with_suffix(".loss_components.csv"), index=False)


def specification_scenarios(name: str, fit_path: Path, epi_base_path: Path, workbook_path: Path,
                            input_draws_path: Path, n_samples: int, seed: int, target_year: int,
                            output_path: Path, constant_pr_prep: bool = False) -> None:
    """Incidence change in ``target_year`` under the paper scenarios, by state and draw.

    With ``constant_pr_prep`` the scenario PrEP change comes from the respondent-level prevalence
    ratio (pipeline/joint_simulation.py), as in the primary scenario analysis.
    """
    selected_states = np.load(fit_path, allow_pickle=True)["selected_states"].astype(str).tolist()
    with tempfile.TemporaryDirectory(prefix="eee_specification_") as tmp:
        bundle_path = Path(tmp) / f"{name}.npz"
        build_research_bundle(epi_base_path, fit_path, bundle_path, workbook_path, input_draws_path,
                              require_success=False)
        bundle = StandardizedBundle(bundle_path)
        sem_loader, cdc_loader = StandardizedSEMParamsLoader(bundle), StandardizedCDCParamsLoader(bundle)
        units = bundle.build_units()
        model_years = build_model_years(cdc_loader.years, target_end_year=target_year)

        def incidence(state_codes, relationship_codes, prep_only_fraction=None):
            out = run_uncertainty(
                sem_loader, cdc_loader, units, unit_ids=selected_states, n_samples=n_samples, seed=seed,
                show_progress=False, model_years=model_years, state_intervention_codes=state_codes,
                relationship_intervention_codes=relationship_codes, intervention_duration_steps=3,
                prep_only_fraction=prep_only_fraction, constant_pr_prep=constant_pr_prep,
            )
            year_idx = int(np.flatnonzero(np.asarray(out.years, dtype=int) == target_year)[0])
            return {st: np.array([s.cdc_output.incidence[year_idx] for s in out.results[st].samples])
                    for st in selected_states}

        baseline = incidence([], [])
        rows = []
        for code in SPECIFICATION_SCENARIOS:
            spec = SCENARIO_CODEBOOK[code]
            scenario = incidence(spec.get("state_codes", []), spec.get("relationship_codes", []), spec.get("prep_only_fraction"))
            for st in selected_states:
                b, sc = baseline[st], scenario[st]
                pct = np.where(np.abs(b) > 1e-12, 100.0 * (sc - b) / b, np.nan)
                rows += [{"variant": name, "state": st, "scenario": code, "draw": i, "year": target_year,
                          "baseline_incidence": float(bv), "scenario_incidence": float(sv), "pct_change": float(pv)}
                         for i, (bv, sv, pv) in enumerate(zip(b, sc, pct))]
    pd.DataFrame(rows).to_csv(output_path, index=False)
