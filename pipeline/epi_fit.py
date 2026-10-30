from __future__ import annotations

"""Hierarchical Bayesian calibration of the epidemiological model, and its bundle.

For every state with AMIS and CDC data, fits

    Inc_{s,t}   = beta_s * no_vs_{s,t} * (1 - PrEP_{s,t}) * (risk_{s,t} / risk0_s)^alpha
    U_{s,t+1}   = U_{s,t} + Inc_{s,t} - Dx_{s,t},     Dx_{s,t} = delta_{s,t} * U_{s,t}
    delta_{s,t} = 1 - exp(-kappa_s * test_{s,t} * m_{s,t}),  m = post_multiplier_s from 2021

with beta and kappa hierarchical by Census division, one bounded alpha shared by all
states, and U0 = rho_s * estimated prevalence in the first year. The likelihood is
CDC estimated incidence (negative binomial, annual, 2020 excluded) and MSM
diagnoses (lognormal on period means).

Inputs (all built in this repository):
  AMIS PrEP use, risk behaviour and HIV testing   state point estimates and Census
      divisions from the model-input file (scripts/3_model/01); state-periods without an
      estimate take the median of the state's input draws, which scripts/1_data/04 fills
      from its division. Each period sits at its end year and is interpolated to
      single years.
  CDC MSM stocks                                   results/00_cdc/cdc_stocks.csv (read once)
The fit window runs from 2017 to the last year in which most states have CDC
diagnoses, estimated prevalence and viral suppression; states with a gap in the
window are excluded, and every diagnosis period inside the window enters the
likelihood.

``write_epi_bundle`` converts the posterior into the arrays the scenarios read
(artifacts.standardized_runtime).
"""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from data.state_inputs import StateInputs
from models.epi.params import POST_YEAR, RISK_BASELINE_FLOOR, RISK_RATIO_BOUNDS
from core.years import align_to_years
from core.transforms import logit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FIRST_YEAR = 2017
BUNDLE_END_YEAR = 2036
SCHEMA_VERSION = 2
AMIS_MEASURES = ("prep_used", "risk_behavior", "hivtest12")
FIT_CDC_NAMES = ("HIV diagnoses", "Estimated HIV prevalence (MSM)", "HIV viral suppression")
BUNDLE_CDC_NAMES = ("PrEP Eligible", "Estimated HIV prevalence (MSM)", "HIV viral suppression",
                    "Estimated HIV incidence (MSM)", "HIV diagnoses", "PrEP")
POSTERIOR_TO_BUNDLE = {"beta_inc": "cdc_beta", "alpha": "cdc_alpha", "kappa": "cdc_kdx",
                       "U0": "cdc_U0", "post_multiplier": "cdc_post_multiplier"}

STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

# 2020 Census resident population; scales how far each state's beta may move
# from its division mean (larger states move more freely).
STATE_POPS = {
    "CA": 39538223, "TX": 29145505, "FL": 21538187, "NY": 20201249,
    "PA": 13002700, "IL": 12812508, "OH": 11799448, "GA": 10711908,
    "NC": 10439388, "MI": 10077331, "NJ": 9288994, "VA": 8631393,
    "WA": 7614893, "AZ": 7151502, "MA": 7029917, "TN": 6910840,
    "IN": 6785528, "MD": 6177224, "MO": 6154913, "WI": 5893718,
    "CO": 5773714, "MN": 5706494, "SC": 5118425, "AL": 5024279,
    "LA": 4657757, "KY": 4505836, "OR": 4237256, "OK": 3959353,
    "CT": 3605944, "UT": 3271616, "IA": 3190369, "NV": 3104614,
    "AR": 3011524, "MS": 2961279, "KS": 2937880, "NM": 2117522,
    "NE": 1961504, "ID": 1839106, "WV": 1793716, "HI": 1455271,
    "NH": 1377529, "ME": 1362359, "MT": 1084225, "RI": 1097379,
    "DE": 989948, "SD": 886667, "ND": 779094, "AK": 733391,
    "VT": 643077, "WY": 576851,
}


# =============================================================================
# Settings and input containers
# =============================================================================

@dataclass(frozen=True)
class HierarchicalEpiFitConfig:
    """Configuration for the long-running hierarchical epi calibration."""

    input_draws_path: Path = RESULTS / "02_inputs/state_inputs.npz"
    cdc_stocks_path: Path = RESULTS / "00_cdc/cdc_stocks.csv"
    output_dir: Path = RESULTS / "02_inputs/epi_fit"
    run_mode: str = "quick"
    diagnosis_periods: tuple[tuple[int, ...], ...] = ((2017, 2018, 2019), (2021, 2022), (2023, 2024))
    random_seed: int = 42
    quick_draws: int = 100
    quick_tune: int = 300
    quick_chains: int = 2
    quick_cores: int = 2
    full_draws: int = 500
    full_tune: int = 1000
    full_chains: int = 4
    full_cores: int = 4
    target_accept: float = 0.99


@dataclass(frozen=True)
class HierarchicalEpiInputs:
    geos: list[str]
    model_years: np.ndarray
    div_id: np.ndarray
    unique_div: list[int]
    no_vs: np.ndarray
    prep_off_prop: np.ndarray
    risk: np.ndarray
    p_test: np.ndarray
    hiv_prevalence: np.ndarray
    dx_obs: np.ndarray
    inc_obs: np.ndarray
    period_labels: list[str]
    period_idx: list[np.ndarray]
    period_targets: pd.DataFrame
    inc_g_idx: np.ndarray
    inc_t_idx: np.ndarray
    inc_obs_vec: np.ndarray
    risk_baseline: np.ndarray
    risk_ratio: np.ndarray
    log_risk_ratio: np.ndarray
    exposure: np.ndarray
    prevalence0: np.ndarray
    state_freedom: np.ndarray
    bundle_cdc: np.ndarray              # state x BUNDLE_CDC_NAMES x fit year, for write_epi_bundle


# =============================================================================
# Inputs: AMIS estimates and CDC stocks
# =============================================================================

def read_cdc_stocks(path: Path) -> pd.DataFrame:
    """MSM stocks, one row per state and indicator, one column per year (NaN where CDC has no value)."""
    stocks = pd.read_csv(path)
    return stocks[stocks["Population"].eq("MSM")].pivot_table(index=["St_Abbr", "Indicator"], columns="Year",
                                                              values="Cases")


def stock_array(wide: pd.DataFrame, geos: list[str], names, years) -> np.ndarray:
    """State x indicator x year array from ``read_cdc_stocks``."""
    wide = wide.reindex(index=pd.MultiIndex.from_product([geos, list(names)]), columns=list(years))
    return wide.to_numpy(dtype=float).reshape(len(geos), len(names), len(years))


def amis_estimates(inputs: StateInputs, geos: list[str]) -> np.ndarray:
    """State x measure x period AMIS estimates (AMIS_MEASURES order).

    Point estimates where the state-period has one; otherwise the median of the
    state's input draws, which scripts/1_data/04 fills from its division.
    """
    k = inputs.unit_index(geos)
    m = [inputs.v_names.index(v) for v in AMIS_MEASURES]
    point = inputs.observed_point[k][:, m]
    filled = np.median(np.asarray(inputs.observed_draws, dtype=float)[:, k][:, :, m], axis=0)
    return np.where(np.isfinite(point), point, filled)


def build_hierarchical_epi_inputs(config: HierarchicalEpiFitConfig) -> HierarchicalEpiInputs:
    """Build PyMC-ready arrays from the AMIS estimates and CDC stocks."""

    state_inputs = StateInputs.load(config.input_draws_path)
    stocks = read_cdc_stocks(config.cdc_stocks_path)
    candidates = [g for g in STATE_ABBRS if g in state_inputs.unit_ids]

    # Fit window: 2017 to the last year in which most states have diagnoses,
    # estimated prevalence and viral suppression; states with any gap in the
    # window are excluded.
    stock_years = np.arange(FIRST_YEAR, int(stocks.columns.max()) + 1)
    fit_cdc = stock_array(stocks, candidates, FIT_CDC_NAMES, stock_years)
    available = (np.isfinite(fit_cdc) & (fit_cdc > 0)).all(axis=1)          # state x year
    n_years = int(np.flatnonzero(available.mean(axis=0) > 0.5).max()) + 1
    keep = available[:, :n_years].all(axis=1)
    geos = [g for g, k in zip(candidates, keep) if k]
    excluded = [g for g, k in zip(candidates, keep) if not k]
    if excluded:
        print(f"Excluded from the epi fit (incomplete CDC data {stock_years[0]}-{stock_years[n_years - 1]}): {excluded}")
    if not geos or n_years < 2:
        raise ValueError("No states with complete CDC data; inspect cdc_stocks.csv")
    fit_cdc = fit_cdc[keep]
    model_years = stock_years[:n_years]
    periods = [p for p in config.diagnosis_periods if max(p) <= model_years[-1]]
    if not periods:
        raise ValueError("No diagnosis period lies inside the fit window")
    period_labels = [f"{min(p)}-{max(p)}" for p in periods]
    period_idx = [np.searchsorted(model_years, p) for p in periods]

    amis, amis_years = amis_estimates(state_inputs, geos), state_inputs.years
    prep_on_prop, risk, p_test = (
        np.clip(align_to_years(amis_years, amis[:, j, :], model_years), 1e-6, 1 - 1e-6) for j in range(3)
    )
    prep_off_prop = np.clip(1.0 - prep_on_prop, 1e-6, 1.0)

    dx_obs, hiv_prevalence, viral_suppression = (fit_cdc[:, i, :n_years] for i in range(3))
    no_vs = np.clip(hiv_prevalence - viral_suppression, 1e-6, None)
    inc_obs = stock_array(stocks, geos, ["Estimated HIV incidence (MSM)"], model_years)[:, 0]
    inc_obs = np.where(inc_obs > 0, inc_obs, np.nan)

    period_targets = pd.DataFrame([
        {"state": g, "period": label, "years": ",".join(map(str, p)),
         "dx_obs": dx_obs[gi, idx].mean(),
         "inc_obs": np.nanmean(inc_obs[gi, idx]) if np.isfinite(inc_obs[gi, idx]).any() else np.nan}
        for gi, g in enumerate(geos) for label, p, idx in zip(period_labels, periods, period_idx)
    ])

    inc_annual_mask = np.isfinite(inc_obs) & (model_years != 2020)[None, :]
    inc_g_idx, inc_t_idx = np.where(inc_annual_mask)
    inc_obs_vec = inc_obs[inc_g_idx, inc_t_idx].astype("int64")

    risk_baseline = np.maximum(risk[:, 0], RISK_BASELINE_FLOOR)
    risk_ratio = np.clip(risk / risk_baseline[:, None], *RISK_RATIO_BOUNDS)
    log_risk_ratio = np.log(risk_ratio)
    exposure = no_vs * prep_off_prop
    prevalence0 = hiv_prevalence[:, 0]

    raw_div = state_inputs.census_divisions[state_inputs.unit_index(geos)]
    unique_div = sorted(np.unique(raw_div).tolist())
    div_id = np.searchsorted(unique_div, raw_div)
    pop = np.array([STATE_POPS[g] for g in geos], dtype=float)
    state_freedom = 0.2 + 0.8 * np.minimum(np.sqrt(pop / pop.max()), 1.0)

    return HierarchicalEpiInputs(
        geos=geos,
        model_years=model_years,
        div_id=div_id,
        unique_div=unique_div,
        no_vs=no_vs,
        prep_off_prop=prep_off_prop,
        risk=risk,
        p_test=p_test,
        hiv_prevalence=hiv_prevalence,
        dx_obs=dx_obs,
        inc_obs=inc_obs,
        period_labels=period_labels,
        period_idx=period_idx,
        period_targets=period_targets,
        inc_g_idx=inc_g_idx,
        inc_t_idx=inc_t_idx,
        inc_obs_vec=inc_obs_vec,
        risk_baseline=risk_baseline,
        risk_ratio=risk_ratio,
        log_risk_ratio=log_risk_ratio,
        exposure=exposure,
        prevalence0=prevalence0,
        state_freedom=state_freedom,
        bundle_cdc=stock_array(stocks, geos, BUNDLE_CDC_NAMES, model_years),
    )


# =============================================================================
# Model: priors, equations, likelihoods and sampling
# =============================================================================

def sample_settings(config: HierarchicalEpiFitConfig) -> dict:
    if config.run_mode == "quick":
        return dict(
            draws=config.quick_draws,
            tune=config.quick_tune,
            chains=config.quick_chains,
            cores=config.quick_cores,
            target_accept=config.target_accept,
        )
    if config.run_mode == "full":
        return dict(
            draws=config.full_draws,
            tune=config.full_tune,
            chains=config.full_chains,
            cores=config.full_cores,
            target_accept=config.target_accept,
        )
    raise ValueError("run_mode must be 'quick' or 'full'")


def fit_hierarchical_epi_model(config: HierarchicalEpiFitConfig) -> Path:
    """Run the PyMC fit; write the posterior, period targets, fit summary and bundle.

    PyMC is imported lazily so ordinary pipeline imports do not initialize PyTensor.
    """

    import os
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

    import pymc as pm
    import pytensor.tensor as pt

    inputs = build_hierarchical_epi_inputs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    G = len(inputs.geos)
    T_model = len(inputs.model_years)
    n_div = len(inputs.unique_div)

    init_vals = {
        "mu_beta_div": np.full(n_div, -3.2),
        "mu_alpha_logit": logit(0.06 / 0.30, 1e-9),
        "sigma_beta": 0.5,
        "z_beta": np.zeros(G),
        "phi_inc": 23.0,
        "mu_log_kappa_div": np.full(n_div, np.log(0.5)),
        "sigma_log_kappa": 0.25,
        "z_kappa": np.zeros(G),
        "mu_logit_rho": logit(0.18, 1e-9),
        "z_rho": np.zeros(G),
        "mu_log_post": np.log(1.4),
        "sigma_log_post": 0.10,
        "z_post": np.zeros(G),
    }

    coords = {
        "state": inputs.geos,
        "year": inputs.model_years,
        "division": [str(d) for d in inputs.unique_div],
        "period": inputs.period_labels,
    }

    with pm.Model(coords=coords):
        mu_beta_div = pm.Normal("mu_beta_div", mu=-3.2, sigma=1.0, dims="division")
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)
        z_beta = pm.Normal("z_beta", 0, 1, dims="state")
        state_freedom = pt.as_tensor_variable(inputs.state_freedom)
        log_beta_inc = pm.Deterministic(
            "log_beta_inc",
            mu_beta_div[inputs.div_id] + sigma_beta * state_freedom * z_beta,
            dims="state",
        )
        beta_inc = pm.Deterministic("beta_inc", pt.exp(log_beta_inc), dims="state")

        # One bounded risk-behavior elasticity shared by all states. State-specific
        # alpha variation was not identifiable in the all-state pilot.
        alpha_upper = 0.30
        alpha_center_scaled = 0.06 / alpha_upper
        mu_alpha_logit = pm.Normal(
            "mu_alpha_logit", mu=logit(alpha_center_scaled, 1e-9), sigma=0.35
        )
        alpha_global = pm.Deterministic(
            "alpha_global", alpha_upper * pm.math.sigmoid(mu_alpha_logit)
        )
        alpha = pm.Deterministic(
            "alpha", pt.ones(G) * alpha_global, dims="state"
        )

        mu_log_kappa_div = pm.Normal("mu_log_kappa_div", mu=np.log(0.5), sigma=0.5, dims="division")
        sigma_log_kappa = pm.HalfNormal("sigma_log_kappa", 0.35)
        z_kappa = pm.Normal("z_kappa", 0, 1, dims="state")
        log_kappa = pm.Deterministic(
            "log_kappa",
            mu_log_kappa_div[inputs.div_id] + sigma_log_kappa * z_kappa,
            dims="state",
        )
        kappa = pm.Deterministic("kappa", pt.exp(log_kappa), dims="state")

        mu_logit_rho = pm.Normal("mu_logit_rho", mu=logit(0.18, 1e-9), sigma=0.35)
        z_rho = pm.Normal("z_rho", 0, 1, dims="state")
        # Fix the state-deviation scale because rho, kappa, and U0 were not
        # separately identifiable with a free hierarchy scale.
        rho_logit_sd = 0.15
        rho = pm.Deterministic(
            "rho", pm.math.sigmoid(mu_logit_rho + rho_logit_sd * z_rho), dims="state"
        )
        U0 = pm.Deterministic("U0", rho * inputs.prevalence0, dims="state")

        mu_log_post = pm.Normal("mu_log_post", mu=np.log(1.4), sigma=0.30)
        sigma_log_post = pm.HalfNormal("sigma_log_post", 0.20)
        z_post = pm.Normal("z_post", 0, 1, dims="state")
        post_multiplier = pm.Deterministic("post_multiplier", pt.exp(mu_log_post + sigma_log_post * z_post), dims="state")

        log_inc = (
            log_beta_inc[:, None]
            + np.log(np.maximum(inputs.exposure, 1e-6))
            + alpha[:, None] * inputs.log_risk_ratio
        )
        Inc = pm.Deterministic("Inc", pt.exp(pt.clip(log_inc, -20, 14)), dims=("state", "year"))

        post_indicator = (inputs.model_years >= POST_YEAR).astype(float)
        diagnosis_multiplier = 1.0 + post_indicator[None, :] * (post_multiplier[:, None] - 1.0)
        delta = pm.Deterministic(
            "diagnosis_probability",
            1.0 - pt.exp(-kappa[:, None] * inputs.p_test * diagnosis_multiplier),
            dims=("state", "year"),
        )

        U_list = [U0]
        Dx_list = [pt.minimum(pt.maximum(delta[:, 0] * U0, 0.0), U0)]
        for ti in range(1, T_model):
            U_next = pt.maximum(U_list[ti - 1] + Inc[:, ti - 1] - Dx_list[ti - 1], 0.0)
            Dx_next = pt.minimum(pt.maximum(delta[:, ti] * U_next, 0.0), U_next)
            U_list.append(U_next)
            Dx_list.append(Dx_next)
        pm.Deterministic("U", pt.stack(U_list, axis=1), dims=("state", "year"))
        Dx = pm.Deterministic("Dx", pt.stack(Dx_list, axis=1), dims=("state", "year"))

        dx_period_mu = pm.Deterministic(
            "dx_period_mu", pt.stack([pt.mean(Dx[:, idx], axis=1) for idx in inputs.period_idx], axis=1),
            dims=("state", "period"),
        )
        pm.Deterministic(
            "inc_period_mu", pt.stack([pt.mean(Inc[:, idx], axis=1) for idx in inputs.period_idx], axis=1),
            dims=("state", "period"),
        )

        # The freely estimated diagnosis-period error collapsed toward zero in
        # pilot fits. The production model uses a fixed 10% log-scale tolerance.
        sigma_dx_period = 0.10
        phi_inc = pm.HalfNormal("phi_inc", 15.0)
        dx_period_obs = inputs.period_targets.pivot(index="state", columns="period", values="dx_obs")
        pm.LogNormal("dx_period_lik", mu=pt.log(dx_period_mu), sigma=sigma_dx_period,
                     observed=dx_period_obs.loc[inputs.geos, inputs.period_labels].to_numpy(),
                     dims=("state", "period"))
        pm.NegativeBinomial("inc_lik", mu=Inc[inputs.inc_g_idx, inputs.inc_t_idx], alpha=phi_inc, observed=inputs.inc_obs_vec)

        start = time.time()
        idata = pm.sample(
            **sample_settings(config),
            initvals=init_vals,
            random_seed=config.random_seed,
            init="jitter+adapt_diag",
            idata_kwargs={"log_likelihood": True},
        )
        elapsed = time.time() - start

    idata.attrs["geos"] = ",".join(inputs.geos)
    idata.attrs["years"] = ",".join(map(str, inputs.model_years.tolist()))
    idata.attrs["diagnosis_periods"] = ",".join(inputs.period_labels)
    idata.attrs["source"] = "pipeline.epi_fit"
    idata.attrs["chains"] = int(sample_settings(config)["chains"])
    idata.attrs["retained_draws_per_chain"] = int(sample_settings(config)["draws"])
    idata.attrs["tuning_steps_per_chain"] = int(sample_settings(config)["tune"])
    idata.attrs["sampling_minutes"] = elapsed / 60

    output_nc = config.output_dir / "posterior.nc"
    idata.to_netcdf(output_nc)
    inputs.period_targets.to_csv(config.output_dir / "period_targets.csv", index=False)
    summarize_hierarchical_epi_fit(idata, inputs).to_csv(config.output_dir / "period_fit_summary.csv", index=False)
    write_epi_bundle(idata.posterior, inputs, config.output_dir / "epi_bundle.npz")
    print(f"Sampling completed in {elapsed / 60:.2f} minutes")
    print(f"Saved posterior and bundle in {config.output_dir}")
    return output_nc


# =============================================================================
# Bundle for the scenarios
# =============================================================================

def write_epi_bundle(posterior, inputs: HierarchicalEpiInputs, output_path: Path) -> None:
    """Posterior draws and CDC stocks in the bundle layout read by the scenarios.

    Parameter draws are flattened chain by chain (draws x states). CDC stocks cover
    the fit window (cdc_raw_native; PrEP stocks carried forward past their last
    year) and are held at its last year through BUNDLE_END_YEAR (cdc_raw). risk0 is the risk baseline used in the fit;
    cdc_kappa_prep is 1, kept for the dashboard's schema (PrEP prevents infection in proportion to coverage;
    the projections do not read it).
    """
    geos = inputs.geos
    if [str(s) for s in posterior["state"].values] != geos:
        raise ValueError("Posterior states do not match the fit inputs")
    native_years = np.asarray(inputs.model_years, dtype=int)
    model_years = np.arange(FIRST_YEAR, BUNDLE_END_YEAR + 1)
    native = inputs.bundle_cdc.copy()
    # PrEP stocks can end before the surveillance years; carry their last year forward.
    for name in ("PrEP Eligible", "PrEP"):
        k = BUNDLE_CDC_NAMES.index(name)
        native[:, k] = pd.DataFrame(native[:, k]).ffill(axis=1).to_numpy()
    held = np.repeat(native[:, :, -1:], len(model_years) - len(native_years), axis=2)
    arrays = {
        "schema_version": np.array([SCHEMA_VERSION]),
        "geo_ids": np.array(geos),
        "model_years": model_years,
        "cdc_native_years": native_years,
        "cdc_raw_names": np.array(BUNDLE_CDC_NAMES),
        "cdc_raw": np.concatenate([native, held], axis=2),
        "cdc_raw_native": native,
        "cdc_kappa_prep": np.ones(len(geos)),
        "cdc_risk0": inputs.risk_baseline,
    }
    for name, key in POSTERIOR_TO_BUNDLE.items():
        arrays[key] = posterior[name].stack(sample=("chain", "draw")).transpose("sample", "state").values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


# =============================================================================
# Fit summaries
# =============================================================================

def _posterior_q(posterior, name: str) -> np.ndarray:
    arr = posterior[name].stack(sample=("chain", "draw")).transpose("state", "period", "sample").values
    return np.quantile(arr, [0.05, 0.50, 0.95], axis=-1)


def summarize_hierarchical_epi_fit(idata, inputs: HierarchicalEpiInputs) -> pd.DataFrame:
    """Period-average posterior fit against the observed targets, for audit tables."""

    targets = inputs.period_targets.set_index(["state", "period"])
    rows = []
    for outcome, obs_col, mu_name in [("diagnoses", "dx_obs", "dx_period_mu"), ("incidence", "inc_obs", "inc_period_mu")]:
        q = _posterior_q(idata.posterior, mu_name)
        for gi, state in enumerate(inputs.geos):
            for pi, period in enumerate(inputs.period_labels):
                obs = targets.loc[(state, period), obs_col]
                rows.append({
                    "state": state,
                    "outcome": outcome,
                    "period": period,
                    "observed": obs,
                    "posterior_p05": q[0, gi, pi],
                    "posterior_median": q[1, gi, pi],
                    "posterior_p95": q[2, gi, pi],
                    "relative_error_median": q[1, gi, pi] / obs - 1.0,
                    "covered_90pct": q[0, gi, pi] <= obs <= q[2, gi, pi],
                })
    return pd.DataFrame(rows)
