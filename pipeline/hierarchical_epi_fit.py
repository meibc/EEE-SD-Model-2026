from __future__ import annotations

"""Hierarchical Bayesian calibration for CDC incidence/diagnosis parameters.

This module ports the core logic from
`sem_sd_1225_refactor/notebooks/transition_bayes_demo_v6.ipynb` into a reusable
pipeline stage. It intentionally does not run during scenario generation or
standardized dashboard export; callers must invoke Stage 04 explicitly.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd

from input.standardized_runtime import StandardizedBundle
from models.shared.alignment import align_to_years


STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

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
    "VT": 643077, "WY": 576851, "DC": 689545, "USA": 331449281,
}


@dataclass(frozen=True)
class HierarchicalEpiFitConfig:
    """Configuration for the long-running hierarchical epi calibration."""

    standardized_input_path: Path = Path("/Users/meibinchen/Documents/GitHub/EEE-Model-Dashboard-Integration/standardized_input_v7.npz")
    data_path: Path = Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx")
    output_dir: Path = Path("output/hierarchical_epi_fit_v7d")
    output_nc_path: Path | None = None
    model_scope: str = "all_states"
    run_mode: str = "quick"
    fit_diagnosis: bool = True
    states: list[str] = field(default_factory=lambda: ["CA", "TX", "NY", "MA", "MS"])
    pre_years: tuple[int, ...] = (2017, 2018, 2019)
    post_years: tuple[int, ...] = (2021, 2022)
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
    pre_idx: np.ndarray
    post_idx: np.ndarray
    period_targets: pd.DataFrame
    inc_g_idx: np.ndarray
    inc_t_idx: np.ndarray
    inc_obs_vec: np.ndarray
    risk_ratio: np.ndarray
    log_risk_ratio: np.ndarray
    exposure: np.ndarray
    prevalence0: np.ndarray
    state_freedom: np.ndarray


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def _as_str_list(arr) -> list[str]:
    return [str(x) for x in list(arr)]


def _load_division_ids(data_path: Path, geos: list[str]) -> tuple[np.ndarray, list[int]]:
    div_map: dict[str, int] = {}
    if data_path.exists():
        states_df = pd.read_excel(data_path, sheet_name="states")
        div_map = dict(zip(states_df["state_calc"].astype(str), states_df["Census_div"].astype(int)))
    raw_div = np.array([div_map.get(g, 0) for g in geos], dtype=int)
    unique_div = sorted(np.unique(raw_div).tolist())
    div_recode = {d: i for i, d in enumerate(unique_div)}
    div_id = np.array([div_recode[d] for d in raw_div], dtype=int)
    return div_id, unique_div


def _cdc_raw_index(bundle: StandardizedBundle, name: str) -> int:
    names = _as_str_list(bundle.cdc_raw_names)
    if name not in names:
        raise ValueError(f"CDC raw variable missing from standardized input: {name}")
    return names.index(name)


def build_hierarchical_epi_inputs(config: HierarchicalEpiFitConfig) -> HierarchicalEpiInputs:
    """Build PyMC-ready arrays from the standardized v7 input artifact.

    This replaces the v6 notebook dependency on `EEE_SD_v4.transition.mc.npz`.
    The current implementation uses observed SEM trajectories for the calibration
    period. Scenario-specific epidemiologic uncertainty is handled downstream by
    Stage 03 using the fitted posterior parameters.
    """

    bundle = StandardizedBundle(config.standardized_input_path)
    geo_ids = _as_str_list(bundle.geo_ids)
    if config.model_scope == "all_states":
        geos = [g for g in STATE_ABBRS if g in geo_ids]
    elif config.model_scope == "selected_states":
        geos = [g for g in config.states if g in geo_ids]
    else:
        raise ValueError("model_scope must be 'all_states' or 'selected_states'")
    if not geos:
        raise ValueError("No geographies selected for hierarchical epi fit")

    years_all = np.asarray(bundle.model_years, dtype=int)
    model_year_mask = years_all <= 2022
    model_years = years_all[model_year_mask]
    pre_idx = np.array([np.where(model_years == y)[0][0] for y in config.pre_years], dtype=int)
    post_idx = np.array([np.where(model_years == y)[0][0] for y in config.post_years], dtype=int)

    sem_names = _as_str_list(bundle.sem_v_names)
    prep_idx = sem_names.index("prep_used")
    risk_idx = sem_names.index("risk_behavior")
    test_idx = sem_names.index("hivtest12")
    sem_obs = np.asarray(bundle.sem_obs, dtype=float)

    geo_index = np.array([geo_ids.index(g) for g in geos], dtype=int)
    sem_source_years = np.asarray(bundle.sem_obs_years, dtype=int)
    prep_on_prop = np.clip(
        align_to_years(sem_source_years, sem_obs[geo_index, prep_idx, :], model_years),
        1e-6,
        1 - 1e-6,
    )
    prep_off_prop = np.clip(1.0 - prep_on_prop, 1e-6, 1.0)
    risk = np.clip(
        align_to_years(sem_source_years, sem_obs[geo_index, risk_idx, :], model_years),
        1e-6,
        1 - 1e-6,
    )
    p_test = np.clip(
        align_to_years(sem_source_years, sem_obs[geo_index, test_idx, :], model_years),
        1e-6,
        1 - 1e-6,
    )

    raw = np.asarray(bundle.cdc_raw, dtype=float)
    cdc_year_index = np.array([list(years_all).index(int(y)) for y in model_years], dtype=int)
    prev_i = _cdc_raw_index(bundle, "Estimated HIV prevalence (MSM)")
    vs_i = _cdc_raw_index(bundle, "HIV viral suppression")
    dx_i = _cdc_raw_index(bundle, "HIV diagnoses")
    inc_i = _cdc_raw_index(bundle, "Estimated HIV incidence (MSM)")

    hiv_prevalence = raw[geo_index[:, None], prev_i, cdc_year_index]
    viral_suppression = raw[geo_index[:, None], vs_i, cdc_year_index]
    no_vs = np.clip(hiv_prevalence - viral_suppression, 1e-6, None)
    dx_obs = raw[geo_index[:, None], dx_i, cdc_year_index]
    inc_obs = raw[geo_index[:, None], inc_i, cdc_year_index]
    dx_obs = np.where(dx_obs > 0, dx_obs, np.nan)
    inc_obs = np.where(inc_obs > 0, inc_obs, np.nan)

    for name, arr in {
        "no_vs": no_vs,
        "prep_off_prop": prep_off_prop,
        "risk": risk,
        "p_test": p_test,
        "hiv_prevalence": hiv_prevalence,
    }.items():
        if not np.isfinite(arr).all():
            bad = np.argwhere(~np.isfinite(arr))[:10]
            raise ValueError(f"{name} contains non-finite values; first locations={bad.tolist()}")

    dx_pre_obs = np.nanmean(dx_obs[:, pre_idx], axis=1)
    dx_post_obs = np.nanmean(dx_obs[:, post_idx], axis=1)
    inc_pre_obs = np.nanmean(inc_obs[:, pre_idx], axis=1)
    inc_post_obs = np.nanmean(inc_obs[:, post_idx], axis=1)
    period_targets = pd.DataFrame({
        "state": geos,
        "dx_pre_obs": dx_pre_obs,
        "dx_post_obs": dx_post_obs,
        "inc_pre_obs": inc_pre_obs,
        "inc_post_obs": inc_post_obs,
        "pre_years": ",".join(map(str, config.pre_years)),
        "post_years": ",".join(map(str, config.post_years)),
    })
    # Diagnosis period means enter the likelihood and must be complete.
    # Incidence period means are audit summaries only; states without CDC
    # incidence estimates may remain missing because the annual likelihood
    # already selects the available state-year observations below.
    if not np.isfinite(period_targets[["dx_pre_obs", "dx_post_obs"]].to_numpy()).all():
        raise ValueError("One or more diagnosis period targets are missing; inspect period_targets")

    inc_annual_mask = np.isfinite(inc_obs) & (model_years != 2020)[None, :]
    inc_g_idx, inc_t_idx = np.where(inc_annual_mask)
    inc_obs_vec = inc_obs[inc_g_idx, inc_t_idx].astype("int64")

    risk_baseline = np.maximum(risk[:, 0], 0.05)
    risk_ratio = np.clip(risk / risk_baseline[:, None], 0.5, 3.0)
    log_risk_ratio = np.log(risk_ratio)
    exposure = no_vs * prep_off_prop
    prevalence0 = hiv_prevalence[:, 0]

    div_id, unique_div = _load_division_ids(config.data_path, geos)
    pop = np.array([STATE_POPS.get(g, 1e6) for g in geos], dtype=float)
    pop_weight = np.minimum(np.sqrt(pop / pop.max()), 1.0)
    state_freedom = 0.2 + 0.8 * pop_weight

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
        pre_idx=pre_idx,
        post_idx=post_idx,
        period_targets=period_targets,
        inc_g_idx=inc_g_idx,
        inc_t_idx=inc_t_idx,
        inc_obs_vec=inc_obs_vec,
        risk_ratio=risk_ratio,
        log_risk_ratio=log_risk_ratio,
        exposure=exposure,
        prevalence0=prevalence0,
        state_freedom=state_freedom,
    )


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


def fit_hierarchical_epi_model(config: HierarchicalEpiFitConfig):
    """Run the long PyMC hierarchical fit and write posterior outputs.

    This function intentionally imports PyMC lazily so ordinary pipeline imports
    do not initialize PyTensor/PyMC. Do not call this from dashboard export.
    """

    import os
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    inputs = build_hierarchical_epi_inputs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    G = len(inputs.geos)
    T_model = len(inputs.model_years)
    n_div = len(inputs.unique_div)

    init_vals = {
        "mu_beta_div": np.full(n_div, -3.2),
        "mu_alpha_logit": _logit(0.06 / 0.30),
        "sigma_beta": 0.5,
        "z_beta": np.zeros(G),
        "phi_inc": 23.0,
        "mu_log_kappa_div": np.full(n_div, np.log(0.5)),
        "sigma_log_kappa": 0.25,
        "z_kappa": np.zeros(G),
        "mu_logit_rho": _logit(0.18),
        "z_rho": np.zeros(G),
        "mu_log_post": np.log(1.4),
        "sigma_log_post": 0.10,
        "z_post": np.zeros(G),
    }

    coords = {
        "state": inputs.geos,
        "year": inputs.model_years,
        "division": [str(d) for d in inputs.unique_div],
        "pre_year": list(config.pre_years),
        "post_year": list(config.post_years),
    }

    with pm.Model(coords=coords) as model:
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

        # Production v7d uses one bounded risk-behavior elasticity shared by
        # all states. State-specific alpha variation was not identifiable in
        # the all-state pilot and is therefore not retained in the final model.
        alpha_upper = 0.30
        alpha_center_scaled = 0.06 / alpha_upper
        mu_alpha_logit = pm.Normal(
            "mu_alpha_logit", mu=_logit(alpha_center_scaled), sigma=0.35
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

        mu_logit_rho = pm.Normal("mu_logit_rho", mu=_logit(0.18), sigma=0.35)
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

        post_indicator = (inputs.model_years >= 2021).astype(float)
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
        U = pm.Deterministic("U", pt.stack(U_list, axis=1), dims=("state", "year"))
        Dx = pm.Deterministic("Dx", pt.stack(Dx_list, axis=1), dims=("state", "year"))

        dx_pre_mu = pm.Deterministic("dx_pre_mu", pt.mean(Dx[:, inputs.pre_idx], axis=1), dims="state")
        dx_post_mu = pm.Deterministic("dx_post_mu", pt.mean(Dx[:, inputs.post_idx], axis=1), dims="state")
        inc_pre_mu = pm.Deterministic("inc_pre_mu", pt.mean(Inc[:, inputs.pre_idx], axis=1), dims="state")
        inc_post_mu = pm.Deterministic("inc_post_mu", pt.mean(Inc[:, inputs.post_idx], axis=1), dims="state")

        # The freely estimated diagnosis-period error collapsed toward zero in
        # pilot fits. The production model uses a fixed 10% log-scale tolerance.
        sigma_dx_period = 0.10
        phi_inc = pm.HalfNormal("phi_inc", 15.0)
        if config.fit_diagnosis:
            pm.LogNormal("dx_pre_lik", mu=pt.log(dx_pre_mu), sigma=sigma_dx_period, observed=inputs.period_targets["dx_pre_obs"].to_numpy(), dims="state")
            pm.LogNormal("dx_post_lik", mu=pt.log(dx_post_mu), sigma=sigma_dx_period, observed=inputs.period_targets["dx_post_obs"].to_numpy(), dims="state")
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
    idata.attrs["source"] = "pipeline.hierarchical_epi_fit; production v7d specification"
    idata.attrs["chains"] = int(sample_settings(config)["chains"])
    idata.attrs["retained_draws_per_chain"] = int(sample_settings(config)["draws"])
    idata.attrs["tuning_steps_per_chain"] = int(sample_settings(config)["tune"])
    idata.attrs["texas_2020_testing_override"] = "none"

    output_nc = config.output_nc_path
    if output_nc is None:
        output_nc = config.output_dir / f"hierarchical_epi_fit_v7d_{config.model_scope}_{config.run_mode}.nc"
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(output_nc)

    inputs.period_targets.to_csv(config.output_dir / f"hierarchical_epi_fit_v7d_{config.model_scope}_period_targets.csv", index=False)
    summarize_hierarchical_epi_fit(idata, inputs).to_csv(
        config.output_dir / f"hierarchical_epi_fit_v7d_{config.model_scope}_{config.run_mode}_period_fit_summary.csv",
        index=False,
    )
    print(f"Sampling completed in {elapsed / 60:.2f} minutes")
    print(f"Saved posterior: {output_nc}")
    return output_nc


def _posterior_q(posterior, name: str) -> np.ndarray:
    arr = posterior[name].stack(sample=("chain", "draw")).transpose("state", "sample").values
    return np.quantile(arr, [0.05, 0.50, 0.95], axis=1).T


def summarize_hierarchical_epi_fit(idata, inputs: HierarchicalEpiInputs) -> pd.DataFrame:
    """Summarize period-average posterior fit for audit tables."""

    posterior = idata.posterior
    rows = []
    targets = inputs.period_targets
    for period, obs_col, mu_name in [
        ("pre", "dx_pre_obs", "dx_pre_mu"),
        ("post", "dx_post_obs", "dx_post_mu"),
        ("pre", "inc_pre_obs", "inc_pre_mu"),
        ("post", "inc_post_obs", "inc_post_mu"),
    ]:
        outcome = "diagnoses" if mu_name.startswith("dx") else "incidence"
        obs_values = targets[obs_col].to_numpy()
        q = _posterior_q(posterior, mu_name)
        for gi, state in enumerate(inputs.geos):
            rows.append({
                "state": state,
                "outcome": outcome,
                "period": period,
                "observed": obs_values[gi],
                "posterior_p05": q[gi, 0],
                "posterior_median": q[gi, 1],
                "posterior_p95": q[gi, 2],
                "relative_error_median": q[gi, 1] / obs_values[gi] - 1.0,
                "covered_90pct": q[gi, 0] <= obs_values[gi] <= q[gi, 2],
            })
    return pd.DataFrame(rows)
