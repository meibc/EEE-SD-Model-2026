from __future__ import annotations

"""Monte Carlo SEM-only uncertainty for national priority populations.

This mirrors the state-level SEM uncertainty structure at the input-draw level:
probability estimates are sampled from Beta distributions using weighted
proportions and effective sample sizes, then each sampled trajectory is refit
with the same centered-logit SEM, structural priors, drift settings, and
scenario machinery used by the deterministic priority-population workflow.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd

from config.interventions import SCENARIO_CODEBOOK
from config.optimization import OptimConfig
from config.sem import SEMConfig
from models.sbm.estimation.jacobian import JacobianEstimator
from models.sbm.estimation.transforms import FeatureTransformer
from models.sbm.prediction.predictor import Predictor
from models.shared.intervention import build_relationship_interventions, build_state_interventions
from pipeline.priority_population_sem import (
    PriorityPopulationInputs,
    PriorityPopulationSEMConfig,
    SEM_V_NAMES,
    _drift_arrays,
    _safe_name,
    infer_population_id,
    load_priority_population_inputs,
)

YEAR_GROUP_TO_YEAR = {
    "2017-2018": 2018,
    "2019-2020": 2020,
    "2021-2022": 2022,
    "2023-2024": 2024,
}


@dataclass(frozen=True)
class PriorityPopulationMCConfig:
    summary_path: Path
    input_base_dir: Path
    output_dir: Path = Path("output/priority_population_sem_mc")
    variable_config_path: Path = Path("config/sem_variable_config_v3.csv")
    n_draws: int = 500
    seed: int = 123
    target_end_year: int = 2036
    intervention_duration_steps: int = 3
    drift_mode: str = "zero"
    drift_tau: float = 0.20
    maxiter: int = 500
    scenario_codes: list[str] | None = None
    eps: float = 1e-6


def _required_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def discover_input_dirs(input_base_dir: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for input_dir in sorted(Path(input_base_dir).glob("*_sem_inputs")):
        pop = infer_population_id(input_dir)
        dirs[pop] = input_dir
    if not dirs:
        raise FileNotFoundError(f"No *_sem_inputs folders found in {input_base_dir}")
    return dirs


def load_summary_table(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    if "weighted_prop" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "weighted_prop"})
    _required_columns(
        df,
        {"population_id", "population_label", "variable", "weighted_prop", "n_eff"},
        str(summary_path),
    )
    df = df.copy()
    if "Year" not in df.columns:
        _required_columns(df, {"year_group"}, str(summary_path))
        df["Year"] = df["year_group"].map(YEAR_GROUP_TO_YEAR)
        if df["Year"].isna().any():
            bad = sorted(df.loc[df["Year"].isna(), "year_group"].astype(str).unique())
            raise ValueError(f"Unsupported year_group values: {bad}")
    df["Year"] = df["Year"].astype(int)
    df["weighted_prop"] = pd.to_numeric(df["weighted_prop"], errors="coerce")
    df["n_eff"] = pd.to_numeric(df["n_eff"], errors="coerce")
    if df[["weighted_prop", "n_eff"]].isna().any().any():
        raise ValueError("weighted_prop and n_eff must be numeric and non-missing")
    return df


def _sample_Y_for_population(
    summary: pd.DataFrame,
    population_id: str,
    years: np.ndarray,
    v_names: list[str],
    n_draws: int,
    rng: np.random.Generator,
    eps: float,
) -> np.ndarray:
    d = summary[summary["population_id"].astype(str).eq(str(population_id))].copy()
    expected = len(years) * len(v_names)
    if len(d) != expected:
        raise ValueError(f"{population_id}: expected {expected} summary rows; found {len(d)}")

    Y = np.full((n_draws, len(v_names), len(years)), np.nan, dtype=np.float32)
    for _, row in d.iterrows():
        if row["variable"] not in v_names:
            continue
        if int(row["Year"]) not in set(map(int, years)):
            continue
        vi = v_names.index(str(row["variable"]))
        ti = list(map(int, years)).index(int(row["Year"]))
        p = float(np.clip(row["weighted_prop"], eps, 1 - eps))
        n_eff = max(float(row["n_eff"]), 1.0)
        alpha = p * n_eff + 1.0
        beta = (1.0 - p) * n_eff + 1.0
        Y[:, vi, ti] = rng.beta(alpha, beta, size=n_draws).astype(np.float32)
    if not np.isfinite(Y).all():
        bad = np.argwhere(~np.isfinite(Y))[:10]
        raise ValueError(f"{population_id}: non-finite sampled Y entries: {bad}")
    return np.clip(Y, eps, 1 - eps)


def generate_priority_population_input_draws(config: PriorityPopulationMCConfig) -> Path:
    summary = load_summary_table(config.summary_path)
    input_dirs = discover_input_dirs(config.input_base_dir)
    rng = np.random.default_rng(config.seed)

    inputs_by_pop: dict[str, PriorityPopulationInputs] = {
        pop: load_priority_population_inputs(path) for pop, path in input_dirs.items()
    }
    population_ids = list(inputs_by_pop)
    v_names = list(SEM_V_NAMES)
    years = np.asarray(next(iter(inputs_by_pop.values())).years, dtype=int)
    for pop, inputs in inputs_by_pop.items():
        if list(inputs.v_names) != v_names:
            raise ValueError(f"{pop}: variable order mismatch")
        if list(np.asarray(inputs.years, dtype=int)) != list(years):
            raise ValueError(f"{pop}: year order mismatch")

    draws = np.full((config.n_draws, len(population_ids), len(v_names), len(years)), np.nan, dtype=np.float32)
    labels = []
    for pi, pop in enumerate(population_ids):
        inputs = inputs_by_pop[pop]
        draws[:, pi] = _sample_Y_for_population(summary, pop, years, v_names, config.n_draws, rng, config.eps)
        labels.append(inputs.population_label)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / f"priority_population_sem_input_draws_n{config.n_draws}_seed{config.seed}.npz"
    np.savez_compressed(
        out_path,
        Y_samples=draws,
        population_ids=np.asarray(population_ids),
        population_labels=np.asarray(labels),
        v_names=np.asarray(v_names),
        years=years,
        seed=np.asarray(config.seed),
        S=np.asarray(config.n_draws),
        summary_path=str(config.summary_path),
        input_base_dir=str(config.input_base_dir),
        artifact_type=np.asarray("priority_population_sem_input_draws"),
    )
    audit = pd.DataFrame({
        "metric": ["n_draws", "n_populations", "n_variables", "n_years", "min_value", "max_value", "finite"],
        "value": [
            config.n_draws,
            len(population_ids),
            len(v_names),
            len(years),
            float(np.nanmin(draws)),
            float(np.nanmax(draws)),
            bool(np.isfinite(draws).all()),
        ],
    })
    audit.to_csv(out_path.with_suffix(".audit.csv"), index=False)
    return out_path


def _fit_one_sample(config: PriorityPopulationMCConfig, inputs: PriorityPopulationInputs, Y: np.ndarray):
    base = SEMConfig(
        structural_prior_path=inputs.priors_path,
        variable_config_path=config.variable_config_path,
    )
    opt = OptimConfig()
    opt.maxiter = int(config.maxiter)
    transformer = FeatureTransformer(reference_probs=base.reference_probs())
    predictor = Predictor(reference_probs=base.reference_probs())
    estimator = JacobianEstimator(opt)

    Xbar, SigmaX = transformer.transform(Y, inputs.SigmaY)
    SigmaX = np.asarray(SigmaX, dtype=float) + np.eye(len(inputs.v_names)) * 1e-8
    sign, prior_mean, prior_tau = base.structural_prior_arrays()
    drift_enabled, drift_prior_mean, drift_prior_tau = _drift_arrays(base, config.drift_mode, config.drift_tau)
    M = np.eye(len(inputs.v_names), dtype=int) + (sign != 0).astype(int)
    u = np.zeros(len(inputs.v_names), dtype=float)

    J_seq = estimator.fit(
        Xbar=Xbar,
        SigmaX=SigmaX,
        M=M,
        sign_mask=sign,
        u=u,
        J_ref=None,
        wJ=0.0,
        structural_prior_mean=prior_mean,
        structural_prior_tau=prior_tau,
        drift_enabled=drift_enabled,
        drift_prior_mean=drift_prior_mean,
        drift_prior_tau=drift_prior_tau,
    )
    c_seq = estimator.last_drift_sequence if estimator.last_drift_sequence is not None else np.zeros_like(Xbar)
    J_final = np.asarray(J_seq[:, :, -1], dtype=float)
    c_final = np.asarray(c_seq[:, -1], dtype=float)
    diagnostics = estimator.last_fit_history[-1]["diagnostics"] if estimator.last_fit_history else {}
    return transformer, predictor, Xbar, J_final, c_final, diagnostics


def _quantiles(x: np.ndarray) -> dict[str, float]:
    return {
        "p025": float(np.nanquantile(x, 0.025)),
        "p50": float(np.nanquantile(x, 0.50)),
        "p975": float(np.nanquantile(x, 0.975)),
    }


def run_priority_population_mc(config: PriorityPopulationMCConfig) -> dict[str, Path]:
    t0 = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    input_draw_path = generate_priority_population_input_draws(config)
    with np.load(input_draw_path, allow_pickle=True) as raw:
        Y_samples = np.asarray(raw["Y_samples"], dtype=float)
        population_ids = [str(x) for x in raw["population_ids"]]
        v_names = [str(x) for x in raw["v_names"]]
        years = np.asarray(raw["years"], dtype=int)

    input_dirs = discover_input_dirs(config.input_base_dir)
    inputs_by_pop = {pop: load_priority_population_inputs(input_dirs[pop]) for pop in population_ids}
    scenario_codes = list(SCENARIO_CODEBOOK.keys()) if config.scenario_codes is None else list(config.scenario_codes)
    invalid = [s for s in scenario_codes if s not in SCENARIO_CODEBOOK]
    if invalid:
        raise ValueError(f"Unknown scenario codes: {invalid}")

    S, P, m, T = Y_samples.shape
    sem_years = np.arange(int(years[-1]), int(config.target_end_year) + 1, 2, dtype=int)
    endpoint_values: dict[tuple[str, str, str], list[float]] = {}
    baseline_values: dict[tuple[str, str, str], list[float]] = {}
    J_rows: list[dict] = []
    drift_rows: list[dict] = []
    fit_rows: list[dict] = []

    for pi, pop in enumerate(population_ids):
        inputs = inputs_by_pop[pop]
        unit = SimpleNamespace(amis_years=years)
        for draw_idx in range(S):
            Y = Y_samples[draw_idx, pi]
            transformer, predictor, Xbar, J_final, c_final, diagnostics = _fit_one_sample(config, inputs, Y)
            success = bool(diagnostics.get("success", False)) if isinstance(diagnostics, dict) else False
            final_loss = diagnostics.get("final_loss", np.nan) if isinstance(diagnostics, dict) else np.nan
            fit_rows.append({
                "population_id": pop,
                "population_label": inputs.population_label,
                "draw": draw_idx,
                "success": success,
                "final_loss": final_loss,
                "drift_norm": float(np.linalg.norm(c_final)),
                "J_abs_mean": float(np.nanmean(np.abs(J_final))),
            })
            for i, target in enumerate(v_names):
                drift_rows.append({
                    "population_id": pop,
                    "population_label": inputs.population_label,
                    "draw": draw_idx,
                    "variable": target,
                    "drift": float(c_final[i]),
                })
                for j, source in enumerate(v_names):
                    if abs(J_final[i, j]) > 0 or i == j:
                        J_rows.append({
                            "population_id": pop,
                            "population_label": inputs.population_label,
                            "draw": draw_idx,
                            "target": target,
                            "source": source,
                            "J": float(J_final[i, j]),
                        })
            X0 = Xbar[:, -1]
            Y_base, _ = predictor.predict_trajectory(J_final, X0, c_final, len(sem_years))
            for scenario_code in scenario_codes:
                spec = SCENARIO_CODEBOOK[scenario_code]
                state_ivs = build_state_interventions(
                    unit, sem_years, v_names, spec.get("state_codes", []), config.intervention_duration_steps
                )
                rel_ivs = build_relationship_interventions(
                    unit, sem_years, v_names, spec.get("relationship_codes", []), config.intervention_duration_steps
                )
                Y_scen, _ = predictor.predict_trajectory(
                    J_final,
                    X0,
                    c_final,
                    len(sem_years),
                    state_interventions=state_ivs,
                    rel_interventions=rel_ivs,
                    baseline_probs=Y_base,
                )
                end_i = len(sem_years) - 1
                for vi, var in enumerate(v_names):
                    key = (pop, scenario_code, var)
                    baseline_values.setdefault(key, []).append(float(Y_base[vi, end_i]))
                    endpoint_values.setdefault(key, []).append(float(Y_scen[vi, end_i]))
        print(f"finished {pop}: {S} draws ({time.time() - t0:.1f}s elapsed)")

    endpoint_rows: list[dict] = []
    for key, scen_list in endpoint_values.items():
        pop, scenario_code, var = key
        inputs = inputs_by_pop[pop]
        b = np.asarray(baseline_values[key], dtype=float)
        s = np.asarray(scen_list, dtype=float)
        d = s - b
        pct = np.where(np.abs(b) > 1e-12, 100.0 * d / b, np.nan)
        brow = _quantiles(b)
        srow = _quantiles(s)
        drow = _quantiles(d)
        prow = _quantiles(pct)
        endpoint_rows.append({
            "population_id": pop,
            "population_label": inputs.population_label,
            "scenario": scenario_code,
            "scenario_label": SCENARIO_CODEBOOK[scenario_code].get("label", scenario_code),
            "year": int(config.target_end_year),
            "variable": var,
            **{f"baseline_{k}": v for k, v in brow.items()},
            **{f"scenario_{k}": v for k, v in srow.items()},
            **{f"delta_{k}": v for k, v in drow.items()},
            **{f"pct_change_{k}": v for k, v in prow.items()},
        })

    suffix = f"n{config.n_draws}_seed{config.seed}_drift_{config.drift_mode}"
    if config.drift_mode == "zero":
        suffix += f"_tau{_safe_name(f'{config.drift_tau:.2f}')}"
    endpoint_path = config.output_dir / f"priority_population_mc_scenario_endpoints_{suffix}.csv"
    fit_path = config.output_dir / f"priority_population_mc_fit_audit_{suffix}.csv"
    j_path = config.output_dir / f"priority_population_mc_J_draws_{suffix}.csv"
    drift_path = config.output_dir / f"priority_population_mc_drift_draws_{suffix}.csv"
    summary_path = config.output_dir / f"priority_population_mc_run_summary_{suffix}.csv"
    compact_path = config.output_dir / f"priority_population_mc_scenario_delta_wide_{suffix}.csv"

    endpoint_df = pd.DataFrame(endpoint_rows)
    endpoint_df.to_csv(endpoint_path, index=False)
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    pd.DataFrame(J_rows).to_csv(j_path, index=False)
    pd.DataFrame(drift_rows).to_csv(drift_path, index=False)

    compact = endpoint_df.copy()
    compact["baseline_pct_p50"] = 100.0 * compact["baseline_p50"]
    compact["scenario_pct_p50"] = 100.0 * compact["scenario_p50"]
    compact["delta_pp_p025"] = 100.0 * compact["delta_p025"]
    compact["delta_pp_p50"] = 100.0 * compact["delta_p50"]
    compact["delta_pp_p975"] = 100.0 * compact["delta_p975"]
    compact.to_csv(compact_path, index=False)

    summary = pd.DataFrame({
        "metric": ["n_draws", "n_populations", "n_variables", "n_years", "n_scenarios", "target_end_year", "elapsed_seconds", "success_rate"],
        "value": [
            S,
            P,
            m,
            T,
            len(scenario_codes),
            int(config.target_end_year),
            float(time.time() - t0),
            float(pd.DataFrame(fit_rows)["success"].mean()) if fit_rows else np.nan,
        ],
    })
    summary.to_csv(summary_path, index=False)
    return {
        "input_draws": input_draw_path,
        "endpoint": endpoint_path,
        "compact_endpoint": compact_path,
        "fit_audit": fit_path,
        "J_draws": j_path,
        "drift_draws": drift_path,
        "summary": summary_path,
    }
