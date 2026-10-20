from __future__ import annotations

"""SEM-only fitting and scenario projection for national priority populations.

This module reuses the centered-logit SEM estimator, structural-prior parser,
drift-aware loss, and intervention machinery used by the main SEM pipeline. It
intentionally excludes CDC/hierarchical epidemiologic outcomes.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.interventions import SCENARIO_CODEBOOK
from config.optimization import OptimConfig
from config.sem import SEMConfig
from models.sbm.estimation.jacobian import JacobianEstimator
from models.sbm.estimation.transforms import FeatureTransformer
from models.sbm.prediction.predictor import Predictor
from models.shared.intervention import build_relationship_interventions, build_state_interventions

SEM_V_NAMES = [
    "stigma_ahs",
    "stigma_gss",
    "stigma_family",
    "out_gid",
    "seehcp",
    "risk_behavior",
    "prep_used",
    "hivtest12",
]


@dataclass(frozen=True)
class PriorityPopulationSEMConfig:
    input_dir: Path
    output_dir: Path = Path("output/priority_population_sem")
    variable_config_path: Path = Path("config/sem_variable_config_v3.csv")
    target_end_year: int = 2036
    intervention_duration_steps: int = 3
    drift_mode: str = "zero"  # zero, none, current
    drift_tau: float = 0.20
    maxiter: int = 500
    make_plots: bool = True
    init_j_path: Path | None = None
    init_j_mode: str = "none"  # none, state_median, national


@dataclass(frozen=True)
class PriorityPopulationInputs:
    population_id: str
    population_label: str
    years: np.ndarray
    v_names: list[str]
    Y: np.ndarray
    SigmaY: np.ndarray
    trends: pd.DataFrame
    priors_path: Path
    metadata: pd.DataFrame | None


def infer_population_id(input_dir: Path) -> str:
    trend_files = sorted(Path(input_dir).glob("*_trends.csv"))
    if len(trend_files) != 1:
        raise ValueError(f"Expected exactly one *_trends.csv in {input_dir}; found {len(trend_files)}")
    return trend_files[0].name.removesuffix("_trends.csv")


def load_priority_population_inputs(input_dir: Path, v_names: list[str] | None = None) -> PriorityPopulationInputs:
    input_dir = Path(input_dir)
    pop = infer_population_id(input_dir)
    v_names = list(SEM_V_NAMES if v_names is None else v_names)

    trends_path = input_dir / f"{pop}_trends.csv"
    cov_path = input_dir / f"{pop}_cov_matrix.csv"
    priors_path = input_dir / f"{pop}_structural_priors.csv"
    metadata_path = input_dir / f"{pop}_metadata.csv"

    for path in [trends_path, cov_path, priors_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    trends = pd.read_csv(trends_path)
    required_trend_cols = {"population_id", "population_label", "Year", "variable", "value"}
    missing = required_trend_cols.difference(trends.columns)
    if missing:
        raise ValueError(f"{trends_path} missing columns: {sorted(missing)}")

    years = np.array(sorted(trends["Year"].dropna().astype(int).unique()), dtype=int)
    if years.size < 3:
        raise ValueError("Priority-population SEM needs at least 3 time points")

    label = str(trends["population_label"].dropna().iloc[0])
    Y = np.full((len(v_names), len(years)), np.nan, dtype=float)
    for ti, year in enumerate(years):
        d = trends[trends["Year"].astype(int) == int(year)]
        for vi, var in enumerate(v_names):
            rows = d[d["variable"].astype(str) == var]
            if len(rows) != 1:
                raise ValueError(f"Expected exactly one trend row for year={year}, variable={var}; found {len(rows)}")
            Y[vi, ti] = float(rows["value"].iloc[0])
    if not np.all(np.isfinite(Y)) or np.any((Y < 0) | (Y > 1)):
        raise ValueError("Trend values must be finite probabilities in [0, 1]")

    cov = pd.read_csv(cov_path)
    required_cov_cols = {"var1", *v_names}
    missing = required_cov_cols.difference(cov.columns)
    if missing:
        raise ValueError(f"{cov_path} missing columns: {sorted(missing)}")
    SigmaY = cov.set_index("var1").loc[v_names, v_names].astype(float).values
    max_asym = float(np.nanmax(np.abs(SigmaY - SigmaY.T)))
    if max_asym > 1e-8:
        raise ValueError(f"Covariance matrix is not symmetric; max asymmetry={max_asym}")
    min_eig = float(np.linalg.eigvalsh((SigmaY + SigmaY.T) / 2).min())
    if min_eig < -1e-6:
        raise ValueError(f"Covariance matrix is not positive semidefinite; min eigenvalue={min_eig}")

    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else None
    return PriorityPopulationInputs(pop, label, years, v_names, Y, SigmaY, trends, priors_path, metadata)


def _drift_arrays(
    base: SEMConfig,
    mode: str,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enabled, mean, prior_tau = base.drift_prior_arrays()
    mode = str(mode).lower()
    if mode == "current":
        return enabled, mean, prior_tau
    if mode == "none":
        return np.zeros_like(enabled, dtype=bool), np.zeros_like(mean), np.full_like(prior_tau, np.nan, dtype=float)
    if mode == "zero":
        zero_mean = np.zeros_like(mean, dtype=float)
        zero_tau = np.full_like(prior_tau, np.nan, dtype=float)
        zero_tau[enabled] = float(tau)
        return enabled, zero_mean, zero_tau
    raise ValueError(f"Unknown drift_mode={mode!r}; expected zero, none, or current")


def _safe_name(value: str) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _load_initial_j_matrix(config: PriorityPopulationSEMConfig, v_names: list[str]) -> np.ndarray | None:
    mode = str(config.init_j_mode).lower()
    if mode == "none":
        return None
    if config.init_j_path is None:
        raise ValueError("init_j_path is required when init_j_mode is not 'none'")

    path = Path(config.init_j_path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as raw:
            if "sem_fit_J_last" not in raw.files:
                raise ValueError(f"{path} does not contain sem_fit_J_last")
            J_all = np.asarray(raw["sem_fit_J_last"], dtype=float)
            sem_names = [str(x) for x in raw["sem_v_names"].tolist()] if "sem_v_names" in raw.files else list(SEM_V_NAMES)
            geos = [str(x) for x in raw["geo_ids"].tolist()] if "geo_ids" in raw.files else []

        idx = [sem_names.index(v) for v in v_names]
        J_all = J_all[:, idx, :][:, :, idx]
        if mode == "state_median":
            return np.nanmedian(J_all, axis=0)
        if mode == "national":
            for label in ("USA", "US"):
                if label in geos:
                    return J_all[geos.index(label)]
            return np.nanmedian(J_all, axis=0)
        raise ValueError(f"Unknown init_j_mode={config.init_j_mode!r}; expected none, state_median, or national")

    J = pd.read_csv(path, index_col=0).loc[v_names, v_names].astype(float).values
    return J


def _run_one(config: PriorityPopulationSEMConfig, inputs: PriorityPopulationInputs, run_output_dir: Path) -> dict[str, Path]:
    run_output_dir.mkdir(parents=True, exist_ok=True)

    base = SEMConfig(
        structural_prior_path=inputs.priors_path,
        variable_config_path=config.variable_config_path,
    )
    opt = OptimConfig()
    opt.maxiter = int(config.maxiter)
    transformer = FeatureTransformer(reference_probs=base.reference_probs())
    predictor = Predictor(reference_probs=base.reference_probs())
    estimator = JacobianEstimator(opt)

    Xbar, SigmaX = transformer.transform(inputs.Y, inputs.SigmaY)
    SigmaX = np.asarray(SigmaX, dtype=float) + np.eye(len(inputs.v_names)) * 1e-8
    sign, prior_mean, prior_tau = base.structural_prior_arrays()
    drift_enabled, drift_prior_mean, drift_prior_tau = _drift_arrays(base, config.drift_mode, config.drift_tau)
    M = np.eye(len(inputs.v_names), dtype=int) + (sign != 0).astype(int)
    u = np.zeros(len(inputs.v_names), dtype=float)
    J_init = _load_initial_j_matrix(config, inputs.v_names)

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
        J_init=J_init,
    )
    c_seq = estimator.last_drift_sequence if estimator.last_drift_sequence is not None else np.zeros_like(Xbar)
    J_final = J_seq[:, :, -1]
    c_final = c_seq[:, -1]

    # Observed one-step fit.
    Y_fit = np.full_like(inputs.Y, np.nan)
    Y_fit[:, 0] = inputs.Y[:, 0]
    fit_rows: list[dict] = []
    for ti, year in enumerate(inputs.years):
        if ti > 0:
            xhat = J_seq[:, :, ti] @ Xbar[:, ti - 1] + c_seq[:, ti]
            Y_fit[:, ti] = transformer.inverse_transform(xhat[:, None])[:, 0]
        for vi, var in enumerate(inputs.v_names):
            fit_rows.append({
                "population_id": inputs.population_id,
                "population_label": inputs.population_label,
                "year": int(year),
                "variable": var,
                "observed": float(inputs.Y[vi, ti]),
                "fitted_one_step": float(Y_fit[vi, ti]),
                "error": float(Y_fit[vi, ti] - inputs.Y[vi, ti]),
            })
    fit_df = pd.DataFrame(fit_rows)
    fit_summary = (
        fit_df[fit_df["year"] != int(inputs.years[0])]
        .groupby("variable", as_index=False)
        .agg(
            rmse_pp=("error", lambda x: 100.0 * float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
            mae_pp=("error", lambda x: 100.0 * float(np.nanmean(np.abs(np.asarray(x, dtype=float))))),
            bias_pp=("error", lambda x: 100.0 * float(np.nanmean(np.asarray(x, dtype=float)))),
        )
    )

    J_rows: list[dict] = []
    for ti, year in enumerate(inputs.years):
        for i, target in enumerate(inputs.v_names):
            for j, source in enumerate(inputs.v_names):
                if M[i, j] == 1:
                    J_rows.append({
                        "population_id": inputs.population_id,
                        "year_fit_through": int(year),
                        "target": target,
                        "source": source,
                        "J": float(J_seq[i, j, ti]),
                    })

    # Baseline and final scenario projections.
    sem_years = np.arange(int(inputs.years[-1]), int(config.target_end_year) + 1, 2, dtype=int)
    unit = SimpleNamespace(amis_years=inputs.years)
    X0 = Xbar[:, -1]
    Y_base, _ = predictor.predict_trajectory(J_final, X0, c_final, len(sem_years))

    scenario_specs = {"baseline": {"state_codes": [], "relationship_codes": [], "label": "Baseline"}}
    scenario_specs.update(SCENARIO_CODEBOOK)
    traj_rows: list[dict] = []
    endpoint_rows: list[dict] = []
    for scenario, spec in scenario_specs.items():
        if scenario == "baseline":
            P = Y_base.copy()
        else:
            state_ivs = build_state_interventions(
                unit, sem_years, inputs.v_names, spec.get("state_codes", []), config.intervention_duration_steps
            )
            rel_ivs = build_relationship_interventions(
                unit, sem_years, inputs.v_names, spec.get("relationship_codes", []), config.intervention_duration_steps
            )
            P, _ = predictor.predict_trajectory(
                J_final,
                X0,
                c_final,
                len(sem_years),
                state_interventions=state_ivs,
                rel_interventions=rel_ivs,
                baseline_probs=Y_base,
            )
        for ti, year in enumerate(sem_years):
            for vi, var in enumerate(inputs.v_names):
                traj_rows.append({
                    "population_id": inputs.population_id,
                    "population_label": inputs.population_label,
                    "drift_mode": config.drift_mode,
                    "drift_tau": np.nan if config.drift_mode != "zero" else float(config.drift_tau),
                    "init_j_mode": config.init_j_mode,
                    "scenario": scenario,
                    "scenario_label": spec.get("label", scenario),
                    "year": int(year),
                    "variable": var,
                    "value": float(P[vi, ti]),
                })
        end_i = len(sem_years) - 1
        for vi, var in enumerate(inputs.v_names):
            baseline = float(Y_base[vi, end_i])
            value = float(P[vi, end_i])
            endpoint_rows.append({
                "population_id": inputs.population_id,
                "population_label": inputs.population_label,
                "drift_mode": config.drift_mode,
                "drift_tau": np.nan if config.drift_mode != "zero" else float(config.drift_tau),
                "init_j_mode": config.init_j_mode,
                "scenario": scenario,
                "scenario_label": spec.get("label", scenario),
                "year": int(sem_years[end_i]),
                "variable": var,
                "baseline_value": baseline,
                "scenario_value": value,
                "delta": value - baseline,
                "pct_change": 100.0 * (value - baseline) / baseline if abs(baseline) > 1e-12 else np.nan,
            })

    paths = {
        "observed_vs_fitted": run_output_dir / "observed_vs_fitted.csv",
        "fit_summary": run_output_dir / "fit_summary_by_variable.csv",
        "J_long": run_output_dir / "J_estimates_long.csv",
        "J_final": run_output_dir / "J_final_matrix.csv",
        "drift": run_output_dir / "drift_estimates.csv",
        "trajectory": run_output_dir / "scenario_trajectories.csv",
        "endpoint": run_output_dir / "scenario_endpoints_2036.csv",
        "compact_endpoint": run_output_dir / "scenario_endpoints_2036_compact.csv",
        "run_summary": run_output_dir / "run_summary.csv",
    }
    fit_df.to_csv(paths["observed_vs_fitted"], index=False)
    fit_summary.to_csv(paths["fit_summary"], index=False)
    pd.DataFrame(J_rows).to_csv(paths["J_long"], index=False)
    pd.DataFrame(J_final, index=inputs.v_names, columns=inputs.v_names).to_csv(paths["J_final"])
    pd.DataFrame({
        "variable": inputs.v_names,
        "drift_final": c_final,
        "drift_enabled": drift_enabled,
        "drift_prior_mean": drift_prior_mean,
        "drift_prior_tau": drift_prior_tau,
    }).to_csv(paths["drift"], index=False)
    traj_df = pd.DataFrame(traj_rows)
    endpoint_df = pd.DataFrame(endpoint_rows)
    traj_df.to_csv(paths["trajectory"], index=False)
    endpoint_df.to_csv(paths["endpoint"], index=False)
    compact = endpoint_df[endpoint_df["scenario"] != "baseline"].copy()
    compact["baseline_pct"] = 100.0 * compact["baseline_value"]
    compact["scenario_pct"] = 100.0 * compact["scenario_value"]
    compact["delta_pp"] = 100.0 * compact["delta"]
    compact[[
        "population_id", "population_label", "drift_mode", "drift_tau", "init_j_mode", "scenario", "scenario_label",
        "variable", "baseline_pct", "scenario_pct", "delta_pp", "pct_change"
    ]].to_csv(paths["compact_endpoint"], index=False)

    diagnostics = estimator.last_fit_history[-1]["diagnostics"] if estimator.last_fit_history else {}
    final_components = diagnostics.get("final_components", {}) if isinstance(diagnostics, dict) else {}
    pd.DataFrame({
        "metric": [
            "population_id", "population_label", "drift_mode", "drift_tau", "n_years", "target_end_year",
            "init_j_mode", "init_j_path", "fit_success", "final_loss", "mean_fit_rmse_pp", "mean_fit_mae_pp",
        ],
        "value": [
            inputs.population_id,
            inputs.population_label,
            config.drift_mode,
            np.nan if config.drift_mode != "zero" else float(config.drift_tau),
            len(inputs.years),
            int(config.target_end_year),
            config.init_j_mode,
            "" if config.init_j_path is None else str(config.init_j_path),
            diagnostics.get("success", np.nan) if isinstance(diagnostics, dict) else np.nan,
            diagnostics.get("final_loss", np.nan) if isinstance(diagnostics, dict) else np.nan,
            float(fit_summary["rmse_pp"].mean()),
            float(fit_summary["mae_pp"].mean()),
        ],
    }).to_csv(paths["run_summary"], index=False)

    if config.make_plots:
        _make_plots(inputs, inputs.Y, Y_fit, sem_years, Y_base, traj_df, run_output_dir)

    return paths


def _make_plots(
    inputs: PriorityPopulationInputs,
    Y: np.ndarray,
    Y_fit: np.ndarray,
    sem_years: np.ndarray,
    Y_base: np.ndarray,
    traj_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_vars = ["stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "prep_used", "hivtest12"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    axes = axes.ravel()
    for ax, var in zip(axes, plot_vars):
        vi = inputs.v_names.index(var)
        ax.plot(inputs.years, Y[vi], marker="o", color="black", label="observed")
        ax.plot(inputs.years, Y_fit[vi], marker="x", color="tab:blue", label="one-step fit")
        ax.plot(sem_years, Y_base[vi], ls="--", color="tab:gray", label="baseline projection")
        ax.set_title(var)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{inputs.population_label}: observed, fitted, and baseline SEM projection", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_dir / "observed_fitted_projection.png", dpi=180)
    plt.close()

    focus_scenarios = [
        "s1_reduce_ahs",
        "s4_weaken_ahs_to_prep",
        "s5_weaken_ahs_to_hivtest",
        "s6_strengthen_seehcp_to_lower_ahs",
        "s7_combined_stigma_package",
    ]
    for var, ylabel in [("prep_used", "PrEP use (%)"), ("hivtest12", "HIV testing (%)"), ("stigma_ahs", "AHS (%)")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for scenario in ["baseline", *focus_scenarios]:
            d = traj_df[(traj_df["scenario"] == scenario) & (traj_df["variable"] == var)]
            if not d.empty:
                ax.plot(d["year"], 100.0 * d["value"], marker="o", label=scenario)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{inputs.population_label}: SEM-only scenario projection for {var}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"scenario_projection_{var}.png", dpi=180)
        plt.close()


def run_priority_population_sem(config: PriorityPopulationSEMConfig) -> dict[str, Path]:
    inputs = load_priority_population_inputs(config.input_dir)
    if config.drift_mode == "zero":
        run_name = f"drift_zero_tau{_safe_name(f'{config.drift_tau:.2f}')}"
    else:
        run_name = f"drift_{config.drift_mode}"
    if str(config.init_j_mode).lower() != "none":
        run_name = f"{run_name}_init_{config.init_j_mode}"
    run_output_dir = Path(config.output_dir) / inputs.population_id / run_name
    return _run_one(config, inputs, run_output_dir)


def run_priority_population_drift_sensitivity(
    config: PriorityPopulationSEMConfig,
    zero_taus: list[float] | None = None,
    include_none: bool = True,
    include_current: bool = True,
) -> Path:
    zero_taus = [0.10, 0.20, 0.35] if zero_taus is None else list(zero_taus)
    inputs = load_priority_population_inputs(config.input_dir)
    population_root = Path(config.output_dir) / inputs.population_id
    rows: list[dict] = []

    configs: list[PriorityPopulationSEMConfig] = []
    if include_none:
        configs.append(replace(config, drift_mode="none"))
    configs.extend(replace(config, drift_mode="zero", drift_tau=float(tau)) for tau in zero_taus)
    if include_current:
        configs.append(replace(config, drift_mode="current"))

    for cfg in configs:
        paths = run_priority_population_sem(cfg)
        summary = pd.read_csv(paths["run_summary"])
        summary_map = dict(zip(summary["metric"], summary["value"]))
        drift = pd.read_csv(paths["drift"])
        endpoint = pd.read_csv(paths["compact_endpoint"])
        key_endpoint = endpoint[endpoint["variable"].isin(["stigma_ahs", "prep_used", "hivtest12"])]
        row = {
            "population_id": inputs.population_id,
            "population_label": inputs.population_label,
            "drift_mode": cfg.drift_mode,
            "drift_tau": np.nan if cfg.drift_mode != "zero" else float(cfg.drift_tau),
            "mean_fit_rmse_pp": float(summary_map.get("mean_fit_rmse_pp", np.nan)),
            "mean_fit_mae_pp": float(summary_map.get("mean_fit_mae_pp", np.nan)),
        }
        for _, r in drift.iterrows():
            if r["variable"] in {"prep_used", "risk_behavior"}:
                row[f"drift_{r['variable']}"] = float(r["drift_final"])
        for scenario in ["s1_reduce_ahs", "s7_combined_stigma_package"]:
            for variable in ["stigma_ahs", "prep_used", "hivtest12"]:
                d = key_endpoint[(key_endpoint["scenario"] == scenario) & (key_endpoint["variable"] == variable)]
                if len(d):
                    row[f"{scenario}_{variable}_delta_pp"] = float(d["delta_pp"].iloc[0])
        rows.append(row)

    population_root.mkdir(parents=True, exist_ok=True)
    sensitivity = pd.DataFrame(rows)
    out_path = population_root / "drift_sensitivity_summary.csv"
    sensitivity.to_csv(out_path, index=False)
    return out_path
