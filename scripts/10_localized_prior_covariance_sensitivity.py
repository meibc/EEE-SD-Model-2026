#!/usr/bin/env python3
from __future__ import annotations

"""Localized prior/covariance sensitivity diagnostic.

Primary purpose: test whether selected states' pathway estimates and scenarios
change when structural priors and/or covariance targets are localized by census
region/division. This is a diagnostic runner, not the production model path.

Guardrails included here:
- PrEP-only drift using national centered-logit PrEP trend.
- No risk_behavior drift.
- Sensitivity-only PrEP self-persistence prior J[prep,prep] ~ N(0.75, 0.20^2).
- Same hard sign constraints as supplied by the selected prior table.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.optimization import OptimConfig
from config.sem import SEMConfig
from config.shrinkage import ShrinkageConfig
from config.interventions import SCENARIO_CODEBOOK
from models.sbm.runner import SBRunner
from models.sbm.estimation.jacobian import JacobianEstimator
from models.sbm.estimation.shrinkage import ShrinkageCalculator
from models.sbm.estimation.transforms import FeatureTransformer
from models.sbm.prediction.predictor import Predictor
from models.shared.intervention import build_relationship_interventions, build_state_interventions
from models.shared.alignment import extend_to_end_year
from pipeline.results import FitResult, FitResults, RunOutput
from core.math.transforms import Transforms


DEFAULT_STATES = ["TN", "TX", "GA", "FL", "NC", "CA", "AZ", "NY"]
DEFAULT_SCENARIOS = ["s1_reduce_ahs", "s4_weaken_ahs_to_prep", "s7_combined_stigma_package"]

SPECS = {
    "primary": ("national", "pooled"),
    "national_region_cov": ("national", "region"),
    "national_division_cov": ("national", "division"),
    "regional_pooled_cov": ("region", "pooled"),
    "regional_region_cov": ("region", "region"),
    "regional_division_cov": ("region", "division"),
    "division_division_cov": ("division", "division"),
}


@dataclass(frozen=True)
class LocalizedPaths:
    workflow_dir: Path
    national_prior_path: Path
    region_prior_path: Path
    division_prior_path: Path
    covariance_long_path: Path
    geography_path: Path

    @classmethod
    def from_workflow_dir(cls, workflow_dir: Path) -> "LocalizedPaths":
        return cls(
            workflow_dir=workflow_dir,
            national_prior_path=workflow_dir / "structural_priors_national.csv",
            region_prior_path=workflow_dir / "structural_priors_region.csv",
            division_prior_path=workflow_dir / "structural_priors_division.csv",
            covariance_long_path=workflow_dir / "state_weighted_covariance_long.csv",
            geography_path=workflow_dir / "state_geography_mapping.csv",
        )


def _as_float_or_nan(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _prior_arrays_from_df(df: pd.DataFrame, v_names: list[str], geo_level: str, geo_id: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if geo_level == "national":
        sub = df[df["geo_level"].astype(str).eq("national")].copy()
    else:
        sub = df[(df["geo_level"].astype(str).eq(geo_level)) & (df["geo_id"].astype(str).eq(str(geo_id)))].copy()
    if sub.empty:
        raise ValueError(f"No prior rows found for geo_level={geo_level}, geo_id={geo_id}")
    m = len(v_names)
    idx = {v: i for i, v in enumerate(v_names)}
    sign = np.zeros((m, m), dtype=int)
    mean = np.full((m, m), np.nan, dtype=float)
    tau = np.full((m, m), np.nan, dtype=float)
    for row in sub.to_dict(orient="records"):
        src = str(row["source"])
        tgt = str(row["target"])
        if src not in idx or tgt not in idx:
            continue
        i = idx[tgt]
        j = idx[src]
        if i == j:
            continue
        s = _as_float_or_nan(row.get("sign"))
        if np.isfinite(s) and s != 0:
            sign[i, j] = int(np.sign(s))
        pm = _as_float_or_nan(row.get("prior_mean"))
        pt = _as_float_or_nan(row.get("prior_tau"))
        if np.isfinite(pm):
            mean[i, j] = pm
            if sign[i, j] == 0 and pm != 0:
                sign[i, j] = int(np.sign(pm))
        if np.isfinite(pt) and pt > 0:
            tau[i, j] = pt
    return sign, mean, tau


def _weighted_average_cov(mats: list[np.ndarray], weights: list[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    mats_arr = np.asarray(mats, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(mats), dtype=float)
    return np.tensordot(weights / weights.sum(), mats_arr, axes=(0, 0))


def _cov_matrix_from_long(cov_long: pd.DataFrame, v_names: list[str], states: list[str]) -> np.ndarray:
    idx = {v: i for i, v in enumerate(v_names)}
    m = len(v_names)
    mats = []
    weights = []
    for state in states:
        sub = cov_long[cov_long["state"].astype(str).eq(str(state))]
        if sub.empty:
            continue
        mat = np.zeros((m, m), dtype=float)
        seen = np.zeros((m, m), dtype=bool)
        for row in sub.to_dict(orient="records"):
            src = str(row["source"])
            tgt = str(row["target"])
            if src in idx and tgt in idx:
                mat[idx[src], idx[tgt]] = float(row["covariance"])
                seen[idx[src], idx[tgt]] = True
        if not seen.all():
            missing = np.argwhere(~seen)
            raise ValueError(f"Missing covariance entries for state={state}; first missing={missing[:5].tolist()}")
        mats.append(mat)
        w = pd.to_numeric(sub["n_eff_pairwise"], errors="coerce").median()
        weights.append(float(w) if np.isfinite(w) else 1.0)
    if not mats:
        raise ValueError(f"No covariance matrices available for states={states}")
    mat = _weighted_average_cov(mats, weights)
    # Symmetrize defensively and keep diagonal nonnegative.
    mat = 0.5 * (mat + mat.T)
    diag = np.diag(mat).copy()
    if np.any(diag <= 0):
        raise ValueError(f"Covariance matrix has nonpositive diagonal: {diag}")
    return mat


def _build_cov_targets(cov_long: pd.DataFrame, geo: pd.DataFrame, v_names: list[str]) -> dict[tuple[str, str], np.ndarray]:
    targets: dict[tuple[str, str], np.ndarray] = {}
    all_states = sorted(geo["state"].astype(str).tolist())
    targets[("pooled", "pooled")] = _cov_matrix_from_long(cov_long, v_names, all_states)
    for region, sub in geo.groupby("census_region"):
        targets[("region", str(region))] = _cov_matrix_from_long(cov_long, v_names, sorted(sub["state"].astype(str).tolist()))
    for division, sub in geo.groupby("census_division"):
        targets[("division", str(division))] = _cov_matrix_from_long(cov_long, v_names, sorted(sub["state"].astype(str).tolist()))
    return targets


def _compute_national_prep_drift(runner: SBRunner, data) -> float:
    usa = next((u for u in data.units if str(u.id) in {"USA", "US"}), None)
    if usa is None:
        raise ValueError("Could not find USA/US unit to estimate PrEP drift")
    v_names = list(data.v_names)
    prep_idx = v_names.index("prep_used")
    transforms = Transforms()
    ref_logits = transforms.logit(runner.reference_probs)
    x = transforms.logit(np.asarray(usa.values[prep_idx], dtype=float)) - ref_logits[prep_idx]
    return float((x[-1] - x[0]) / max(x.size - 1, 1))


def _drift_arrays(
    v_names: list[str],
    prep_drift_mean: float,
    prep_drift_tau: float,
    extra_drift_vars: list[str] | None = None,
    extra_drift_tau: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = len(v_names)
    enabled = np.zeros(m, dtype=bool)
    mean = np.zeros(m, dtype=float)
    tau = np.full(m, np.nan, dtype=float)

    k = v_names.index("prep_used")
    enabled[k] = True
    mean[k] = prep_drift_mean
    tau[k] = prep_drift_tau

    for var in extra_drift_vars or []:
        if var not in v_names:
            raise ValueError(f"Unknown extra drift variable: {var}; available={v_names}")
        if var == "prep_used":
            continue
        j = v_names.index(var)
        enabled[j] = True
        mean[j] = 0.0
        tau[j] = float(extra_drift_tau)
    return enabled, mean, tau


def _unit_geo_info(unit, geo: pd.DataFrame) -> tuple[str | None, str | None]:
    if unit.kind == "state":
        row = geo[geo["state"].astype(str).eq(str(unit.id))]
        if row.empty:
            raise ValueError(f"State missing from geography mapping: {unit.id}")
        return str(row.iloc[0]["census_region"]), str(row.iloc[0]["census_division"])
    if unit.kind == "division":
        # The Unit.census_div is numeric; for this diagnostic, match by state rows' numeric order is not reliable.
        return None, None
    return None, None


def _select_prior(prior_level: str, unit, geo: pd.DataFrame, prior_dfs: dict[str, pd.DataFrame], v_names: list[str], fallback_national: tuple[np.ndarray, np.ndarray, np.ndarray]):
    if unit.kind != "state" or prior_level == "national":
        return fallback_national
    region, division = _unit_geo_info(unit, geo)
    if prior_level == "region":
        return _prior_arrays_from_df(prior_dfs["region"], v_names, "region", region)
    if prior_level == "division":
        return _prior_arrays_from_df(prior_dfs["division"], v_names, "division", division)
    raise ValueError(f"Unknown prior_level={prior_level}")


def _select_cov(cov_level: str, unit, geo: pd.DataFrame, cov_targets: dict[tuple[str, str], np.ndarray], pooled_cov: np.ndarray) -> np.ndarray:
    if unit.kind != "state" or cov_level == "pooled":
        return pooled_cov
    region, division = _unit_geo_info(unit, geo)
    if cov_level == "region":
        return cov_targets[("region", region)]
    if cov_level == "division":
        return cov_targets[("division", division)]
    raise ValueError(f"Unknown cov_level={cov_level}")


def _apply_diagonal_priors(
    mean: np.ndarray,
    tau: np.ndarray,
    v_names: list[str],
    prep_diag_mean: float,
    prep_diag_tau: float,
    extra_diag_vars: list[str] | None = None,
    extra_diag_mean: float = 0.85,
    extra_diag_tau: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    mean = mean.copy()
    tau = tau.copy()

    k = v_names.index("prep_used")
    mean[k, k] = float(prep_diag_mean)
    tau[k, k] = float(prep_diag_tau)

    for var in extra_diag_vars or []:
        if var not in v_names:
            raise ValueError(f"Unknown extra diagonal prior variable: {var}; available={v_names}")
        if var == "prep_used":
            continue
        j = v_names.index(var)
        mean[j, j] = float(extra_diag_mean)
        tau[j, j] = float(extra_diag_tau)
    return mean, tau


def _fit_localized(spec_name: str, prior_level: str, cov_level: str, base_runner: SBRunner, data, geo: pd.DataFrame, prior_dfs: dict[str, pd.DataFrame], cov_targets: dict[tuple[str, str], np.ndarray], prep_drift_mean: float, prep_drift_tau: float, prep_diag_mean: float, prep_diag_tau: float, extra_drift_vars: list[str] | None = None, extra_drift_tau: float = 0.15, extra_diag_vars: list[str] | None = None, extra_diag_mean: float = 0.85, extra_diag_tau: float = 0.20):
    opt = OptimConfig()
    estimator = JacobianEstimator(opt)
    shrink_calc = ShrinkageCalculator(ShrinkageConfig())
    transformer = FeatureTransformer(reference_probs=base_runner.reference_probs)
    v_names = list(data.v_names)
    drift_enabled, drift_prior_mean, drift_prior_tau = _drift_arrays(v_names, prep_drift_mean, prep_drift_tau, extra_drift_vars, extra_drift_tau)
    national_prior = _prior_arrays_from_df(prior_dfs["national"], v_names, "national", "National")
    pooled_cov = cov_targets[("pooled", "pooled")]
    results = {}
    us_J, div_Js = None, {}
    u = np.zeros(data.M.shape[0], dtype=float)

    for unit in data.units:
        sign, mean, tau = _select_prior(prior_level, unit, geo, prior_dfs, v_names, national_prior)
        mean, tau = _apply_diagonal_priors(mean, tau, v_names, prep_diag_mean, prep_diag_tau, extra_diag_vars, extra_diag_mean, extra_diag_tau)
        M = np.eye(len(v_names), dtype=int) + (sign != 0).astype(int)
        SigmaY = _select_cov(cov_level, unit, geo, cov_targets, pooled_cov)
        Xbar, SigmaX = transformer.transform(unit.values, SigmaY)
        J_ref, wJ = shrink_calc.get_params(unit, us_J, div_Js)
        J_est = estimator.fit(
            Xbar=Xbar,
            SigmaX=SigmaX,
            M=M,
            sign_mask=sign,
            u=u,
            J_ref=J_ref,
            wJ=wJ,
            structural_prior_mean=mean,
            structural_prior_tau=tau,
            drift_enabled=drift_enabled,
            drift_prior_mean=drift_prior_mean,
            drift_prior_tau=drift_prior_tau,
        )
        drift_seq = estimator.last_drift_sequence
        drift_final = drift_seq[:, -1].copy() if drift_seq is not None else np.zeros(data.M.shape[0])
        results[unit.id] = FitResult(
            unit_id=unit.id,
            J=J_est,
            Ybar=unit.values,
            Xbar=Xbar,
            drift=drift_final,
            loss_diagnostics=estimator.last_fit_history,
        )
        J_final = J_est[:, :, -1]
        if unit.id in ("USA", "US"):
            us_J = J_final
        elif unit.kind == "division":
            div_Js[unit.census_div] = J_final
    return RunOutput(inputs=data, fit=FitResults(results=results, ts=data.ts, v_names=v_names), predictions=None)


def _last_components(fit) -> dict[str, float]:
    diagnostics = fit.loss_diagnostics or []
    if not diagnostics:
        return {}
    last = diagnostics[-1].get("diagnostics", {})
    comps = dict(last.get("final_components", {}) or {})
    comps["final_loss"] = float(last.get("final_loss", np.nan))
    comps["success"] = bool(last.get("success", False))
    return comps


def _spectral_radius(J: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(J, dtype=float)))))


def _summarize_fit(output, spec_name: str, prior_level: str, cov_level: str, states: list[str], prep_drift_mean: float) -> list[dict]:
    rows = []
    v_names = list(output.fit.v_names)
    idx = {v: i for i, v in enumerate(v_names)}
    edges = [
        ("stigma_ahs", "prep_used"),
        ("stigma_ahs", "out_gid"),
        ("stigma_ahs", "hivtest12"),
        ("seehcp", "stigma_ahs"),
        ("out_gid", "stigma_ahs"),
        ("hivtest12", "prep_used"),
        ("out_gid", "prep_used"),
        ("risk_behavior", "prep_used"),
    ]
    for unit_id in states:
        fit = output.fit.results[unit_id]
        J = np.asarray(fit.J, dtype=float)[:, :, -1]
        c = np.zeros(J.shape[0], dtype=float) if fit.drift is None else np.asarray(fit.drift, dtype=float)
        comps = _last_components(fit)
        row = {
            "spec": spec_name,
            "prior_level": prior_level,
            "covariance_level": cov_level,
            "unit_id": unit_id,
            "success": comps.get("success", False),
            "final_loss": comps.get("final_loss", np.nan),
            "LX": comps.get("LX", np.nan),
            "LR": comps.get("LR", np.nan),
            "L_prior_J": comps.get("L_prior_J", np.nan),
            "L_prior_c": comps.get("L_prior_c", np.nan),
            "L_stab": comps.get("L_stab", np.nan),
            "rho_J": _spectral_radius(J),
            "diag_mean": float(np.mean(np.diag(J))),
            "J_prep_diag": float(J[idx["prep_used"], idx["prep_used"]]),
            "drift_prep_used": float(c[idx["prep_used"]]),
            "national_prep_drift_mean_used": prep_drift_mean,
        }
        for var in v_names:
            row[f"drift_{var}"] = float(c[idx[var]])
            row[f"J_{var}_diag"] = float(J[idx[var], idx[var]])
        for src, tgt in edges:
            row[f"J_{src}_to_{tgt}"] = float(J[idx[tgt], idx[src]])
        rows.append(row)
    return rows


def _predict_state_scenario(output, unit_id: str, scenario_code: str, target_end_year: int, duration_steps: int):
    fit = output.fit.results[unit_id]
    data_unit = next(u for u in output.inputs.units if u.id == unit_id)
    v_names = list(output.fit.v_names)
    sem_years = extend_to_end_year(output.fit.ts, target_end_year=target_end_year, step=2)
    n_steps = len(sem_years)
    J = np.asarray(fit.J, dtype=float)[:, :, -1]
    c = np.zeros(J.shape[0], dtype=float) if fit.drift is None else np.asarray(fit.drift, dtype=float)
    x0 = np.asarray(fit.Xbar[:, 0], dtype=float)
    transforms = Transforms()
    y0 = np.asarray(fit.Ybar[:, 0], dtype=float)
    ref_logits = transforms.logit(y0) - x0
    ref_probs = transforms.inverse_logit(ref_logits)
    predictor = Predictor(reference_probs=ref_probs)
    baseline_probs, _ = predictor.predict_trajectory(J, x0, c, n_steps)
    if scenario_code == "baseline":
        return sem_years, baseline_probs
    spec = SCENARIO_CODEBOOK[scenario_code]
    state_iv = build_state_interventions(data_unit, sem_years, v_names, spec.get("state_codes", []), duration_steps)
    rel_iv = build_relationship_interventions(data_unit, sem_years, v_names, spec.get("relationship_codes", []), duration_steps)
    scenario_probs, _ = predictor.predict_trajectory(
        J, x0, c, n_steps,
        state_interventions=state_iv,
        rel_interventions=rel_iv,
        baseline_probs=baseline_probs,
    )
    return sem_years, scenario_probs


def _summarize_scenarios(output, spec_name: str, prior_level: str, cov_level: str, states: list[str], scenarios: list[str], target_end_year: int, duration_steps: int) -> list[dict]:
    rows = []
    v_names = list(output.fit.v_names)
    idx = {v: i for i, v in enumerate(v_names)}
    variables = ["stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "prep_used", "hivtest12", "risk_behavior"]
    for unit_id in states:
        years, baseline = _predict_state_scenario(output, unit_id, "baseline", target_end_year, duration_steps)
        target_idx = int(np.where(np.asarray(years, dtype=int) == int(target_end_year))[0][0])
        for scenario_code in scenarios:
            _, scenario = _predict_state_scenario(output, unit_id, scenario_code, target_end_year, duration_steps)
            for var in variables:
                k = idx[var]
                b = float(baseline[k, target_idx])
                s = float(scenario[k, target_idx])
                rows.append({
                    "spec": spec_name,
                    "prior_level": prior_level,
                    "covariance_level": cov_level,
                    "unit_id": unit_id,
                    "scenario": scenario_code,
                    "scenario_label": SCENARIO_CODEBOOK[scenario_code].get("label", scenario_code),
                    "year": int(target_end_year),
                    "variable": var,
                    "baseline": b,
                    "scenario_value": s,
                    "delta_pp": 100.0 * (s - b),
                    "pct_change": np.nan if abs(b) < 1e-12 else 100.0 * (s - b) / b,
                })
    return rows


def run(args) -> dict[str, Path]:
    workflow_dir = Path(args.workflow_dir)
    paths = LocalizedPaths.from_workflow_dir(workflow_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    spec_names = [s.strip() for s in args.specs.split(",") if s.strip()]
    unknown = [s for s in spec_names if s not in SPECS]
    if unknown:
        raise ValueError(f"Unknown specs: {unknown}; available={sorted(SPECS)}")

    geo = pd.read_csv(paths.geography_path)
    cov_long = pd.read_csv(paths.covariance_long_path)
    prior_dfs = {
        "national": pd.read_csv(paths.national_prior_path),
        "region": pd.read_csv(paths.region_prior_path),
        "division": pd.read_csv(paths.division_prior_path),
    }

    base_sem = SEMConfig(data_path=Path(args.data_path))
    base_runner = SBRunner(base_sem, OptimConfig(), ShrinkageConfig())
    data = base_runner.data_prep.prepare_inputs()
    v_names = list(data.v_names)
    cov_targets = _build_cov_targets(cov_long, geo, v_names)
    national_prep_drift = _compute_national_prep_drift(base_runner, data)
    prep_drift_mean = national_prep_drift * float(args.prep_drift_scale)

    fit_rows = []
    scenario_rows = []
    spec_rows = []
    for spec_name in spec_names:
        prior_level, cov_level = SPECS[spec_name]
        print(f"fitting spec={spec_name}; prior={prior_level}; covariance={cov_level}; prep drift={prep_drift_mean:.6f}")
        output = _fit_localized(
            spec_name=spec_name,
            prior_level=prior_level,
            cov_level=cov_level,
            base_runner=base_runner,
            data=data,
            geo=geo,
            prior_dfs=prior_dfs,
            cov_targets=cov_targets,
            prep_drift_mean=prep_drift_mean,
            prep_drift_tau=float(args.prep_drift_tau),
            prep_diag_mean=float(args.prep_diag_mean),
            prep_diag_tau=float(args.prep_diag_tau),
            extra_drift_vars=[v.strip() for v in str(args.extra_drift_vars).split(",") if v.strip()],
            extra_drift_tau=float(args.extra_drift_tau),
            extra_diag_vars=[v.strip() for v in str(args.extra_diag_vars).split(",") if v.strip()],
            extra_diag_mean=float(args.extra_diag_mean),
            extra_diag_tau=float(args.extra_diag_tau),
        )
        fit_rows.extend(_summarize_fit(output, spec_name, prior_level, cov_level, states, prep_drift_mean))
        scenario_rows.extend(_summarize_scenarios(output, spec_name, prior_level, cov_level, states, scenarios, args.target_end_year, args.intervention_duration_steps))
        spec_rows.append({
            "spec": spec_name,
            "prior_level": prior_level,
            "covariance_level": cov_level,
            "prep_drift_scale": float(args.prep_drift_scale),
            "national_prep_drift_raw": national_prep_drift,
            "prep_drift_mean_used": prep_drift_mean,
            "prep_drift_tau": float(args.prep_drift_tau),
            "prep_diag_mean": float(args.prep_diag_mean),
            "prep_diag_tau": float(args.prep_diag_tau),
            "extra_drift_vars": str(args.extra_drift_vars),
            "extra_drift_tau": float(args.extra_drift_tau),
            "extra_diag_vars": str(args.extra_diag_vars),
            "extra_diag_mean": float(args.extra_diag_mean),
            "extra_diag_tau": float(args.extra_diag_tau),
        })

    fit_df = pd.DataFrame(fit_rows)
    ref = fit_df[fit_df["spec"] == "primary"][["unit_id", "LX", "LR", "final_loss"]].rename(columns={
        "LX": "primary_LX",
        "LR": "primary_LR",
        "final_loss": "primary_final_loss",
    })
    fit_df = fit_df.merge(ref, on="unit_id", how="left")
    fit_df["relative_LX_vs_primary"] = fit_df["LX"] / fit_df["primary_LX"]
    fit_df["relative_LR_vs_primary"] = fit_df["LR"] / fit_df["primary_LR"]
    fit_df["relative_final_loss_vs_primary"] = fit_df["final_loss"] / fit_df["primary_final_loss"]

    fit_path = output_dir / "localized_prior_covariance_fit_summary.csv"
    scenario_path = output_dir / "localized_prior_covariance_scenario_summary.csv"
    specs_path = output_dir / "localized_prior_covariance_specs.csv"
    fit_df.to_csv(fit_path, index=False)
    pd.DataFrame(scenario_rows).to_csv(scenario_path, index=False)
    pd.DataFrame(spec_rows).to_csv(specs_path, index=False)
    return {"fit_summary": fit_path, "scenario_summary": scenario_path, "specs": specs_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run localized prior/covariance sensitivity diagnostic")
    parser.add_argument("--workflow-dir", type=Path, default=Path("/Users/meibinchen/Documents/Codex/2026-08-13/d/outputs/localized_prior_covariance_workflow"))
    parser.add_argument("--data-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/localized_prior_covariance_sensitivity"))
    parser.add_argument("--states", type=str, default=",".join(DEFAULT_STATES))
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--specs", type=str, default=",".join(SPECS.keys()))
    parser.add_argument("--target-end-year", type=int, default=2036)
    parser.add_argument("--intervention-duration-steps", type=int, default=3)
    parser.add_argument("--prep-drift-scale", type=float, default=1.0)
    parser.add_argument("--prep-drift-tau", type=float, default=0.25)
    parser.add_argument("--prep-diag-mean", type=float, default=0.75)
    parser.add_argument("--prep-diag-tau", type=float, default=0.20)
    parser.add_argument("--extra-drift-vars", type=str, default="", help="Comma-separated extra variables allowed zero-centered drift, e.g. stigma_gss,stigma_family")
    parser.add_argument("--extra-drift-tau", type=float, default=0.15, help="Prior tau for extra zero-centered drift variables")
    parser.add_argument("--extra-diag-vars", type=str, default="", help="Comma-separated extra variables receiving weak diagonal persistence priors")
    parser.add_argument("--extra-diag-mean", type=float, default=0.85)
    parser.add_argument("--extra-diag-tau", type=float, default=0.20)
    args = parser.parse_args()
    paths = run(args)
    for key, path in paths.items():
        print(f"wrote {key}: {path}")


if __name__ == "__main__":
    main()
