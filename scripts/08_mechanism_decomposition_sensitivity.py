#!/usr/bin/env python3
from __future__ import annotations

"""Alternative mechanism decomposition sensitivity prototype.

This script tests whether meaningfully different J/c decompositions can fit the
same observed state trajectories under the same centered-logit variable system
and hard structural sign constraints.

It intentionally does not modify the main model config files. For each mechanism
family, it writes temporary structural-prior and variable-config CSVs under the
output directory, points SEMConfig at those temporary files, fits the standard
SBRunner, and summarizes selected states.
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
from models.sbm.prediction.predictor import Predictor
from models.shared.intervention import build_relationship_interventions, build_state_interventions
from models.shared.alignment import extend_to_end_year
from core.math.transforms import Transforms


STIGMA_VARS = {"stigma_ahs", "stigma_gss", "stigma_family"}
DEFAULT_STATES = ["CA", "TX", "TN", "AZ"]
DEFAULT_SCENARIOS = ["s1_reduce_ahs", "s4_weaken_ahs_to_prep", "s7_combined_stigma_package"]


@dataclass(frozen=True)
class FamilySpec:
    name: str
    description: str
    stigma_prior_scale: float = 1.0
    all_prior_tau_scale: float = 1.0
    stigma_prior_tau_scale: float = 1.0
    drift_enabled_override: bool | None = None
    drift_tau_scale: float = 1.0
    drift_mean_scale: float = 1.0
    wX_scale: float = 1.0
    wR_scale: float = 1.0
    shrink_scale: float = 1.0
    B0_diag: float | None = None
    B0_off_diag: float | None = None


FAMILIES: dict[str, FamilySpec] = {
    "reference": FamilySpec(
        name="reference",
        description="Current v8 model settings.",
    ),
    "high_stigma_coupling": FamilySpec(
        name="high_stigma_coupling",
        description="Stronger stigma-related priors; tests compatibility of larger stigma coupling.",
        stigma_prior_scale=1.5,
        stigma_prior_tau_scale=0.75,
    ),
    "high_drift_low_pathway": FamilySpec(
        name="high_drift_low_pathway",
        description="More flexible drift and weaker structural-prior pull; tests exogenous-trend explanation.",
        all_prior_tau_scale=2.0,
        drift_tau_scale=2.0,
        wX_scale=0.75,
    ),
    "high_persistence": FamilySpec(
        name="high_persistence",
        description="Initialization favoring stronger self-persistence; tests momentum-heavy decompositions.",
        B0_diag=0.05,
        B0_off_diag=0.05,
    ),
    "low_drift_high_pathway": FamilySpec(
        name="low_drift_high_pathway",
        description="Tighter drift around zero and stronger pathway priors; tests endogenous-pathway explanation.",
        drift_tau_scale=0.5,
        drift_mean_scale=0.5,
        all_prior_tau_scale=0.75,
        wX_scale=1.25,
    ),
}


def _copy_and_modify_structural_priors(base_path: Path, out_path: Path, spec: FamilySpec) -> Path:
    df = pd.read_csv(base_path)
    df_out = df.copy()
    finite_mean = pd.to_numeric(df_out["prior_mean"], errors="coerce").notna()
    finite_tau = pd.to_numeric(df_out["prior_tau"], errors="coerce").notna()

    df_out.loc[finite_tau, "prior_tau"] = pd.to_numeric(df_out.loc[finite_tau, "prior_tau"]) * spec.all_prior_tau_scale

    stigma_edge = df_out["source"].isin(STIGMA_VARS) | df_out["target"].isin(STIGMA_VARS)
    mask_mean = finite_mean & stigma_edge
    mask_tau = finite_tau & stigma_edge
    df_out.loc[mask_mean, "prior_mean"] = pd.to_numeric(df_out.loc[mask_mean, "prior_mean"]) * spec.stigma_prior_scale
    df_out.loc[mask_tau, "prior_tau"] = pd.to_numeric(df_out.loc[mask_tau, "prior_tau"]) * spec.stigma_prior_tau_scale

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return out_path


def _copy_and_modify_variable_config(base_path: Path, out_path: Path, spec: FamilySpec) -> Path:
    df = pd.read_csv(base_path)
    df_out = df.copy()
    if spec.drift_enabled_override is not None:
        df_out["drift_enabled"] = 1 if spec.drift_enabled_override else 0

    tau = pd.to_numeric(df_out["drift_prior_tau"], errors="coerce")
    mean = pd.to_numeric(df_out["drift_prior_mean"], errors="coerce")
    tau_mask = tau.notna() & (tau > 0)
    mean_mask = mean.notna()
    df_out.loc[tau_mask, "drift_prior_tau"] = tau[tau_mask] * spec.drift_tau_scale
    df_out.loc[mean_mask, "drift_prior_mean"] = mean[mean_mask] * spec.drift_mean_scale

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return out_path


def _fit_unit_subset(runner: SBRunner, states: list[str]):
    output = runner.run(fit=True, predict=True)
    state_set = set(states)
    selected = {uid: fit for uid, fit in output.fit.results.items() if uid in state_set}
    missing = [s for s in states if s not in selected]
    if missing:
        raise ValueError(f"Requested states missing from fit output: {missing}")
    return output, selected


def _last_components(fit) -> dict[str, float]:
    diagnostics = fit.loss_diagnostics or []
    if not diagnostics:
        return {}
    last = diagnostics[-1].get("diagnostics", {})
    comps = dict(last.get("final_components", {}) or {})
    comps["final_loss"] = float(last.get("final_loss", np.nan))
    comps["success"] = bool(last.get("success", False))
    comps["n_iter"] = int(last.get("n_iter", -1))
    return comps


def _spectral_radius(J: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(J, dtype=float)))))


def _summarize_fit(family: FamilySpec, selected: dict, v_names: list[str]) -> list[dict]:
    rows = []
    idx = {v: i for i, v in enumerate(v_names)}
    edges = [
        ("stigma_ahs", "prep_used"),
        ("stigma_ahs", "out_gid"),
        ("stigma_ahs", "hivtest12"),
        ("seehcp", "stigma_ahs"),
        ("out_gid", "stigma_ahs"),
        ("out_gid", "prep_used"),
        ("hivtest12", "prep_used"),
    ]
    for unit_id, fit in selected.items():
        J = np.asarray(fit.J, dtype=float)[:, :, -1]
        c = np.zeros(J.shape[0], dtype=float) if fit.drift is None else np.asarray(fit.drift, dtype=float)
        comps = _last_components(fit)
        row = {
            "family": family.name,
            "description": family.description,
            "unit_id": unit_id,
            "success": comps.get("success", False),
            "final_loss": comps.get("final_loss", np.nan),
            "LX": comps.get("LX", np.nan),
            "LR": comps.get("LR", np.nan),
            "L_prior_J": comps.get("L_prior_J", np.nan),
            "L_prior_c": comps.get("L_prior_c", np.nan),
            "L_stab": comps.get("L_stab", np.nan),
            "rho_J": _spectral_radius(J),
            "drift_norm": float(np.linalg.norm(c)),
            "J_abs_mean": float(np.mean(np.abs(J))),
            "diag_mean": float(np.mean(np.diag(J))),
        }
        for src, tgt in edges:
            if src in idx and tgt in idx:
                row[f"J_{src}_to_{tgt}"] = float(J[idx[tgt], idx[src]])
        for var in v_names:
            row[f"drift_{var}"] = float(c[idx[var]])
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
    state_iv = build_state_interventions(
        data_unit,
        sem_years,
        v_names,
        codes=spec.get("state_codes", []),
        duration_steps=duration_steps,
    )
    rel_iv = build_relationship_interventions(
        data_unit,
        sem_years,
        v_names,
        codes=spec.get("relationship_codes", []),
        duration_steps=duration_steps,
    )
    scenario_probs, _ = predictor.predict_trajectory(
        J,
        x0,
        c,
        n_steps,
        state_interventions=state_iv,
        rel_interventions=rel_iv,
        baseline_probs=baseline_probs,
    )
    return sem_years, scenario_probs


def _summarize_scenarios(output, family: FamilySpec, states: list[str], scenarios: list[str], target_end_year: int, duration_steps: int) -> list[dict]:
    rows = []
    v_names = list(output.fit.v_names)
    var_idx = {v: i for i, v in enumerate(v_names)}
    variables = ["stigma_ahs", "out_gid", "seehcp", "prep_used", "hivtest12", "risk_behavior"]
    for unit_id in states:
        years, baseline = _predict_state_scenario(output, unit_id, "baseline", target_end_year, duration_steps)
        target_idx = int(np.where(np.asarray(years, dtype=int) == int(target_end_year))[0][0])
        for scenario_code in scenarios:
            _, scenario = _predict_state_scenario(output, unit_id, scenario_code, target_end_year, duration_steps)
            for var in variables:
                if var not in var_idx:
                    continue
                k = var_idx[var]
                b = float(baseline[k, target_idx])
                s = float(scenario[k, target_idx])
                rows.append({
                    "family": family.name,
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


def _make_configs(base_sem: SEMConfig, base_opt: OptimConfig, base_shrink: ShrinkageConfig, output_dir: Path, spec: FamilySpec):
    family_dir = output_dir / "family_configs" / spec.name
    structural_path = _copy_and_modify_structural_priors(
        Path(base_sem.structural_prior_path),
        family_dir / "sem_structural_priors_v3.csv",
        spec,
    )
    variable_path = _copy_and_modify_variable_config(
        Path(base_sem.variable_config_path),
        family_dir / "sem_variable_config_v3.csv",
        spec,
    )
    sem = SEMConfig(
        data_path=base_sem.data_path,
        v_names=list(base_sem.v_names),
        use_centered_logit=base_sem.use_centered_logit,
        variable_config_path=variable_path,
        structural_prior_path=structural_path,
        n_steps=base_sem.n_steps,
        n_forecast=base_sem.n_forecast,
    )
    opt = OptimConfig()
    opt.wX = base_opt.wX * spec.wX_scale
    opt.wR = base_opt.wR * spec.wR_scale
    opt.B0_diag = spec.B0_diag if spec.B0_diag is not None else base_opt.B0_diag
    opt.B0_off_diag = spec.B0_off_diag if spec.B0_off_diag is not None else base_opt.B0_off_diag
    opt.bounds = base_opt.bounds
    opt.sigma_eta_scale = base_opt.sigma_eta_scale
    opt.flag_reinitialize_JmI0 = base_opt.flag_reinitialize_JmI0
    opt.use_diag_whitening_x = base_opt.use_diag_whitening_x
    opt.normalize_x_by_counts = base_opt.normalize_x_by_counts
    opt.normalize_r_by_counts = base_opt.normalize_r_by_counts
    opt.normalize_components_by_baseline = base_opt.normalize_components_by_baseline
    opt.norm_eps = base_opt.norm_eps
    opt.drift_bounds = base_opt.drift_bounds
    opt.w_stab = base_opt.w_stab
    opt.stability_eps = base_opt.stability_eps
    opt.stability_fd_eps = base_opt.stability_fd_eps
    opt.maxiter = base_opt.maxiter
    opt.ftol = base_opt.ftol
    opt.gtol = base_opt.gtol
    opt.maxls = base_opt.maxls

    shrink = ShrinkageConfig(
        state_size_ref=base_shrink.state_size_ref,
        state_shrink_n_ref=base_shrink.state_shrink_n_ref,
        state_shrink_n_low=base_shrink.state_shrink_n_low,
        state_shrink_n_high=base_shrink.state_shrink_n_high,
        state_shrink_boost_super=base_shrink.state_shrink_boost_super,
        state_shrink_boost_mid_max=base_shrink.state_shrink_boost_mid_max,
        state_shrink_cap=base_shrink.state_shrink_cap,
        wJ_state=base_shrink.wJ_state * spec.shrink_scale,
        wJ=base_shrink.wJ * spec.shrink_scale,
    )
    return sem, opt, shrink


def run_sensitivity(args) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    family_names = [f.strip() for f in args.families.split(",") if f.strip()]
    scenario_codes = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown_families = [f for f in family_names if f not in FAMILIES]
    unknown_scenarios = [s for s in scenario_codes if s not in SCENARIO_CODEBOOK]
    if unknown_families:
        raise ValueError(f"Unknown families: {unknown_families}. Available: {sorted(FAMILIES)}")
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {unknown_scenarios}. Available: {sorted(SCENARIO_CODEBOOK)}")

    base_sem = SEMConfig(
        data_path=Path(args.data_path),
        structural_prior_path=Path(args.structural_prior_path),
        variable_config_path=Path(args.variable_config_path),
    )
    base_opt = OptimConfig()
    base_shrink = ShrinkageConfig()

    fit_rows: list[dict] = []
    scenario_rows: list[dict] = []
    family_rows: list[dict] = []

    for family_name in family_names:
        spec = FAMILIES[family_name]
        print(f"fitting family: {family_name}")
        sem, opt, shrink = _make_configs(base_sem, base_opt, base_shrink, output_dir, spec)
        runner = SBRunner(sem, opt, shrink)
        output, selected = _fit_unit_subset(runner, states)
        fit_rows.extend(_summarize_fit(spec, selected, list(output.fit.v_names)))
        scenario_rows.extend(_summarize_scenarios(output, spec, states, scenario_codes, args.target_end_year, args.intervention_duration_steps))
        family_rows.append({
            "family": spec.name,
            "description": spec.description,
            "stigma_prior_scale": spec.stigma_prior_scale,
            "all_prior_tau_scale": spec.all_prior_tau_scale,
            "stigma_prior_tau_scale": spec.stigma_prior_tau_scale,
            "drift_tau_scale": spec.drift_tau_scale,
            "drift_mean_scale": spec.drift_mean_scale,
            "wX_scale": spec.wX_scale,
            "wR_scale": spec.wR_scale,
            "B0_diag": opt.B0_diag,
            "B0_off_diag": opt.B0_off_diag,
        })

    fit_df = pd.DataFrame(fit_rows)
    scenario_df = pd.DataFrame(scenario_rows)
    family_df = pd.DataFrame(family_rows)

    ref_cols = ["unit_id", "final_loss", "LX", "LR", "drift_norm", "J_abs_mean"]
    ref = fit_df[fit_df["family"] == "reference"][ref_cols].rename(columns={
        "final_loss": "reference_final_loss",
        "LX": "reference_LX",
        "LR": "reference_LR",
        "drift_norm": "reference_drift_norm",
        "J_abs_mean": "reference_J_abs_mean",
    })
    fit_df = fit_df.merge(ref, on="unit_id", how="left")
    fit_df["relative_loss_vs_reference"] = fit_df["final_loss"] / fit_df["reference_final_loss"]
    fit_df["relative_LX_vs_reference"] = fit_df["LX"] / fit_df["reference_LX"]
    fit_df["relative_LR_vs_reference"] = fit_df["LR"] / fit_df["reference_LR"]
    fit_df["relative_drift_norm_vs_reference"] = fit_df["drift_norm"] / fit_df["reference_drift_norm"]

    fit_path = output_dir / "mechanism_sensitivity_fit_summary.csv"
    scenario_path = output_dir / "mechanism_sensitivity_scenario_summary.csv"
    family_path = output_dir / "mechanism_sensitivity_family_specs.csv"
    fit_df.to_csv(fit_path, index=False)
    scenario_df.to_csv(scenario_path, index=False)
    family_df.to_csv(family_path, index=False)
    return {"fit_summary": fit_path, "scenario_summary": scenario_path, "family_specs": family_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run alternative mechanism decomposition sensitivity prototype")
    parser.add_argument("--data-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx"))
    parser.add_argument("--structural-prior-path", type=Path, default=Path("config/sem_structural_priors_v3.csv"))
    parser.add_argument("--variable-config-path", type=Path, default=Path("config/sem_variable_config_v3.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/mechanism_decomposition_sensitivity"))
    parser.add_argument("--states", type=str, default=",".join(DEFAULT_STATES))
    parser.add_argument("--families", type=str, default=",".join(FAMILIES.keys()))
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--target-end-year", type=int, default=2036)
    parser.add_argument("--intervention-duration-steps", type=int, default=3)
    args = parser.parse_args()

    paths = run_sensitivity(args)
    for key, path in paths.items():
        print(f"wrote {key}: {path}")


if __name__ == "__main__":
    main()
