#!/usr/bin/env python3
from __future__ import annotations

"""PrEP-only drift and masked-suppression sensitivity.

This script keeps the main fitted model untouched. It creates temporary config
CSVs under the output directory and fits the existing stigma dynamics model with:

1. no drift except prep_used,
2. prep_used drift prior centered on the national/USA centered-logit PrEP trend,
3. optional stronger AHS -> PrEP prior to test masked suppression.
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


DEFAULT_STATES = ["CA", "TX", "TN", "AZ"]
DEFAULT_SCENARIOS = ["s1_reduce_ahs", "s4_weaken_ahs_to_prep", "s7_combined_stigma_package"]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    prep_drift_mean_scale: float = 1.0
    prep_drift_tau: float = 0.25
    ahs_prep_prior_mean: float | None = None
    ahs_prep_prior_tau: float | None = None
    prep_diag_prior_mean: float | None = None
    prep_diag_prior_tau: float | None = None


VARIANTS: dict[str, VariantSpec] = {
    "prep_only_drift": VariantSpec(
        name="prep_only_drift",
        description="Only prep_used has drift; drift mean is national centered-logit PrEP trend.",
    ),
    "masked_suppression": VariantSpec(
        name="masked_suppression",
        description="PrEP-only drift plus stronger negative AHS -> PrEP prior.",
        ahs_prep_prior_mean=-0.15,
        ahs_prep_prior_tau=0.40,
    ),
    "high_prep_drift_masked_suppression": VariantSpec(
        name="high_prep_drift_masked_suppression",
        description="Masked suppression with 1.5x national PrEP drift mean.",
        prep_drift_mean_scale=1.5,
        ahs_prep_prior_mean=-0.15,
        ahs_prep_prior_tau=0.40,
    ),
    "low_prep_drift_masked_suppression": VariantSpec(
        name="low_prep_drift_masked_suppression",
        description="Masked suppression with 0.5x national PrEP drift mean.",
        prep_drift_mean_scale=0.5,
        ahs_prep_prior_mean=-0.15,
        ahs_prep_prior_tau=0.40,
    ),
    "strong_masked_suppression": VariantSpec(
        name="strong_masked_suppression",
        description="PrEP-only drift plus strong negative AHS -> PrEP prior.",
        ahs_prep_prior_mean=-0.30,
        ahs_prep_prior_tau=0.20,
    ),
    "very_strong_masked_suppression": VariantSpec(
        name="very_strong_masked_suppression",
        description="PrEP-only drift plus very strong negative AHS -> PrEP prior.",
        ahs_prep_prior_mean=-0.50,
        ahs_prep_prior_tau=0.10,
    ),
    "prep_diag_prior": VariantSpec(
        name="prep_diag_prior",
        description="PrEP-only drift plus data-informed PrEP self-persistence prior.",
        prep_diag_prior_mean=0.75,
        prep_diag_prior_tau=0.20,
    ),
    "prep_diag_masked_suppression": VariantSpec(
        name="prep_diag_masked_suppression",
        description="PrEP-only drift plus PrEP self-persistence prior and moderate AHS -> PrEP suppression prior.",
        ahs_prep_prior_mean=-0.15,
        ahs_prep_prior_tau=0.40,
        prep_diag_prior_mean=0.75,
        prep_diag_prior_tau=0.20,
    ),
    "prep_diag_strong_masked_suppression": VariantSpec(
        name="prep_diag_strong_masked_suppression",
        description="PrEP-only drift plus PrEP self-persistence prior and strong AHS -> PrEP suppression prior.",
        ahs_prep_prior_mean=-0.30,
        ahs_prep_prior_tau=0.20,
        prep_diag_prior_mean=0.75,
        prep_diag_prior_tau=0.20,
    ),
}


def _compute_national_prep_drift(data_path: Path, variable_config_path: Path, structural_prior_path: Path) -> float:
    sem = SEMConfig(
        data_path=data_path,
        variable_config_path=variable_config_path,
        structural_prior_path=structural_prior_path,
    )
    runner = SBRunner(sem, OptimConfig(), ShrinkageConfig())
    data = runner.data_prep.prepare_inputs()
    usa = next((u for u in data.units if str(u.id) in {"USA", "US"}), None)
    if usa is None:
        raise ValueError("Could not find USA/US aggregate unit for national PrEP drift calculation")
    v_names = list(data.v_names)
    prep_idx = v_names.index("prep_used")
    ref_probs = sem.reference_probs()
    transforms = Transforms()
    x = transforms.logit(np.asarray(usa.values[prep_idx], dtype=float)) - transforms.logit(ref_probs[prep_idx])
    if x.size < 2:
        raise ValueError("Need at least two national PrEP observations to estimate drift")
    return float((x[-1] - x[0]) / max(x.size - 1, 1))


def _write_variable_config(base_path: Path, out_path: Path, prep_drift_mean: float, prep_drift_tau: float) -> Path:
    df = pd.read_csv(base_path)
    df_out = df.copy()
    df_out["drift_enabled"] = 0
    df_out["drift_prior_mean"] = 0.0
    df_out["drift_prior_tau"] = ""
    prep = df_out["variable"].astype(str).eq("prep_used")
    if prep.sum() != 1:
        raise ValueError("Expected exactly one prep_used row in variable config")
    df_out.loc[prep, "drift_enabled"] = 1
    df_out.loc[prep, "drift_prior_mean"] = float(prep_drift_mean)
    df_out.loc[prep, "drift_prior_tau"] = float(prep_drift_tau)
    df_out.loc[prep, "drift_note"] = "PrEP-only drift sensitivity; mean from national centered-logit PrEP trend."
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return out_path


def _write_structural_priors(base_path: Path, out_path: Path, spec: VariantSpec) -> Path:
    df = pd.read_csv(base_path)
    df_out = df.copy()
    if spec.ahs_prep_prior_mean is not None:
        mask = df_out["source"].astype(str).eq("stigma_ahs") & df_out["target"].astype(str).eq("prep_used")
        if mask.sum() != 1:
            raise ValueError("Expected exactly one stigma_ahs -> prep_used row in structural priors")
        df_out.loc[mask, "prior_mean"] = float(spec.ahs_prep_prior_mean)
        df_out.loc[mask, "prior_tau"] = float(spec.ahs_prep_prior_tau)
        df_out.loc[mask, "evidence_tier"] = "masked_suppression_sensitivity"
        df_out.loc[mask, "evidence_note"] = "Sensitivity: stronger negative AHS -> PrEP prior to test masked suppression with positive PrEP drift."
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return out_path




def _apply_diagonal_priors(runner: SBRunner, spec: VariantSpec) -> None:
    """Apply sensitivity-only diagonal priors after SEMConfig loading.

    SEMConfig intentionally ignores diagonal rows in the CSV structural prior
    parser. This helper keeps the main model unchanged while allowing this
    diagnostic script to regularize PrEP self-persistence.
    """
    if spec.prep_diag_prior_mean is None:
        return
    idx = {v: i for i, v in enumerate(runner.base.v_names)}
    k = idx["prep_used"]
    runner.structural_prior_mean[k, k] = float(spec.prep_diag_prior_mean)
    runner.structural_prior_tau[k, k] = float(spec.prep_diag_prior_tau)


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


def _summarize_fit(spec: VariantSpec, output, states: list[str], prep_drift_mean: float) -> list[dict]:
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
            "variant": spec.name,
            "description": spec.description,
            "unit_id": unit_id,
            "national_prep_drift_mean_used": prep_drift_mean,
            "success": comps.get("success", False),
            "final_loss": comps.get("final_loss", np.nan),
            "LX": comps.get("LX", np.nan),
            "LR": comps.get("LR", np.nan),
            "L_prior_J": comps.get("L_prior_J", np.nan),
            "L_prior_c": comps.get("L_prior_c", np.nan),
            "L_stab": comps.get("L_stab", np.nan),
            "rho_J": _spectral_radius(J),
            "diag_mean": float(np.mean(np.diag(J))),
            "drift_norm": float(np.linalg.norm(c)),
        }
        for src, tgt in edges:
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
    state_iv = build_state_interventions(data_unit, sem_years, v_names, spec.get("state_codes", []), duration_steps)
    rel_iv = build_relationship_interventions(data_unit, sem_years, v_names, spec.get("relationship_codes", []), duration_steps)
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


def _summarize_scenarios(spec: VariantSpec, output, states: list[str], scenarios: list[str], target_end_year: int, duration_steps: int) -> list[dict]:
    rows = []
    v_names = list(output.fit.v_names)
    idx = {v: i for i, v in enumerate(v_names)}
    variables = ["stigma_ahs", "out_gid", "seehcp", "prep_used", "hivtest12", "risk_behavior"]
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
                    "variant": spec.name,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; available={sorted(VARIANTS)}")

    data_path = Path(args.data_path)
    base_var_path = Path(args.variable_config_path)
    base_prior_path = Path(args.structural_prior_path)
    national_prep_drift = _compute_national_prep_drift(data_path, base_var_path, base_prior_path)

    fit_rows = []
    scenario_rows = []
    variant_rows = []
    for variant_name in variants:
        spec = VARIANTS[variant_name]
        prep_drift_mean = national_prep_drift * spec.prep_drift_mean_scale
        print(f"fitting variant: {variant_name}; prep drift mean={prep_drift_mean:.6f}; tau={spec.prep_drift_tau}")
        cfg_dir = output_dir / "variant_configs" / variant_name
        var_path = _write_variable_config(base_var_path, cfg_dir / "sem_variable_config_v3.csv", prep_drift_mean, spec.prep_drift_tau)
        prior_path = _write_structural_priors(base_prior_path, cfg_dir / "sem_structural_priors_v3.csv", spec)

        sem = SEMConfig(data_path=data_path, variable_config_path=var_path, structural_prior_path=prior_path)
        runner = SBRunner(sem, OptimConfig(), ShrinkageConfig())
        _apply_diagonal_priors(runner, spec)
        output = runner.run(fit=True, predict=True)
        missing = [s for s in states if s not in output.fit.results]
        if missing:
            raise ValueError(f"Requested states missing from fit output: {missing}")
        fit_rows.extend(_summarize_fit(spec, output, states, prep_drift_mean))
        scenario_rows.extend(_summarize_scenarios(spec, output, states, scenarios, args.target_end_year, args.intervention_duration_steps))
        variant_rows.append({
            "variant": spec.name,
            "description": spec.description,
            "national_prep_drift_raw": national_prep_drift,
            "prep_drift_mean_scale": spec.prep_drift_mean_scale,
            "prep_drift_mean_used": prep_drift_mean,
            "prep_drift_tau": spec.prep_drift_tau,
            "ahs_prep_prior_mean": spec.ahs_prep_prior_mean,
            "ahs_prep_prior_tau": spec.ahs_prep_prior_tau,
            "prep_diag_prior_mean": spec.prep_diag_prior_mean,
            "prep_diag_prior_tau": spec.prep_diag_prior_tau,
        })

    fit_df = pd.DataFrame(fit_rows)
    ref = fit_df[fit_df["variant"] == "prep_only_drift"][["unit_id", "LX", "LR", "final_loss"]].rename(columns={
        "LX": "reference_LX",
        "LR": "reference_LR",
        "final_loss": "reference_final_loss",
    })
    fit_df = fit_df.merge(ref, on="unit_id", how="left")
    fit_df["relative_LX_vs_prep_only_drift"] = fit_df["LX"] / fit_df["reference_LX"]
    fit_df["relative_LR_vs_prep_only_drift"] = fit_df["LR"] / fit_df["reference_LR"]
    fit_df["relative_final_loss_vs_prep_only_drift"] = fit_df["final_loss"] / fit_df["reference_final_loss"]

    fit_path = output_dir / "prep_only_drift_fit_summary.csv"
    scenario_path = output_dir / "prep_only_drift_scenario_summary.csv"
    variant_path = output_dir / "prep_only_drift_variant_specs.csv"
    fit_df.to_csv(fit_path, index=False)
    pd.DataFrame(scenario_rows).to_csv(scenario_path, index=False)
    pd.DataFrame(variant_rows).to_csv(variant_path, index=False)
    return {"fit_summary": fit_path, "scenario_summary": scenario_path, "variant_specs": variant_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PrEP-only drift / masked-suppression sensitivity")
    parser.add_argument("--data-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx"))
    parser.add_argument("--variable-config-path", type=Path, default=Path("config/sem_variable_config_v3.csv"))
    parser.add_argument("--structural-prior-path", type=Path, default=Path("config/sem_structural_priors_v3.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/prep_only_drift_sensitivity"))
    parser.add_argument("--states", type=str, default=",".join(DEFAULT_STATES))
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS.keys()))
    parser.add_argument("--target-end-year", type=int, default=2036)
    parser.add_argument("--intervention-duration-steps", type=int, default=3)
    args = parser.parse_args()
    paths = run(args)
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
