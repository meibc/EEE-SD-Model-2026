#!/usr/bin/env python3
from __future__ import annotations

"""Paired final-v9 ablation of information sources in the state SEM.

The script uses identical AMIS Monte Carlo input draws for every specification,
fits only the national/division hierarchy plus the states passing the explicit
effective-N screen, and propagates two core stigma scenarios to 2036 incidence.

Specifications
--------------
primary
    Final v9 model.
trajectory_only
    Covariance-reconstruction weight set to zero.
covariance_moderate
    Covariance-reconstruction weight increased from 0.01 to 0.10.
covariance_balanced
    Covariance-reconstruction weight increased from 0.01 to 0.40, selected
    to make its weighted contribution approximately comparable to trajectory
    fit at the primary solution.
neutral_prior
    Off-diagonal structural-prior means set to zero while retaining the same
    structural zeros, hard signs, and prior standard deviations.
weak_hierarchy
    National-to-division and division-to-state shrinkage weights halved.
strong_hierarchy
    National-to-division and division-to-state shrinkage weights doubled.
no_hierarchy
    Geographic shrinkage weights set to zero at both hierarchy transitions;
    national, division, and state matrices are therefore fitted without
    geographic borrowing while retaining all other model components.
regional_covariance
    State fits use weighted Census-region covariance targets; national and
    division fits retain the primary pooled covariance target.
sign_only
    Off-diagonal magnitude priors are removed while retaining structural
    zeros, hard signs, diagonal persistence priors, drift, hierarchy, and the
    stability penalty.
common_weak
    Every permitted off-diagonal pathway receives the same zero-centered weak
    ridge penalty (tau = 0.70), removing evidence-tier-specific centers and
    scales while retaining the same structural signs and zeros.
half_prep_drift
    The PrEP secular-drift prior mean is reduced from 0.468 to 0.234
    centered-logit units per two-year transition; all other components are
    unchanged.
zero_prep_drift
    The PrEP secular-drift prior mean is set to zero while retaining its
    primary regularization scale; all other components are unchanged.
"""

import argparse
import contextlib
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from config.interventions import SCENARIO_CODEBOOK  # noqa: E402
from config.optimization import OptimConfig  # noqa: E402
from config.sem import SEMConfig  # noqa: E402
from config.shrinkage import ShrinkageConfig  # noqa: E402
from input.standardized_runtime import (  # noqa: E402
    StandardizedBundle,
    StandardizedCDCParamsLoader,
    StandardizedSEMParamsLoader,
)
from models.sbm.estimation.jacobian import JacobianEstimator  # noqa: E402
from models.sbm.estimation.shrinkage import ShrinkageCalculator  # noqa: E402
from models.sbm.estimation.transforms import FeatureTransformer  # noqa: E402
from models.shared.alignment import build_model_years  # noqa: E402
from pipeline.joint_simulation import run_uncertainty  # noqa: E402


VARIANTS = (
    "primary",
    "trajectory_only",
    "covariance_moderate",
    "covariance_balanced",
    "neutral_prior",
    "weak_hierarchy",
    "strong_hierarchy",
    "no_hierarchy",
    "regional_covariance",
)
ADDITIONAL_PRIOR_VARIANTS = ("sign_only", "common_weak")
DRIFT_VARIANTS = ("half_prep_drift", "zero_prep_drift")
DRIFT_FLEXIBILITY_VARIANTS = ("flexible_prep_drift", "very_flexible_prep_drift")
TOPOLOGY_DIAGNOSTIC_VARIANTS = ("expanded_stigma_to_prep",)
REFERENCE_VARIANTS = ("empirical_p05_references", "empirical_p05_stigma_references")
KNOWN_VARIANTS = (VARIANTS + ADDITIONAL_PRIOR_VARIANTS + DRIFT_VARIANTS
                  + DRIFT_FLEXIBILITY_VARIANTS + TOPOLOGY_DIAGNOSTIC_VARIANTS
                  + REFERENCE_VARIANTS)
SCENARIOS = ("s1_reduce_ahs", "s7_combined_stigma_package")


def _sem_config(data_path: Path) -> SEMConfig:
    return SEMConfig(
        data_path=data_path,
        variable_config_path=MODEL_ROOT / "config/sem_variable_config_v3.csv",
        structural_prior_path=MODEL_ROOT / "config/sem_structural_priors_v3.csv",
        diagonal_prior_path=MODEL_ROOT / "config/sem_diagonal_priors_v3.csv",
    )


def _load_selected_states(path: Path) -> list[str]:
    data = pd.read_csv(path)
    column = "state" if "state" in data.columns else "state_calc"
    states = data[column].astype(str).tolist()
    if not states:
        raise ValueError(f"No states found in {path}")
    return states


def _override_values(prepared, draw_values: np.ndarray, input_units: list[str]) -> None:
    draw_idx = {unit: i for i, unit in enumerate(input_units)}
    for unit in prepared.units:
        uid = str(unit.id)
        if uid not in draw_idx:
            raise ValueError(f"Prepared unit absent from input draws: {uid}")
        unit.amis_values = np.asarray(draw_values[:, :, draw_idx[uid]], dtype=float)


def _weighted_covariance(
    cov_long: pd.DataFrame,
    v_names: list[str],
    states: list[str],
) -> np.ndarray:
    index = {name: i for i, name in enumerate(v_names)}
    matrices: list[np.ndarray] = []
    weights: list[float] = []
    for state in states:
        sub = cov_long[cov_long["state"].astype(str).eq(state)]
        if sub.empty:
            continue
        mat = np.full((len(v_names), len(v_names)), np.nan, dtype=float)
        for row in sub.itertuples(index=False):
            source = str(row.source)
            target = str(row.target)
            if source in index and target in index:
                mat[index[source], index[target]] = float(row.covariance)
        if not np.isfinite(mat).all():
            continue
        matrices.append(0.5 * (mat + mat.T))
        weight = pd.to_numeric(sub["n_eff_pairwise"], errors="coerce").median()
        weights.append(float(weight) if np.isfinite(weight) and weight > 0 else 1.0)
    if not matrices:
        raise ValueError(f"No complete covariance matrices for states={states}")
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    return np.tensordot(w, np.asarray(matrices), axes=(0, 0))


def _regional_covariances(
    covariance_path: Path,
    geography_path: Path,
    v_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    cov_long = pd.read_csv(covariance_path)
    geography = pd.read_csv(geography_path)
    state_region = dict(
        zip(geography["state"].astype(str), geography["census_region"].astype(str))
    )
    targets = {}
    for region, group in geography.groupby("census_region"):
        targets[str(region)] = _weighted_covariance(
            cov_long,
            v_names,
            group["state"].astype(str).tolist(),
        )
    return targets, state_region


def _variant_configs(name: str) -> tuple[OptimConfig, ShrinkageConfig]:
    opt = OptimConfig()
    shrink = ShrinkageConfig()
    if name == "trajectory_only":
        opt.wR = 0.0
    elif name == "covariance_moderate":
        opt.wR = 0.10
    elif name == "covariance_balanced":
        opt.wR = 0.40
    elif name == "weak_hierarchy":
        shrink.wJ *= 0.5
        shrink.wJ_state *= 0.5
    elif name == "strong_hierarchy":
        shrink.wJ *= 2.0
        shrink.wJ_state *= 2.0
    elif name == "no_hierarchy":
        shrink.wJ = 0.0
        shrink.wJ_state = 0.0
    return opt, shrink


def _last_diagnostics(estimator: JacobianEstimator) -> dict:
    if not estimator.last_fit_history:
        return {}
    return estimator.last_fit_history[-1].get("diagnostics", {}) or {}


def fit_variant(
    variant: str,
    draw_indices: np.ndarray,
    selected_states: list[str],
    input_path: Path,
    data_path: Path,
    regional_targets: dict[str, np.ndarray],
    state_region: dict[str, str],
    output_path: Path,
) -> None:
    input_data = np.load(input_path, allow_pickle=True)
    all_draws = np.asarray(input_data["Xbars_samples"], dtype=float)
    draw_indices = np.asarray(draw_indices, dtype=int)
    draws = all_draws[draw_indices]
    n_draws = int(draw_indices.size)
    input_units = np.asarray(input_data["unit_order"]).astype(str).tolist()
    input_vars = np.asarray(input_data["v_names"]).astype(str).tolist()
    input_years = np.asarray(input_data["ts"], dtype=int)

    base = _sem_config(data_path)
    prepared = __import__("data.data_prep", fromlist=["DataPrep"]).DataPrep(base).prepare_inputs()
    keep = set(selected_states)
    prepared.units = [u for u in prepared.units if u.kind != "state" or str(u.id) in keep]
    unit_order = [str(u.id) for u in prepared.units]
    state_set = {str(u.id) for u in prepared.units if u.kind == "state"}
    missing_states = sorted(set(selected_states) - state_set)
    if missing_states:
        raise ValueError(f"Selected states missing from prepared data: {missing_states}")
    if input_vars != list(prepared.v_names) or not np.array_equal(input_years, prepared.ts):
        raise ValueError("Input-draw variables/years do not match final v9 model")

    reference_probs = base.reference_probs()
    if variant in REFERENCE_VARIANTS:
        # Fixed, construct-specific low empirical anchors calculated from the
        # production point estimates for the primary 17 states and four
        # observed periods. References are not recalculated within bootstrap
        # draws, so this changes the coordinate origin rather than injecting
        # draw-specific reference uncertainty.
        state_values = np.concatenate(
            [
                np.asarray(unit.values, dtype=float).reshape(len(input_vars), -1)
                for unit in prepared.units
                if unit.kind == "state" and str(unit.id) in set(selected_states)
            ],
            axis=1,
        )
        empirical_p05 = np.clip(np.quantile(state_values, 0.05, axis=1), 1e-3, 1 - 1e-3)
        if variant == "empirical_p05_references":
            reference_probs = empirical_p05
        else:
            reference_probs = np.asarray(reference_probs, dtype=float).copy()
            for variable in ("stigma_ahs", "stigma_gss", "stigma_family"):
                k = input_vars.index(variable)
                reference_probs[k] = empirical_p05[k]
    sign, prior_mean, prior_tau = base.structural_prior_arrays()
    M = np.asarray(prepared.M, dtype=int).copy()
    drift_enabled, drift_mean, drift_tau = base.drift_prior_arrays()
    if variant == "neutral_prior":
        prior_mean = np.asarray(prior_mean, dtype=float).copy()
        off_diagonal = ~np.eye(prior_mean.shape[0], dtype=bool)
        prior_mean[off_diagonal & np.isfinite(prior_mean)] = 0.0
    elif variant == "sign_only":
        prior_mean = np.asarray(prior_mean, dtype=float).copy()
        prior_tau = np.asarray(prior_tau, dtype=float).copy()
        off_diagonal = ~np.eye(prior_mean.shape[0], dtype=bool)
        prior_mean[off_diagonal] = np.nan
        prior_tau[off_diagonal] = np.nan
    elif variant == "common_weak":
        prior_mean = np.asarray(prior_mean, dtype=float).copy()
        prior_tau = np.asarray(prior_tau, dtype=float).copy()
        off_diagonal_allowed = (~np.eye(prior_mean.shape[0], dtype=bool)) & (sign != 0)
        prior_mean[off_diagonal_allowed] = 0.0
        prior_tau[off_diagonal_allowed] = 0.70
    elif variant == "expanded_stigma_to_prep":
        sign = np.asarray(sign, dtype=int).copy()
        prior_mean = np.asarray(prior_mean, dtype=float).copy()
        prior_tau = np.asarray(prior_tau, dtype=float).copy()
        prep_idx = input_vars.index("prep_used")
        for source in ("stigma_gss", "stigma_family"):
            source_idx = input_vars.index(source)
            M[prep_idx, source_idx] = 1
            sign[prep_idx, source_idx] = -1
            prior_mean[prep_idx, source_idx] = -0.04
            prior_tau[prep_idx, source_idx] = 0.70
    if variant in DRIFT_VARIANTS:
        drift_mean = np.asarray(drift_mean, dtype=float).copy()
        prep_idx = input_vars.index("prep_used")
        if not bool(np.asarray(drift_enabled)[prep_idx]):
            raise ValueError("PrEP drift is not enabled in the primary configuration")
        if variant == "half_prep_drift":
            drift_mean[prep_idx] *= 0.5
        else:
            drift_mean[prep_idx] = 0.0
    elif variant in DRIFT_FLEXIBILITY_VARIANTS:
        drift_tau = np.asarray(drift_tau, dtype=float).copy()
        prep_idx = input_vars.index("prep_used")
        if not bool(np.asarray(drift_enabled)[prep_idx]):
            raise ValueError("PrEP drift is not enabled in the primary configuration")
        # Retain the primary positive mean while allowing substantially more
        # geography-specific departure from it. Primary tau is 0.35.
        drift_tau[prep_idx] = 0.70 if variant == "flexible_prep_drift" else 1.40

    opt, shrink_config = _variant_configs(variant)
    transformer = FeatureTransformer(reference_probs=reference_probs)
    shrink_calc = ShrinkageCalculator(shrink_config)

    n_units = len(prepared.units)
    m = len(prepared.v_names)
    J_stack = np.full((n_draws, n_units, m, m), np.nan, dtype=np.float32)
    drift_stack = np.full((n_draws, n_units, m), np.nan, dtype=np.float32)
    success_stack = np.zeros((n_draws, n_units), dtype=bool)
    loss_rows: list[dict] = []
    t0 = time.time()

    for draw_idx in range(n_draws):
        _override_values(prepared, draws[draw_idx], input_units)
        us_J = None
        division_J: dict[int, np.ndarray] = {}
        u = np.zeros(m, dtype=float)

        for unit_idx, unit in enumerate(prepared.units):
            estimator = JacobianEstimator(opt)
            covariance = prepared.SigmaY
            if variant == "regional_covariance" and unit.kind == "state":
                region = state_region[str(unit.id)]
                covariance = regional_targets[region]
            Xbar, SigmaX = transformer.transform(unit.values, covariance)
            J_ref, wJ = shrink_calc.get_params(unit, us_J, division_J)
            J_seq = estimator.fit(
                Xbar=Xbar,
                SigmaX=SigmaX,
                M=M,
                sign_mask=sign,
                u=u,
                J_ref=J_ref,
                wJ=wJ,
                structural_prior_mean=prior_mean,
                structural_prior_tau=prior_tau,
                drift_enabled=drift_enabled,
                drift_prior_mean=drift_mean,
                drift_prior_tau=drift_tau,
            )
            J = np.asarray(J_seq[:, :, -1], dtype=float)
            drift_seq = estimator.last_drift_sequence
            drift = (
                np.zeros(m, dtype=float)
                if drift_seq is None
                else np.asarray(drift_seq[:, -1], dtype=float)
            )
            J_stack[draw_idx, unit_idx] = J
            drift_stack[draw_idx, unit_idx] = drift

            diag = _last_diagnostics(estimator)
            components = diag.get("final_components", {}) or {}
            success = bool(diag.get("success", False))
            success_stack[draw_idx, unit_idx] = success
            row = {
                "variant": variant,
                "draw": draw_idx,
                "unit_id": str(unit.id),
                "unit_kind": str(unit.kind),
                "success": success,
                "final_loss": float(diag.get("final_loss", np.nan)),
                "hierarchy_weight": np.nan if wJ is None else float(wJ),
            }
            for component in ("LX", "LR", "LJ", "L_prior_J", "L_prior_c", "L_stab"):
                row[component] = float(components.get(component, np.nan))
            loss_rows.append(row)

            if str(unit.id) in {"USA", "US"}:
                us_J = J
            elif unit.kind == "division":
                division_J[unit.census_div] = J

        elapsed = time.time() - t0
        print(
            f"{variant}: draw {draw_idx + 1}/{n_draws}; "
            f"success={success_stack[draw_idx].mean():.3f}; elapsed={elapsed:.1f}s",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        variant=np.array(variant),
        Jmeans_stack=J_stack,
        drift_stack=drift_stack,
        success_stack=success_stack,
        unit_order=np.asarray(unit_order),
        v_names=np.asarray(input_vars),
        ts=input_years,
        reference_probs=reference_probs,
        source_draw_indices=draw_indices,
        selected_states=np.asarray(selected_states),
    )
    pd.DataFrame(loss_rows).to_csv(output_path.with_suffix(".loss_components.csv"), index=False)


def _make_variant_standardized(
    base_path: Path,
    fit_path: Path,
    temp_path: Path,
) -> None:
    base = np.load(base_path, allow_pickle=True)
    fit = np.load(fit_path, allow_pickle=True)
    arrays = {key: base[key] for key in base.files}
    J = np.asarray(arrays["sem_J_samples"], dtype=float)
    drift = np.asarray(arrays["sem_drift_samples"], dtype=float)
    n = int(fit["Jmeans_stack"].shape[0])
    J = J[:n].copy()
    drift = drift[:n].copy()
    geo_ids = np.asarray(arrays["geo_ids"]).astype(str).tolist()
    fit_units = np.asarray(fit["unit_order"]).astype(str).tolist()
    for state in np.asarray(fit["selected_states"]).astype(str).tolist():
        J[:, geo_ids.index(state)] = fit["Jmeans_stack"][:, fit_units.index(state)]
        drift[:, geo_ids.index(state)] = fit["drift_stack"][:, fit_units.index(state)]
    arrays["sem_J_samples"] = J
    arrays["sem_drift_samples"] = drift
    arrays["sem_fit_J_last"] = np.nanmedian(J, axis=0)
    arrays["sem_fit_drift"] = np.nanmedian(drift, axis=0)
    arrays["sem_reference_probs"] = np.asarray(fit["reference_probs"], dtype=float)
    np.savez_compressed(temp_path, **arrays)


def _incidence_at(output, state: str, target_year: int) -> np.ndarray:
    years = np.asarray(output.years, dtype=int)
    idx = int(np.flatnonzero(years == target_year)[0])
    return np.asarray(
        [sample.cdc_output.incidence[idx] for sample in output.results[state].samples],
        dtype=float,
    )


def run_variant_scenarios(
    variant: str,
    fit_path: Path,
    base_standardized_path: Path,
    selected_states: list[str],
    n_samples: int,
    seed: int,
    target_year: int,
    output_path: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="eee_ablation_") as tmp:
        standardized_path = Path(tmp) / f"{variant}.npz"
        _make_variant_standardized(base_standardized_path, fit_path, standardized_path)
        bundle = StandardizedBundle(standardized_path)
        sem_loader = StandardizedSEMParamsLoader(bundle)
        cdc_loader = StandardizedCDCParamsLoader(bundle)
        units = bundle.build_units()
        model_years = build_model_years(cdc_loader.years, target_end_year=target_year)

        baseline = run_uncertainty(
            sem_loader,
            cdc_loader,
            units,
            unit_ids=selected_states,
            n_samples=n_samples,
            seed=seed,
            show_progress=False,
            model_years=model_years,
            state_intervention_codes=[],
            relationship_intervention_codes=[],
            intervention_duration_steps=3,
        )
        rows = []
        for scenario_code in SCENARIOS:
            spec = SCENARIO_CODEBOOK[scenario_code]
            scenario = run_uncertainty(
                sem_loader,
                cdc_loader,
                units,
                unit_ids=selected_states,
                n_samples=n_samples,
                seed=seed,
                show_progress=False,
                model_years=model_years,
                state_intervention_codes=spec.get("state_codes", []),
                relationship_intervention_codes=spec.get("relationship_codes", []),
                intervention_duration_steps=3,
            )
            for state in selected_states:
                b = _incidence_at(baseline, state, target_year)
                s = _incidence_at(scenario, state, target_year)
                pct = np.where(np.abs(b) > 1e-12, 100.0 * (s - b) / b, np.nan)
                for draw, (bv, sv, pv) in enumerate(zip(b, s, pct)):
                    rows.append(
                        {
                            "variant": variant,
                            "state": state,
                            "scenario": scenario_code,
                            "draw": draw,
                            "year": target_year,
                            "baseline_incidence": float(bv),
                            "scenario_incidence": float(sv),
                            "pct_change": float(pv),
                        }
                    )
        pd.DataFrame(rows).to_csv(output_path, index=False)


def classify_scenarios(
    scenario_draws: pd.DataFrame,
    response_cutoff: float = 10.0,
    ahs_share_cutoff: float = 0.80,
) -> pd.DataFrame:
    med = scenario_draws.groupby(["variant", "state", "scenario"])["pct_change"].median().unstack()
    med = med.reset_index()
    med["combined_reduction"] = -med["s7_combined_stigma_package"]
    med["ahs_reduction"] = -med["s1_reduce_ahs"]
    med["ahs_share"] = med["ahs_reduction"] / med["combined_reduction"]

    def category(row: pd.Series) -> str:
        if row["combined_reduction"] < response_cutoff:
            return "Limited aggregate responsiveness"
        if row["ahs_share"] >= ahs_share_cutoff:
            return "AHS-centered responsiveness"
        return "Broader stigma responsiveness"

    med["category"] = med.apply(category, axis=1)
    return med


def summarize_ablation(
    output_dir: Path,
    variants: list[str],
    selected_states: list[str],
) -> None:
    primary_fit = np.load(output_dir / "fits/primary.npz", allow_pickle=True)
    units = np.asarray(primary_fit["unit_order"]).astype(str).tolist()
    state_indices = [units.index(state) for state in selected_states]
    v_names = np.asarray(primary_fit["v_names"]).astype(str).tolist()
    sign, _, _ = _sem_config(Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx")).structural_prior_arrays()
    edge_mask = (sign != 0) & (~np.eye(len(v_names), dtype=bool))
    primary_J = np.asarray(primary_fit["Jmeans_stack"], dtype=float)[:, state_indices]
    primary_median = np.nanmedian(primary_J, axis=0)

    scenario_draws = pd.concat(
        [pd.read_csv(output_dir / f"scenarios/{variant}.csv") for variant in variants],
        ignore_index=True,
    )
    scenario_draws.to_csv(output_dir / "ablation_scenario_draws.csv", index=False)
    categories = classify_scenarios(scenario_draws)
    categories.to_csv(output_dir / "ablation_state_categories.csv", index=False)
    primary_categories = categories[categories["variant"].eq("primary")].set_index("state")

    rows = []
    for variant in variants:
        fit = np.load(output_dir / f"fits/{variant}.npz", allow_pickle=True)
        fit_units = np.asarray(fit["unit_order"]).astype(str).tolist()
        idx = [fit_units.index(state) for state in selected_states]
        J = np.asarray(fit["Jmeans_stack"], dtype=float)[:, idx]
        median_J = np.nanmedian(J, axis=0)
        a = primary_median[:, edge_mask].ravel()
        b = median_J[:, edge_mask].ravel()
        pathway_rank = float(spearmanr(a, b, nan_policy="omit").statistic)
        median_abs_change = float(np.nanmedian(np.abs(J[:, :, edge_mask] - primary_J[:, :, edge_mask])))

        cat = categories[categories["variant"].eq(variant)].set_index("state")
        common = primary_categories.index.intersection(cat.index)
        category_agreement = float(
            (primary_categories.loc[common, "category"] == cat.loc[common, "category"]).mean()
        )
        primary_response = primary_categories.loc[common, "combined_reduction"]
        variant_response = cat.loc[common, "combined_reduction"]
        response_rank = float(spearmanr(primary_response, variant_response).statistic)

        losses = pd.read_csv(output_dir / f"fits/{variant}.loss_components.csv")
        state_losses = losses[losses["unit_id"].isin(selected_states)]
        row = {
            "variant": variant,
            "pathway_rank_correlation": pathway_rank,
            "median_abs_pathway_change": median_abs_change,
            "combined_response_rank_correlation": response_rank,
            "category_agreement": category_agreement,
            "fit_success_rate": float(state_losses["success"].mean()),
        }
        for component in ("final_loss", "LX", "LR", "LJ", "L_prior_J", "L_prior_c", "L_stab"):
            row[f"median_{component}"] = float(state_losses[component].median())
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "ablation_summary.csv", index=False)

    category_wide = categories.pivot(index="state", columns="variant", values="category")
    category_wide.to_csv(output_dir / "ablation_category_crosswalk.csv")

    response_wide = categories.pivot(index="state", columns="variant", values="combined_reduction")
    response_wide.to_csv(output_dir / "ablation_combined_response_crosswalk.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument(
        "--draw-seed",
        type=int,
        default=20260921,
        help="Seed for sampling paired SEM input draws without replacement from the full draw set.",
    )
    parser.add_argument(
        "--scenario-samples",
        type=int,
        default=1000,
        help="Scenario resamples from the fitted SEM draw subset; larger than n-draws stabilizes medians.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-year", type=int, default=2036)
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    parser.add_argument(
        "--selected-states-path",
        type=Path,
        default=Path("output/final_state_screen_and_response_categories/primary_states.csv"),
    )
    parser.add_argument(
        "--input-draws-path",
        type=Path,
        default=MODEL_ROOT / "output/sem_input_draws_v5.npz",
    )
    parser.add_argument(
        "--base-standardized-path",
        type=Path,
        default=MODEL_ROOT / "output/standardized_input_v9.npz",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx"),
    )
    parser.add_argument(
        "--localized-workflow-dir",
        type=Path,
        default=Path("/Users/meibinchen/Documents/Codex/2026-08-13/d/outputs/localized_prior_covariance_workflow"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/final_v9_information_source_ablation_n100")
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-scenarios", action="store_true")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variants) - set(KNOWN_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    selected_states = _load_selected_states(args.selected_states_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fits").mkdir(exist_ok=True)
    (args.output_dir / "scenarios").mkdir(exist_ok=True)

    input_artifact = np.load(args.input_draws_path, allow_pickle=True)
    v_names = np.asarray(input_artifact["v_names"]).astype(str).tolist()
    n_available_draws = int(input_artifact["Xbars_samples"].shape[0])
    if args.n_draws > n_available_draws:
        raise ValueError(f"Requested {args.n_draws} draws but only {n_available_draws} are available")
    rng = np.random.default_rng(args.draw_seed)
    draw_indices = np.sort(
        rng.choice(n_available_draws, size=args.n_draws, replace=False).astype(int)
    )
    pd.DataFrame({"source_draw_index": draw_indices}).to_csv(
        args.output_dir / "paired_source_draw_indices.csv", index=False
    )
    regional_targets, state_region = _regional_covariances(
        args.localized_workflow_dir / "state_weighted_covariance_long.csv",
        args.localized_workflow_dir / "state_geography_mapping.csv",
        v_names,
    )

    for variant in variants:
        fit_path = args.output_dir / f"fits/{variant}.npz"
        if args.force or not fit_path.exists():
            fit_variant(
                variant=variant,
                draw_indices=draw_indices,
                selected_states=selected_states,
                input_path=args.input_draws_path,
                data_path=args.data_path,
                regional_targets=regional_targets,
                state_region=state_region,
                output_path=fit_path,
            )
        else:
            print(f"using existing fit: {fit_path}")

        if not args.skip_scenarios:
            scenario_path = args.output_dir / f"scenarios/{variant}.csv"
            if args.force or not scenario_path.exists():
                print(f"running scenarios for {variant}", flush=True)
                run_variant_scenarios(
                    variant=variant,
                    fit_path=fit_path,
                    base_standardized_path=args.base_standardized_path,
                    selected_states=selected_states,
                    n_samples=args.scenario_samples,
                    seed=args.seed,
                    target_year=args.target_year,
                    output_path=scenario_path,
                )
            else:
                print(f"using existing scenarios: {scenario_path}")

    if not args.skip_scenarios and "primary" in variants:
        summarize_ablation(args.output_dir, variants, selected_states)
        print(f"Wrote consolidated ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
