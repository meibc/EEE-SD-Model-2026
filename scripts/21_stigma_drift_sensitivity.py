#!/usr/bin/env python3
from __future__ import annotations

"""Focused deterministic sensitivity for secular drift in stigma measures.

This diagnostic retains the final model topology, signs, priors, covariance
term, PrEP/risk drift, and US -> division -> state shrinkage. It compares the
primary specification with weak, zero-centered drift for GSS alone and for GSS
plus family stigma. It is intentionally a fast point-estimate diagnostic, not
the final respondent-bootstrap uncertainty analysis.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from config.optimization import OptimConfig  # noqa: E402
from config.sem import SEMConfig  # noqa: E402
from config.shrinkage import ShrinkageConfig  # noqa: E402
from data.data_prep import DataPrep  # noqa: E402
from models.sbm.estimation.jacobian import JacobianEstimator  # noqa: E402
from models.sbm.estimation.shrinkage import ShrinkageCalculator  # noqa: E402
from models.sbm.estimation.transforms import FeatureTransformer  # noqa: E402


OUT = ROOT / "output/stigma_drift_sensitivity_deterministic"
STATE_FILE = ROOT / "output/final_state_screen_and_response_categories/primary_states.csv"
DRIFT_TAU = 0.35
VARIANTS = {
    "primary": {"added_drift": (), "prep_to_gss": False, "construct_refs": False},
    "gss_drift": {"added_drift": ("stigma_gss",), "prep_to_gss": False, "construct_refs": False},
    "prep_to_gss": {"added_drift": (), "prep_to_gss": True, "construct_refs": False},
    "gss_drift_prep_to_gss": {"added_drift": ("stigma_gss",), "prep_to_gss": True, "construct_refs": False},
    "gss_family_drift": {
        "added_drift": ("stigma_gss", "stigma_family"),
        "prep_to_gss": False,
        "construct_refs": False,
    },
    "construct_refs": {"added_drift": (), "prep_to_gss": False, "construct_refs": True},
    "construct_refs_gss_drift": {
        "added_drift": ("stigma_gss",),
        "prep_to_gss": False,
        "construct_refs": True,
    },
}

# Weak structural tier used elsewhere in the final model. The positive sign is
# fixed, but the center is close to zero and the scale is broad.
PREP_TO_GSS_PRIOR_MEAN = 0.04
PREP_TO_GSS_PRIOR_TAU = 0.70


def sem_config() -> SEMConfig:
    return SEMConfig(
        data_path=MODEL_ROOT / "Factor Analysis v3.xlsx",
        variable_config_path=MODEL_ROOT / "config/sem_variable_config_v3.csv",
        structural_prior_path=MODEL_ROOT / "config/sem_structural_priors_v3.csv",
        diagonal_prior_path=MODEL_ROOT / "config/sem_diagonal_priors_v3.csv",
    )


def reconstruct(J: np.ndarray, c: np.ndarray, x0: np.ndarray, n_time: int) -> np.ndarray:
    fitted = np.zeros((x0.size, n_time), dtype=float)
    fitted[:, 0] = x0
    for t in range(1, n_time):
        fitted[:, t] = J @ fitted[:, t - 1] + c
    return fitted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    states = pd.read_csv(STATE_FILE)["state"].astype(str).tolist()
    base = sem_config()
    prepared = DataPrep(base).prepare_inputs()
    keep = set(states)
    prepared.units = [u for u in prepared.units if u.kind != "state" or str(u.id) in keep]

    sign, prior_mean, prior_tau = base.structural_prior_arrays()
    drift_enabled_base, drift_mean_base, drift_tau_base = base.drift_prior_arrays()
    shrink_calc = ShrinkageCalculator(ShrinkageConfig())
    opt = OptimConfig()
    idx = {v: i for i, v in enumerate(prepared.v_names)}

    # Construct-specific low references: five percentage points below the
    # minimum value observed among the primary states and four study periods.
    construct_reference_probs = np.asarray(base.reference_probs(), dtype=float).copy()
    state_units = [u for u in prepared.units if u.kind == "state"]
    for variable in ("stigma_ahs", "stigma_gss", "stigma_family"):
        k = idx[variable]
        observed_minimum = min(float(np.min(u.values[k])) for u in state_units)
        construct_reference_probs[k] = np.clip(observed_minimum - 0.05, 0.01, 0.99)
    pd.DataFrame({
        "variable": prepared.v_names,
        "primary_reference": base.reference_probs(),
        "construct_specific_reference": construct_reference_probs,
    }).to_csv(OUT / "reference_probabilities.csv", index=False)

    fit_rows: list[dict] = []
    drift_rows: list[dict] = []
    pathway_rows: list[dict] = []

    for variant, specification in VARIANTS.items():
        print(f"Fitting {variant}...", flush=True)
        added_drift = specification["added_drift"]
        reference_probs = (
            construct_reference_probs
            if specification["construct_refs"]
            else np.asarray(base.reference_probs(), dtype=float)
        )
        transformer = FeatureTransformer(reference_probs)
        drift_enabled = np.asarray(drift_enabled_base, dtype=bool).copy()
        drift_mean = np.asarray(drift_mean_base, dtype=float).copy()
        drift_tau = np.asarray(drift_tau_base, dtype=float).copy()
        for variable in added_drift:
            k = idx[variable]
            drift_enabled[k] = True
            drift_mean[k] = 0.0
            drift_tau[k] = DRIFT_TAU

        variant_sign = np.asarray(sign, dtype=int).copy()
        variant_prior_mean = np.asarray(prior_mean, dtype=float).copy()
        variant_prior_tau = np.asarray(prior_tau, dtype=float).copy()
        if specification.get("gss_persistence") is not None:
            k = idx["stigma_gss"]
            variant_prior_mean[k, k] = float(specification["gss_persistence"])
        if specification["prep_to_gss"]:
            i, j = idx["stigma_gss"], idx["prep_used"]
            variant_sign[i, j] = 1
            variant_prior_mean[i, j] = PREP_TO_GSS_PRIOR_MEAN
            variant_prior_tau[i, j] = PREP_TO_GSS_PRIOR_TAU
        variant_M = np.eye(len(idx), dtype=int) + (variant_sign != 0).astype(int)

        us_J = None
        division_J: dict[int, np.ndarray] = {}
        for unit in prepared.units:
            estimator = JacobianEstimator(opt)
            X, SigmaX = transformer.transform(unit.values, prepared.SigmaY)
            J_ref, wJ = shrink_calc.get_params(unit, us_J, division_J)
            J_seq = estimator.fit(
                Xbar=X,
                SigmaX=SigmaX,
                M=variant_M,
                sign_mask=variant_sign,
                u=np.zeros(len(idx), dtype=float),
                J_ref=J_ref,
                wJ=wJ,
                structural_prior_mean=variant_prior_mean,
                structural_prior_tau=variant_prior_tau,
                drift_enabled=drift_enabled,
                drift_prior_mean=drift_mean,
                drift_prior_tau=drift_tau,
            )
            J = np.asarray(J_seq[:, :, -1], dtype=float)
            drift_seq = estimator.last_drift_sequence
            c = np.zeros(len(idx)) if drift_seq is None else np.asarray(drift_seq[:, -1], dtype=float)

            fitted_x = reconstruct(J, c, X[:, 0], X.shape[1])
            fitted_y = transformer.inverse_transform(fitted_x)
            residual_pp = 100.0 * (fitted_y - unit.values)

            if unit.kind == "state":
                for k, variable in enumerate(prepared.v_names):
                    post = residual_pp[k, 1:]
                    fit_rows.append({
                        "variant": variant,
                        "state": str(unit.id),
                        "variable": variable,
                        "mean_residual_pp": float(np.mean(post)),
                        "mae_pp": float(np.mean(np.abs(post))),
                        "rmse_pp": float(np.sqrt(np.mean(post**2))),
                        "fitted_persistence": float(J[k, k]),
                    })
                    drift_rows.append({
                        "variant": variant,
                        "state": str(unit.id),
                        "variable": variable,
                        "drift_centered_logit_per_2y": float(c[k]),
                    })
                for source in prepared.v_names:
                    for target in prepared.v_names:
                        i, j = idx[target], idx[source]
                        if i != j and variant_sign[i, j] != 0:
                            pathway_rows.append({
                                "variant": variant,
                                "state": str(unit.id),
                                "source": source,
                                "target": target,
                                "coefficient": float(J[i, j]),
                            })

            if str(unit.id) in {"US", "USA"}:
                us_J = J
            elif unit.kind == "division":
                division_J[unit.census_div] = J

    fits = pd.DataFrame(fit_rows)
    drifts = pd.DataFrame(drift_rows)
    paths = pd.DataFrame(pathway_rows)
    fits.to_csv(OUT / "state_fit_metrics.csv", index=False)
    drifts.to_csv(OUT / "state_drift_estimates.csv", index=False)
    paths.to_csv(OUT / "state_pathway_estimates.csv", index=False)

    focus = fits[fits.variable.isin(["stigma_gss", "stigma_family"])]
    summary = focus.groupby(["variant", "variable"], as_index=False).agg(
        mean_bias_pp=("mean_residual_pp", "mean"),
        median_state_mae_pp=("mae_pp", "median"),
        mean_state_mae_pp=("mae_pp", "mean"),
        mean_state_rmse_pp=("rmse_pp", "mean"),
    )
    primary = paths[paths.variant == "primary"][["state", "source", "target", "coefficient"]]
    comparisons = []
    for variant in (name for name in VARIANTS if name != "primary"):
        alt = paths[paths.variant == variant][["state", "source", "target", "coefficient"]]
        # Restrict stability comparison to pathways shared with the primary model;
        # the added PrEP -> GSS coefficient is summarized separately below.
        joined = primary.merge(alt, on=["state", "source", "target"], suffixes=("_primary", "_alternative"))
        comparisons.append({
            "variant": variant,
            "pearson_all_pathways": joined.coefficient_primary.corr(joined.coefficient_alternative),
            "median_absolute_coefficient_change": float(np.median(np.abs(joined.coefficient_alternative - joined.coefficient_primary))),
        })
    comparison = pd.DataFrame(comparisons)
    added_path = paths[(paths.source == "prep_used") & (paths.target == "stigma_gss")].copy()
    if added_path.empty:
        added_path_summary = pd.DataFrame(columns=[
            "variant", "mean_coefficient", "median_coefficient",
            "minimum_coefficient", "maximum_coefficient", "boundary_share",
        ])
    else:
        added_path_summary = added_path.groupby("variant", as_index=False).agg(
            mean_coefficient=("coefficient", "mean"),
            median_coefficient=("coefficient", "median"),
            minimum_coefficient=("coefficient", "min"),
            maximum_coefficient=("coefficient", "max"),
            boundary_share=("coefficient", lambda x: float(np.mean(np.asarray(x) <= 1e-8))),
        )
    summary.to_csv(OUT / "stigma_fit_summary.csv", index=False)
    comparison.to_csv(OUT / "pathway_stability_summary.csv", index=False)
    added_path.to_csv(OUT / "prep_to_gss_state_estimates.csv", index=False)
    added_path_summary.to_csv(OUT / "prep_to_gss_summary.csv", index=False)
    print("\nStigma reconstruction summary\n", summary.to_string(index=False))
    print("\nPathway stability summary\n", comparison.to_string(index=False))
    print("\nPrEP -> GSS summary\n", added_path_summary.to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
