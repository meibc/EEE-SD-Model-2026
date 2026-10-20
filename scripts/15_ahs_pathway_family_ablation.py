#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic AHS pathway-family ablation under the final-v9 SEM.

This is a diagnostic of incremental, model-conditioned AHS information. It is
not a causal test and does not use likelihood-ratio inference. Each variant is
refit through the complete US -> Census division -> state hierarchy.
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


OUT = ROOT / "output/ahs_pathway_family_ablation_deterministic"
STATE_FILE = ROOT / "output/final_state_screen_and_response_categories/primary_states.csv"

VARIANTS: dict[str, tuple[tuple[str, str], ...]] = {
    "full": (),
    "no_direct_ahs_prevention": (
        ("stigma_ahs", "prep_used"),
        ("stigma_ahs", "hivtest12"),
    ),
    "no_downstream_ahs": (
        ("stigma_ahs", "prep_used"),
        ("stigma_ahs", "hivtest12"),
        ("stigma_ahs", "out_gid"),
        ("stigma_ahs", "seehcp"),
    ),
}

FOCUS_STATES = ("GA", "NY", "TN", "VA")
FOCUS_TARGETS = ("prep_used", "hivtest12", "out_gid", "seehcp")


def sem_config() -> SEMConfig:
    return SEMConfig(
        data_path=MODEL_ROOT / "Factor Analysis v3.xlsx",
        variable_config_path=MODEL_ROOT / "config/sem_variable_config_v3.csv",
        structural_prior_path=MODEL_ROOT / "config/sem_structural_priors_v3.csv",
        diagonal_prior_path=MODEL_ROOT / "config/sem_diagonal_priors_v3.csv",
    )


def selected_states() -> list[str]:
    data = pd.read_csv(STATE_FILE)
    return data["state"].astype(str).tolist()


def last_diagnostics(estimator: JacobianEstimator) -> dict:
    if not estimator.last_fit_history:
        return {}
    return estimator.last_fit_history[-1].get("diagnostics", {}) or {}


def residual_metrics(X: np.ndarray, J: np.ndarray, drift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = X[:, 1:] - J @ X[:, :-1] - drift[:, None]
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    sse = np.sum(residual**2, axis=1)
    return rmse, sse


def fit_variant(name: str, removed: tuple[tuple[str, str], ...], states: list[str]):
    base = sem_config()
    prepared = DataPrep(base).prepare_inputs()
    keep = set(states)
    prepared.units = [u for u in prepared.units if u.kind != "state" or str(u.id) in keep]

    sign, prior_mean, prior_tau = base.structural_prior_arrays()
    sign = np.asarray(sign, dtype=int).copy()
    prior_mean = np.asarray(prior_mean, dtype=float).copy()
    prior_tau = np.asarray(prior_tau, dtype=float).copy()
    index = {v: i for i, v in enumerate(prepared.v_names)}
    for source, target in removed:
        i, j = index[target], index[source]
        sign[i, j] = 0
        prior_mean[i, j] = np.nan
        prior_tau[i, j] = np.nan
    M = np.eye(len(index), dtype=int) + (sign != 0).astype(int)

    drift_enabled, drift_mean, drift_tau = base.drift_prior_arrays()
    transformer = FeatureTransformer(base.reference_probs())
    shrink_calc = ShrinkageCalculator(ShrinkageConfig())
    opt = OptimConfig()

    us_J = None
    division_J: dict[int, np.ndarray] = {}
    fits: dict[str, dict] = {}
    rows: list[dict] = []
    component_rows: list[dict] = []
    u = np.zeros(len(index), dtype=float)

    for unit in prepared.units:
        estimator = JacobianEstimator(opt)
        X, SigmaX = transformer.transform(unit.values, prepared.SigmaY)
        J_ref, wJ = shrink_calc.get_params(unit, us_J, division_J)
        J_seq = estimator.fit(
            Xbar=X,
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
        drift = np.zeros(len(index)) if drift_seq is None else np.asarray(drift_seq[:, -1], dtype=float)
        diagnostics = last_diagnostics(estimator)
        components = diagnostics.get("final_components", {}) or {}
        rmse, sse = residual_metrics(X, J, drift)
        fits[str(unit.id)] = {"J": J, "drift": drift, "X": X, "rmse": rmse, "sse": sse}

        component_rows.append(
            {
                "variant": name,
                "unit_id": str(unit.id),
                "unit_kind": str(unit.kind),
                "success": bool(diagnostics.get("success", False)),
                "final_loss": float(diagnostics.get("final_loss", np.nan)),
                "hierarchy_weight": np.nan if wJ is None else float(wJ),
                **{key: float(components.get(key, np.nan)) for key in
                   ("LX", "LR", "LJ", "L_prior_J", "L_prior_c", "L_stab")},
            }
        )
        for k, variable in enumerate(prepared.v_names):
            rows.append(
                {
                    "variant": name,
                    "unit_id": str(unit.id),
                    "unit_kind": str(unit.kind),
                    "variable": variable,
                    "transition_rmse": float(rmse[k]),
                    "transition_sse": float(sse[k]),
                }
            )

        if str(unit.id) in {"US", "USA"}:
            us_J = J
        elif unit.kind == "division":
            division_J[unit.census_div] = J

    return fits, pd.DataFrame(rows), pd.DataFrame(component_rows), prepared.v_names, sign


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    states = selected_states()
    all_fits = {}
    metrics = []
    components = []
    signs = {}
    v_names = None

    for name, removed in VARIANTS.items():
        print(f"Fitting {name}...", flush=True)
        fits, metric, component, v_names, sign = fit_variant(name, removed, states)
        all_fits[name] = fits
        metrics.append(metric)
        components.append(component)
        signs[name] = sign

    metrics_df = pd.concat(metrics, ignore_index=True)
    components_df = pd.concat(components, ignore_index=True)
    metrics_df.to_csv(OUT / "transition_error_by_variable.csv", index=False)
    components_df.to_csv(OUT / "loss_components.csv", index=False)

    idx = {v: i for i, v in enumerate(v_names)}
    pathway_rows = []
    for variant, fits in all_fits.items():
        for state in states:
            J = fits[state]["J"]
            for source in v_names:
                for target in v_names:
                    i, j = idx[target], idx[source]
                    if i == j or signs[variant][i, j] == 0:
                        continue
                    pathway_rows.append(
                        {
                            "variant": variant,
                            "state": state,
                            "source": source,
                            "target": target,
                            "coefficient": float(J[i, j]),
                        }
                    )
    pd.DataFrame(pathway_rows).to_csv(OUT / "fitted_pathways.csv", index=False)

    primary = metrics_df[(metrics_df.variant == "full") & metrics_df.unit_id.isin(states)]
    primary = primary.set_index(["unit_id", "variable"])
    comparison_rows = []
    compensation_rows = []
    for variant in ("no_direct_ahs_prevention", "no_downstream_ahs"):
        current = metrics_df[(metrics_df.variant == variant) & metrics_df.unit_id.isin(states)]
        current = current.set_index(["unit_id", "variable"])
        joined = primary[["transition_rmse", "transition_sse"]].join(
            current[["transition_rmse", "transition_sse"]],
            lsuffix="_full",
            rsuffix="_ablated",
        ).reset_index()
        joined["variant"] = variant
        joined["delta_rmse"] = joined.transition_rmse_ablated - joined.transition_rmse_full
        joined["delta_sse"] = joined.transition_sse_ablated - joined.transition_sse_full
        joined["pct_sse_change"] = np.where(
            joined.transition_sse_full > 0,
            100 * joined.delta_sse / joined.transition_sse_full,
            np.nan,
        )
        comparison_rows.append(joined)

        for state in states:
            full_J = all_fits["full"][state]["J"]
            ablated_J = all_fits[variant][state]["J"]
            for source in v_names:
                for target in v_names:
                    i, j = idx[target], idx[source]
                    if i == j or signs[variant][i, j] == 0:
                        continue
                    compensation_rows.append(
                        {
                            "variant": variant,
                            "state": state,
                            "source": source,
                            "target": target,
                            "full_J": float(full_J[i, j]),
                            "ablated_J": float(ablated_J[i, j]),
                            "change": float(ablated_J[i, j] - full_J[i, j]),
                            "absolute_change": float(abs(ablated_J[i, j] - full_J[i, j])),
                        }
                    )

    comparison = pd.concat(comparison_rows, ignore_index=True)
    compensation = pd.DataFrame(compensation_rows)
    comparison.to_csv(OUT / "ablation_transition_error_comparison.csv", index=False)
    compensation.to_csv(OUT / "remaining_pathway_compensation.csv", index=False)

    state_summary = (
        comparison.groupby(["variant", "unit_id"], as_index=False)
        .agg(
            total_delta_sse=("delta_sse", "sum"),
            total_full_sse=("transition_sse_full", "sum"),
            mean_delta_rmse=("delta_rmse", "mean"),
        )
    )
    state_summary["total_pct_sse_change"] = np.where(
        state_summary.total_full_sse > 0,
        100 * state_summary.total_delta_sse / state_summary.total_full_sse,
        np.nan,
    )
    target_summary = (
        comparison[comparison.variable.isin(FOCUS_TARGETS)]
        .groupby(["variant", "unit_id"], as_index=False)
        .agg(
            prevention_delta_sse=("delta_sse", "sum"),
            prevention_full_sse=("transition_sse_full", "sum"),
            prevention_mean_delta_rmse=("delta_rmse", "mean"),
        )
    )
    target_summary["prevention_pct_sse_change"] = np.where(
        target_summary.prevention_full_sse > 0,
        100 * target_summary.prevention_delta_sse / target_summary.prevention_full_sse,
        np.nan,
    )
    state_summary = state_summary.merge(target_summary, on=["variant", "unit_id"], how="left")
    max_comp = (
        compensation.sort_values("absolute_change", ascending=False)
        .groupby(["variant", "state"], as_index=False)
        .first()
        .rename(columns={"state": "unit_id", "source": "largest_comp_source",
                         "target": "largest_comp_target", "change": "largest_comp_change",
                         "absolute_change": "largest_comp_absolute_change"})
    )
    state_summary = state_summary.merge(
        max_comp[["variant", "unit_id", "largest_comp_source", "largest_comp_target",
                  "largest_comp_change", "largest_comp_absolute_change"]],
        on=["variant", "unit_id"], how="left"
    )
    state_summary.to_csv(OUT / "state_ablation_summary.csv", index=False)

    focus = state_summary[state_summary.unit_id.isin(FOCUS_STATES)].copy()
    ranked = state_summary.sort_values(["variant", "prevention_pct_sse_change"], ascending=[True, False])
    lines = [
        "# Deterministic AHS pathway-family ablation",
        "",
        "This diagnostic refits the final-v9 hierarchy after removing AHS pathway families.",
        "Changes are in-sample transition-reconstruction diagnostics, not causal tests or likelihood-ratio tests.",
        "",
        "## Focus states",
        "",
        focus.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "## All states ranked by prevention-target SSE increase",
        "",
        ranked.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(f"Saved diagnostic outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
