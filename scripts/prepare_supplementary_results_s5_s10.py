from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/supplementary_results_data"
OUT.mkdir(parents=True, exist_ok=True)

ROBUST = ["CA", "FL", "GA", "IL", "IN", "MA", "MD", "MI", "NC", "NJ", "NY", "OH", "PA", "TN", "TX", "VA", "WA"]
LABEL = {
    "stigma_ahs": "Anticipated healthcare stigma",
    "stigma_gss": "General social stigma",
    "stigma_family": "Family stigma",
    "out_gid": "Disclosure to a healthcare provider",
    "seehcp": "Saw a healthcare provider",
    "risk_behavior": "HIV-related risk behavior",
    "prep_used": "PrEP use",
    "hivtest12": "HIV testing frequency",
}


def qsum(df, keys, value):
    def quant(x, p):
        return np.quantile(x.dropna(), p) if x.notna().any() else np.nan
    return (
        df.groupby(keys, sort=False)[value]
        .agg(
            p025=lambda x: quant(x, 0.025),
            p25=lambda x: quant(x, 0.25),
            p50="median",
            p75=lambda x: quant(x, 0.75),
            p975=lambda x: quant(x, 0.975),
        )
        .reset_index()
    )


# S5: complete permitted off-diagonal state pathways.
fit = np.load(OUT.parent / "amis_joint_bootstrap_direct_v2_n500/sem_fits_direct_bootstrap_n500.npz", allow_pickle=True)
J = fit["Jmeans_stack"]
units = list(fit["unit_order"].astype(str))
vars_ = list(fit["v_names"].astype(str))
priors = pd.read_csv(Path("/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/config/sem_structural_priors_v3.csv"))
rows = []
for state in ROBUST:
    ui = units.index(state)
    for _, p in priors.iterrows():
        ti, si = vars_.index(p["target"]), vars_.index(p["source"])
        x = J[:, ui, ti, si].astype(float)
        rows.append({
            "state": state,
            "source": LABEL.get(p["source"], p["source"]),
            "target": LABEL.get(p["target"], p["target"]),
            "pathway": f'{LABEL.get(p["source"], p["source"])} → {LABEL.get(p["target"], p["target"])}',
            "permitted_sign": "Positive" if float(p["sign"]) > 0 else "Negative",
            "evidence_tier": p["evidence_tier"],
            "p025": np.quantile(x, .025), "p25": np.quantile(x, .25),
            "p50": np.quantile(x, .50), "p75": np.quantile(x, .75),
            "p975": np.quantile(x, .975),
            "zero_boundary_share": np.mean(np.isclose(x, 0.0, atol=1e-10)),
        })
pd.DataFrame(rows).to_csv(OUT / "S5_state_pathway_estimates.csv", index=False)


# S6: incidence outcomes for all production scenarios and fixed-dose checks.
scen_dir = OUT.parent / "amis_joint_bootstrap_direct_v2_n500/sem_scenarios_direct_bootstrap_n500"
epi_draws = pd.read_csv(scen_dir / "scenario_endpoint_epi_draws.csv")
epi_draws = epi_draws[epi_draws.unit_id.isin(ROBUST)]
inc = epi_draws[epi_draws.variable.eq("incidence")].copy()
inc_sum = qsum(inc, ["unit_id", "scenario", "scenario_label", "year", "variable"], "pct_change")
inc_sum.insert(2, "scenario_type", "Proportional/pathway")

abs_draws = pd.read_csv(OUT.parent / "amis_joint_bootstrap_direct_v2_n500/absolute_stigma_reduction_direct_bootstrap_n500/absolute_scenario_incidence_draws.csv")
abs_draws = abs_draws[abs_draws.state.isin(ROBUST)].rename(columns={"state": "unit_id"})
abs_sum = qsum(abs_draws, ["unit_id", "scenario", "scenario_label", "year"], "pct_change")
abs_sum["variable"] = "incidence"
abs_sum.insert(2, "scenario_type", "Fixed-dose")
inc_all = pd.concat([inc_sum, abs_sum[inc_sum.columns]], ignore_index=True)
inc_all.rename(columns={"unit_id": "state"}).to_csv(OUT / "S6_scenario_incidence_outcomes.csv", index=False)


# S7: intermediate SEM and epidemiological outcomes for production scenarios.
sem_draws = pd.read_csv(scen_dir / "scenario_endpoint_sem_draws.csv")
sem_draws = sem_draws[sem_draws.unit_id.isin(ROBUST)]
sem_delta = qsum(sem_draws, ["unit_id", "scenario", "scenario_label", "year", "variable"], "delta")
sem_pct = qsum(sem_draws, ["unit_id", "scenario", "scenario_label", "year", "variable"], "pct_change")
sem_delta = sem_delta.rename(columns={c: f"delta_{c}" for c in ["p025", "p25", "p50", "p75", "p975"]})
sem_pct = sem_pct.rename(columns={c: f"pct_{c}" for c in ["p025", "p25", "p50", "p75", "p975"]})
intermediate = sem_delta.merge(sem_pct, on=["unit_id", "scenario", "scenario_label", "year", "variable"])

epi_mid = epi_draws[epi_draws.variable.ne("incidence")].copy()
epi_delta = qsum(epi_mid, ["unit_id", "scenario", "scenario_label", "year", "variable"], "delta")
epi_pct = qsum(epi_mid, ["unit_id", "scenario", "scenario_label", "year", "variable"], "pct_change")
epi_delta = epi_delta.rename(columns={c: f"delta_{c}" for c in ["p025", "p25", "p50", "p75", "p975"]})
epi_pct = epi_pct.rename(columns={c: f"pct_{c}" for c in ["p025", "p25", "p50", "p75", "p975"]})
epi_mid = epi_delta.merge(epi_pct, on=["unit_id", "scenario", "scenario_label", "year", "variable"])
intermediate = pd.concat([intermediate, epi_mid], ignore_index=True)
intermediate.rename(columns={"unit_id": "state"}).to_csv(OUT / "S7_intermediate_outcomes.csv", index=False)


# S8: complete national demographic pathways plus the California bridge estimates shown in the manuscript.
pop = pd.read_csv(OUT.parent / "population_joint_bootstrap_n500/fits_state_drift/population_J_draws_n500_state_drift.csv")
pop = pop[pop.population_id.ne("ca_hispanic_latino")].copy()
pop = pop.merge(priors[["source", "target", "sign", "evidence_tier"]], on=["source", "target"], how="inner")
pop_rows = []
for keys, g in pop.groupby(["population_id", "population_label", "source", "target", "sign", "evidence_tier"], sort=False):
    pid, plabel, source, target, sign, tier = keys
    x = g.J.astype(float).to_numpy()
    pop_rows.append({
        "analysis": "National demographic group", "population_id": pid, "population_label": plabel,
        "source": LABEL.get(source, source), "target": LABEL.get(target, target),
        "pathway": f"{LABEL.get(source, source)} → {LABEL.get(target, target)}",
        "permitted_sign": "Positive" if float(sign) > 0 else "Negative", "evidence_tier": tier,
        "p025": np.quantile(x, .025), "p25": np.quantile(x, .25), "p50": np.quantile(x, .5),
        "p75": np.quantile(x, .75), "p975": np.quantile(x, .975),
        "zero_boundary_share": np.mean(np.isclose(x, 0.0, atol=1e-10)),
    })
pop_out = pd.DataFrame(pop_rows)
bridge = pd.read_csv(OUT.parent / "results_refinement/final_figure_inputs/figure5_california_bridge_panel_data.csv")
bridge_rows = bridge.rename(columns={"group": "population_label", "p_nonzero": "nonzero_share"}).copy()
bridge_rows.insert(0, "analysis", "California bridge comparison")
bridge_rows.to_csv(OUT / "S8_california_bridge_estimates.csv", index=False)
pop_out.to_csv(OUT / "S8_national_demographic_pathways.csv", index=False)


# S9: dynamic-fit and epidemiological calibration diagnostics.
audit = pd.read_csv(OUT.parent / "amis_joint_bootstrap_direct_v2_n500/sem_fits_direct_bootstrap_n500.audit.csv")
audit = audit[audit.unit_id.isin(ROBUST)]
dyn = audit.groupby("unit_id").agg(
    n_refits=("success", "size"), success_rate=("success", "mean"),
    median_final_loss=("final_loss", "median"), p025_final_loss=("final_loss", lambda x: np.quantile(x, .025)),
    p975_final_loss=("final_loss", lambda x: np.quantile(x, .975)), median_drift_norm=("drift_norm", "median"),
    median_J_abs_mean=("J_abs_mean", "median"),
).reset_index().rename(columns={"unit_id": "state"})
dyn.to_csv(OUT / "S9_dynamic_fit_diagnostics.csv", index=False)

epi_diag = pd.read_csv(ROOT / "sem_sd_1225_refactor/notebooks/transition_bayes_v7_pilot_outputs/v7_all_states_full_diagnostics.csv")
epi_diag.to_csv(OUT / "S9_epidemiological_sampling_diagnostics.csv", index=False)
ppc = pd.read_csv(ROOT / "sem_sd_1225_refactor/notebooks/transition_bayes_v7_pilot_outputs/v7d_final_all_state_panels/v7d_all_states_annual_fit_with_incidence_ppc.csv")
ppc[ppc.state.isin(ROBUST)].to_csv(OUT / "S9_incidence_calibration_ppc.csv", index=False)
dxppc = pd.read_csv(ROOT / "sem_sd_1225_refactor/notebooks/transition_bayes_v7_pilot_outputs/v7d_final_all_state_panels/v7d_diagnosis_period_ppc.csv")
dxppc[dxppc.state.isin(ROBUST)].to_csv(OUT / "S9_diagnosis_calibration_ppc.csv", index=False)


# S10: concise core sensitivity summary.
sens_rows = []
def add_ablation(path, variants, family, draws=100):
    d = pd.read_csv(path)
    for variant in variants:
        r = d[d.variant.eq(variant)].iloc[0]
        sens_rows.append({
            "sensitivity_family": family, "specification": variant, "n_draws": draws,
            "pathway_rank_correlation": r.pathway_rank_correlation,
            "combined_response_rank_correlation": r.combined_response_rank_correlation,
            "median_absolute_pathway_change": r.median_abs_pathway_change,
            "category_agreement": r.category_agreement,
            "fit_success_rate": r.fit_success_rate,
        })
add_ablation(OUT.parent / "final_direct_bootstrap_core_ablation_n100/ablation_summary.csv", ["trajectory_only", "no_hierarchy", "half_prep_drift", "zero_prep_drift"], "Core specification")
add_ablation(OUT.parent / "final_direct_bootstrap_covariance_gradient_n100/ablation_summary.csv", ["covariance_moderate", "covariance_balanced"], "Covariance weight")
add_ablation(OUT.parent / "final_direct_bootstrap_prior_regularization_n100/ablation_summary.csv", ["sign_only", "common_weak"], "Pathway regularization")
pd.DataFrame(sens_rows).to_csv(OUT / "S10_model_specification_sensitivity.csv", index=False)

fixed = pd.read_csv(OUT.parent / "amis_joint_bootstrap_direct_v2_n500/absolute_stigma_reduction_direct_bootstrap_n500/comparison_metrics.csv")
fixed.to_csv(OUT / "S10_fixed_dose_sensitivity.csv", index=False)
threshold = pd.read_csv(OUT.parent / "amis_joint_bootstrap_direct_v2_n500/state_screen_and_response_categories/response_category_threshold_stability.csv")
threshold.to_csv(OUT / "S10_threshold_stability.csv", index=False)

print(OUT)
