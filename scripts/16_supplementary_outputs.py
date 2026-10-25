#!/usr/bin/env python3
from __future__ import annotations

"""Supplementary tables, data tables, and figures, collected in results/paper/supplement/.

Tables
  S2  National PrEP calibration by period and age (scripts/02).
  S3  State precision screen, effective sample sizes (scripts/07).
  S4  Model specification: references, drift, persistence and pathway priors,
      optimization weights, data-derived constants.
Data tables (one CSV each, also collected in supplementary_data.xlsx)
  D1  Pathway coefficients and drift by state (quantiles, share at zero).
  D2  AHS-PrEP coupling strength and channel decomposition (scripts/13).
  D3  Incidence outcomes, all scenarios, 2030/2032/2036 and cumulative.
  D4  Intermediate outcomes (eight modeled measures), all scenarios.
  D5  Diagnostics: optimizer convergence, loss components, spectral radius,
      one-step-ahead fit, epidemic-model calibration.
  D6  Sensitivity analyses: coupling and incidence rank agreement by
      specification, fixed-dose scenarios, projection starting point.
  D7  Demographic group changes and cross-sectional associations.
Figures
  S1  One-step-ahead fit: observed vs fitted, eight measures.
  S2  Epidemic-model calibration: annual incidence, primary states.
  S3  Rank agreement of state results across sensitivity specifications.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.optimization import OptimConfig  # noqa: E402
RESULTS = ROOT / "results"

VARIABLE_LABELS = {
    "stigma_ahs": "Anticipated healthcare stigma", "stigma_gss": "General social stigma",
    "stigma_family": "Family stigma", "out_gid": "Disclosure to provider", "seehcp": "Saw healthcare provider",
    "risk_behavior": "Risk behavior", "prep_used": "PrEP use", "hivtest12": "HIV testing",
}
SPEC_LABELS = {
    "sign_only": "Sign constraints only", "common_weak": "Common weak prior", "trajectory_only": "Trajectory fit only",
    "covariance_moderate": "Covariance weight 0.10", "covariance_balanced": "Covariance weight 0.40",
    "no_hierarchy": "No hierarchy", "half_prep_drift": "Half PrEP drift", "zero_prep_drift": "Zero PrEP drift",
    "stability_w10": "Stability weight 10",
}
EPS = 1e-2  # centered-logit clipping used by the fit (models/sbm/estimation/transforms.py)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# --- Tables ---------------------------------------------------------------

def table_s2(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    return pd.DataFrame({
        "Period": t.year_group, "Age group": t.age_group, "PrEP-eligible N": t.eligible_n,
        "Unweighted PrEP, %": (100 * t.unweighted_prep).round(1),
        "Weighted PrEP, %": (100 * t.weighted_prep).round(1),
        "Calibration target, %": (100 * t.calibration_target).round(1),
        "Weighted - unweighted, pp": t.weighted_minus_unweighted_pp.round(1),
    })


def table_s3(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path).sort_values("median_cell_neff", ascending=False)
    yes = {True: "Yes", False: "No"}
    return pd.DataFrame({
        "State": t.state, "Complete cells (of 32)": t.n_complete_cells,
        "Minimum n_eff": t.min_cell_neff.round(0), "25th percentile n_eff": t.q25_cell_neff.round(0),
        "Median n_eff": t.median_cell_neff.round(0), "Primary screen": t.primary_precision_screen.map(yes),
        "Median >= 55": t.median55_precision_screen.map(yes), "Strict screen": t.strict_precision_screen.map(yes),
    })


def table_s4(config_dir: Path, constants: Path) -> pd.DataFrame:
    rows = []
    var = pd.read_csv(config_dir / "sem_variable_config_v3.csv").set_index("variable")
    diag = pd.read_csv(config_dir / "sem_diagonal_priors_v3.csv").set_index("variable")
    for v, r in var.iterrows():
        rows.append(("Centered-logit reference (probability)", VARIABLE_LABELS[v], f"{r.reference_prob:.4f}",
                     "5th percentile of weighted point estimates, 17 primary states x 4 periods"))
    for v, r in var[var.drift_enabled == 1].iterrows():
        rows.append(("Drift prior (mean, tau)", VARIABLE_LABELS[v], f"{r.drift_prior_mean:.3f}, {r.drift_prior_tau:.2f}",
                     r.drift_note))
    for v, r in diag.iterrows():
        rows.append(("Persistence prior (mean, tau)", VARIABLE_LABELS[v], f"{r.prior_mean:.2f}, {r.prior_tau:.2f}",
                     "Diagonal regularization"))
    pri = pd.read_csv(config_dir / "sem_structural_priors_v3.csv")
    for _, r in pri.iterrows():
        rows.append((f"Pathway prior, {r.evidence_tier} tier (mean, tau)",
                     f"{VARIABLE_LABELS[r.source]} -> {VARIABLE_LABELS[r.target]}",
                     f"{r.prior_mean:+.2f}, {r.prior_tau:.1f}", f"Sign constraint {int(r.sign):+d}"))
    opt = OptimConfig()
    for name, value, note in [
        ("wX", opt.wX, "Trajectory-fit weight"), ("wR", opt.wR, "Covariance-fit weight"),
        ("w_stab", opt.w_stab, "Soft spectral-radius penalty, threshold 0.999"),
        ("sigma_eta_scale", opt.sigma_eta_scale, "Innovation SD in the covariance target, as a fraction of each measure's SD"),
        ("Coefficient bounds", opt.bounds, "Off-diagonal pathway bounds"),
        ("Drift bounds", opt.drift_bounds, "Absolute bound on drift terms"),
    ]:
        rows.append(("Optimization", name, str(value), note))
    for _, r in pd.read_csv(constants).iterrows():
        se = "" if pd.isna(r.se) else f" (SE {r.se:.3f})"
        rows.append(("Data-derived constant", r.constant, f"{r.value:.3f}{se}",
                     f"{r.definition}; {int(r.n_states)} states, {int(r.n_obs)} observations"))
    return pd.DataFrame(rows, columns=["Component", "Item", "Value", "Note"])


# --- Data tables ------------------------------------------------------------

def d1_pathways(fit: dict, priors: Path) -> pd.DataFrame:
    v, units, states = fit["v_names"], fit["unit_order"], fit["selected_states"]
    pri = pd.read_csv(priors)
    entries = [(s, t, r.evidence_tier) for s, t, r in zip(pri.source, pri.target, pri.itertuples())]
    entries += [(x, x, "persistence") for x in v]
    rows = []
    for state in states:
        u = units.index(state)
        for source, target, tier in entries:
            vals = fit["J"][:, u, v.index(target), v.index(source)]
            q = np.quantile(vals, [0.025, 0.25, 0.5, 0.75, 0.975])
            rows.append({"state": state, "term": "pathway", "source": source, "target": target, "prior_tier": tier,
                         **dict(zip(["p025", "p25", "p50", "p75", "p975"], q)),
                         "share_at_zero": float(np.mean(vals == 0))})
        for x in ("prep_used", "risk_behavior"):
            vals = fit["drift"][:, u, v.index(x)]
            q = np.quantile(vals, [0.025, 0.25, 0.5, 0.75, 0.975])
            rows.append({"state": state, "term": "drift", "source": "", "target": x, "prior_tier": "drift",
                         **dict(zip(["p025", "p25", "p50", "p75", "p975"], q)),
                         "share_at_zero": float(np.mean(vals == 0))})
    return pd.DataFrame(rows)


def d5_diagnostics(fit: dict, loss_path: Path, std: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    loss = pd.read_csv(loss_path)
    conv = loss.groupby(["unit_kind", "unit_id"]).agg(
        refits=("draw", "size"), converged=("success", "sum"),
        final_loss_median=("final_loss", "median"), LX_median=("LX", "median"), LR_median=("LR", "median"),
        LJ_median=("LJ", "median"), L_prior_J_median=("L_prior_J", "median"), L_stab_median=("L_stab", "median"),
    ).reset_index()
    v, units = fit["v_names"], fit["unit_order"]
    radius = np.max(np.abs(np.linalg.eigvals(fit["J"])), axis=-1)  # draws x units
    conv["spectral_radius_median"] = [np.median(radius[:, units.index(u)]) for u in conv.unit_id]
    conv["spectral_radius_max"] = [np.max(radius[:, units.index(u)]) for u in conv.unit_id]

    # One-step-ahead fit: each draw's fitted system applied to that draw's observed values.
    ref = logit(std["sem_reference_probs"])[None, None, :, None]
    obs = std["sem_obs_samples"]  # draws x states x vars x periods
    X = logit(obs) - ref
    Xhat = np.einsum("dsij,dsjt->dsit", std["sem_J_samples"], X[..., :-1]) + std["sem_drift_samples"][..., None]
    phat = 1 / (1 + np.exp(-(Xhat + ref)))
    years = std["sem_obs_years"].astype(int)
    rows = []
    for s, state in enumerate(std["selected_states"]):
        for i, var in enumerate(std["sem_v_names"]):
            for t in range(1, len(years)):
                q = np.quantile(phat[:, s, i, t - 1], [0.025, 0.5, 0.975])
                o = std["sem_obs"][s, i, t]
                rows.append({"state": state, "variable": var, "year": years[t], "observed": o,
                             "fitted_p025": q[0], "fitted_p50": q[1], "fitted_p975": q[2],
                             "covered": bool(q[0] <= o <= q[2])})
    return conv, pd.DataFrame(rows)


def d6_sensitivity(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = pd.read_csv(results / "07_structural/state_coupling_specification_summary.csv")
    draws = []
    for f in sorted((results / "05_sensitivity/model_specification/scenarios").glob("*.csv")):
        d = pd.read_csv(f)
        draws.append(d[d.scenario.eq("s1_reduce_ahs") & d.year.eq(2036)])
    draws = pd.concat(draws)
    med = draws.groupby(["variant", "state"])["pct_change"].median().unstack("variant")
    spec["rank_rho_incidence_2036"] = [spearmanr(med["primary"], med[s]).statistic for s in spec.specification]
    spec["median_incidence_2036_pct"] = [med[s].median() for s in spec.specification]

    def cumulative(path: Path) -> pd.Series:
        e = pd.read_csv(path)
        e = e[e.scenario.eq("s1_reduce_ahs") & e.year.eq(2036) & e.variable.eq("incidence_cumulative")]
        return e.set_index("unit_id").pct_change_p50

    prim = cumulative(results / "04_scenarios/primary/scenario_endpoint_epi.csv")
    recon = cumulative(results / "05_sensitivity/start_2018_reconstruction/scenario_endpoint_epi.csv")
    start = pd.DataFrame({"state": prim.index, "cumulative_pct_primary": prim.values,
                          "cumulative_pct_2018_reconstruction": recon.reindex(prim.index).values})
    start["rank_rho"] = spearmanr(start.cumulative_pct_primary, start.cumulative_pct_2018_reconstruction).statistic
    return spec, start


def d7_demographic(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    change = pd.read_csv(results / "08_demographic/group_change_2017_2024.csv")
    assoc = pd.read_csv(results / "08_demographic/group_cross_sectional_associations.csv")
    # HIV testing is coded min(tests in 2 years, 4) / 4, so a unit change is four tests; report per test.
    per_test = assoc.relationship.str.startswith("hivtest12 ->")
    for col in ("odds_ratio", "conf_low_or", "conf_high_or"):
        assoc.loc[per_test, col] = assoc.loc[per_test, col] ** 0.25
    assoc["scale"] = np.where(per_test, "per additional test (of up to 4 in 2 years)", "per unit of predictor")
    return change, assoc


# --- Figures -----------------------------------------------------------------

def figure_s1(onestep: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for ax, (var, label) in zip(axes.flat, VARIABLE_LABELS.items()):
        d = onestep[onestep.variable == var]
        ax.errorbar(100 * d.observed, 100 * d.fitted_p50,
                    yerr=[100 * (d.fitted_p50 - d.fitted_p025), 100 * (d.fitted_p975 - d.fitted_p50)],
                    fmt="o", ms=3.5, color=BLUE, ecolor=BLUE, elinewidth=0.6, alpha=0.7)
        lo = 100 * min(d.observed.min(), d.fitted_p025.min())
        hi = 100 * max(d.observed.max(), d.fitted_p975.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
        mae = 100 * np.mean(np.abs(d.observed - d.fitted_p50))
        ax.set_title(f"{label}\nMAE {mae:.1f} pp, coverage {100 * d.covered.mean():.0f}%", loc="left", fontsize=9, color=INK)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Observed, %", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fitted one step ahead, %", fontsize=9)
    fig.text(0.01, -0.01, "Each point is one state-period (2020, 2022, 2024; 17 states). Fitted values apply each refit's "
             "system to that replicate's previous-period values; bars are 95% intervals across 500 refits.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, out, "figS1_one_step_fit")


def figure_s2(annual: pd.DataFrame, states: list[str], out: Path) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.5), sharex=True)
    for ax, state in zip(axes.flat, states):
        d = annual[annual.state == state].sort_values("year")
        ax.fill_between(d.year, d.inc_ppc_p05, d.inc_ppc_p95, color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(d.year, d.inc_median, color=BLUE, lw=2)
        ax.plot(d.year, d.inc_observed, "o", color=ORANGE, ms=4)
        ax.set_title(state, loc="left", fontsize=9, color=INK)
        ax.set_ylim(bottom=0)
        style(ax)
    for ax in axes.flat[len(states):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=BLUE, lw=2, label="Model median"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.15, label="90% posterior predictive interval"),
               plt.Line2D([], [], color=ORANGE, marker="o", ls="", label="CDC estimated incidence")]
    axes.flat[-1].legend(handles=handles, frameon=False, fontsize=8, loc="center")
    fig.supylabel("Annual new infections among MSM", fontsize=9)
    fig.tight_layout()
    save(fig, out, "figS2_epi_calibration")


def figure_s3(spec: pd.DataFrame, start_rho: float, fixed: pd.DataFrame, out: Path) -> None:
    metrics = {"rank_rho_total": "Coupling\nstrength", "rank_rho_direct": "Direct\nroute",
               "rank_rho_engagement": "Engagement\nroutes", "rank_rho_incidence_2036": "Incidence\nchange 2036"}
    grid = spec.set_index("specification")[list(metrics)]
    grid.index = [SPEC_LABELS[s] for s in grid.index]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid.iat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=SURFACE if val > 0.6 else INK)
    ax.set_xticks(range(len(metrics)), metrics.values(), fontsize=9)
    ax.set_yticks(range(len(grid)), grid.index, fontsize=9)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman rank correlation with primary (17 states)", fontsize=9)
    ahs = fixed[fixed.family.eq("ahs")]
    note = (f"Projection from 2018 reconstruction: cumulative incidence rank rho {start_rho:.2f}. "
            f"Fixed-dose AHS scenarios vs relative: rho {ahs.spearman_rank_correlation.min():.2f}-"
            f"{ahs.spearman_rank_correlation.max():.2f}.")
    fig.text(0.01, -0.02, note, fontsize=8, color=MUTED)
    save(fig, out, "figS3_sensitivity_rank_agreement")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--epi-dir", type=Path, default=ROOT / "inputs/epi_v7d")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "paper/supplement")
    args = parser.parse_args()
    res, out = args.results, args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    npz = np.load(res / "03_fits/primary/fits/primary.npz", allow_pickle=True)
    fit = {"J": npz["Jmeans_stack"], "drift": npz["drift_stack"], "v_names": npz["v_names"].astype(str).tolist(),
           "unit_order": npz["unit_order"].astype(str).tolist(), "selected_states": npz["selected_states"].astype(str).tolist()}
    s = np.load(res / "03_fits/primary/standardized_research.npz", allow_pickle=True)
    std = {k: s[k] for k in ("sem_reference_probs", "sem_obs_samples", "sem_J_samples", "sem_drift_samples",
                             "sem_obs_years", "sem_obs", "sem_v_names")}
    std["sem_v_names"] = std["sem_v_names"].astype(str)
    std["selected_states"] = fit["selected_states"]

    tables = {
        "S2_prep_calibration": table_s2(res / "01_weights/national_prep_calibration_table.csv"),
        "S3_state_screen": table_s3(res / "02_inputs/state_screen/state_effective_n_screen.csv"),
        "S4_specification": table_s4(ROOT / "config", res / "02_inputs/reported_constants.csv"),
    }
    conv, onestep = d5_diagnostics(fit, res / "03_fits/primary/fits/primary.loss_components.csv", std)
    spec, start = d6_sensitivity(res)
    change, assoc = d7_demographic(res)
    epi_metrics = pd.read_csv(args.epi_dir / "v7d_fit_metrics.csv")
    epi_mcmc = pd.read_csv(args.epi_dir / "v7_all_states_full_diagnostics.csv")
    fixed = pd.read_csv(res / "05_sensitivity/fixed_dose/comparison_metrics.csv")
    data = {
        "D1_pathways": d1_pathways(fit, ROOT / "config/sem_structural_priors_v3.csv"),
        "D2_coupling": pd.read_csv(res / "07_structural/state_coupling_summary.csv"),
        "D3_incidence": pd.read_csv(res / "04_scenarios/primary/scenario_endpoint_epi.csv"),
        "D4_intermediate": pd.read_csv(res / "04_scenarios/primary/scenario_endpoint_sem.csv"),
        "D5a_convergence": conv,
        "D5b_one_step_fit": onestep,
        "D5c_epi_fit_metrics": epi_metrics,
        "D5d_epi_mcmc": epi_mcmc,
        "D5e_epi_diagnosis_ppc": pd.read_csv(args.epi_dir / "v7d_diagnosis_period_ppc.csv"),
        "D6a_specifications": spec,
        "D6b_fixed_dose": fixed,
        "D6c_projection_start": start,
        "D7a_group_change": change,
        "D7b_group_associations": assoc,
    }
    with pd.ExcelWriter(out / "supplementary_data.xlsx") as xl:
        for name, df in {**tables, **data}.items():
            df.to_csv(out / f"{name}.csv", index=False)
            df.to_excel(xl, sheet_name=name[:31], index=False)

    figure_s1(onestep, out)
    figure_s2(pd.read_csv(args.epi_dir / "v7d_all_states_annual_fit_with_incidence_ppc.csv"), fit["selected_states"], out)
    figure_s3(spec, start.rank_rho.iloc[0], fixed, out)

    states_conv = conv[conv.unit_kind == "state"]
    print(f"State refits converged: {states_conv.converged.sum()} of {states_conv.refits.sum()}; "
          f"max spectral radius {states_conv.spectral_radius_max.max():.4f}")
    print(f"One-step fit: MAE {100 * np.mean(np.abs(onestep.observed - onestep.fitted_p50)):.1f} pp, "
          f"95% coverage {100 * onestep.covered.mean():.0f}%")
    print(spec[["specification", "rank_rho_total", "rank_rho_incidence_2036"]].round(2).to_string(index=False))
    print(f"Projection-start cumulative incidence rank rho {start.rank_rho.iloc[0]:.2f}")
    print(f"Wrote supplementary outputs to {out}")


if __name__ == "__main__":
    main()
