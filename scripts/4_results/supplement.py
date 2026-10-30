#!/usr/bin/env python3
from __future__ import annotations

"""Supplement: model specification table, dynamic-model fit and epidemic-model calibration.

Table S  Model specification: centred-logit references, state anchoring, drift, persistence and pathway
         priors, the evidence constraint, optimization and geographic shrinkage settings
         (config/, results/02_inputs/reported_constants.csv).
Fig. S   Pathway structure: permitted pathways by sign constraint and prior tier, persistence and drift
         (config/), with the AHS -> PrEP and PrEP -> AHS links highlighted. A draft for the author's figure.
Fig. S   Dynamic-model fit: observed state-period values (2020, 2022, 2024) against one-step-ahead fitted
         values, eight measures, primary states. Each draw's fitted system (transition matrix, drift,
         state anchor) is applied to that draw's previous-period values; medians and 95% intervals across
         the 500 respondent-bootstrap refits. In-sample: the fits used these data. Each panel gives the mean
         absolute error next to the median sampling SD of the observed values (bootstrap), and the share of
         observed values whose per-draw residual (observed minus fitted, same draw, so it includes the
         observation's own sampling error) has a 95% interval containing zero.
Fig. S   Epidemic-model calibration: observed CDC period means (new diagnoses, estimated incidence) against
         posterior period means with 90% intervals, primary states (scripts/3_model/02).

Outputs (results/sections/supplement/): tableS_model_specification.csv; figS_pathways.{png,pdf}; figS_dynamic_fit.{png,pdf} and
dynamic_fit.csv; figS_epi_calibration.{png,pdf} and epi_calibration_summary.csv.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.optimization import OptimConfig  # noqa: E402
from config.shrinkage import ShrinkageConfig  # noqa: E402
from core.transforms import LOGIT_EPS, inverse_logit, logit  # noqa: E402
from reporting import sources, style  # noqa: E402
from reporting.summaries import primary_states  # noqa: E402

OUT = sources.SECTIONS / "supplement"
CONFIG = ROOT / "config"
BUNDLE = sources.RESULTS / "03_fits/primary/standardized_research.npz"
EPI_FIT = sources.RESULTS / "02_inputs/epi_fit/period_fit_summary.csv"
CONSTANTS = sources.RESULTS / "02_inputs/reported_constants.csv"
LABELS = {**style.MEASURE_LABELS, "seehcp": "Healthcare provider visit"}


def specification_table() -> pd.DataFrame:
    rows = []
    variables = pd.read_csv(CONFIG / "dynamic_variables.csv").set_index("variable")
    for v, r in variables.iterrows():
        rows.append(("Centred-logit reference (probability)", LABELS[v], f"{r.reference_prob:.3f}",
                     "5th percentile of weighted state-period estimates, 24 primary states x 4 periods; sets the "
                     "AHS benchmark and the evidence conversion"))
    rows.append(("State anchoring", "All measures", "State mean logit",
                 "Each measure centred at the state's own mean logit over the observed periods (per draw)"))
    for v, r in variables[variables.drift_enabled == 1].iterrows():
        rows.append(("Drift prior (centre, scale)", LABELS[v], f"{r.drift_prior_mean:.2f}, {r.drift_prior_tau:.2f}",
                     r.drift_note))
    for v in variables.index[variables.drift_enabled == 0]:
        rows.append(("Drift", LABELS[v], "Fixed at 0", ""))
    diagonal = pd.read_csv(CONFIG / "dynamic_persistence_priors.csv").set_index("variable")
    for v, r in diagonal.iterrows():
        rows.append(("Persistence prior (centre, scale)", LABELS[v], f"{r.prior_mean:.2f}, {r.prior_tau:.2f}",
                     r.evidence_note))
    rows.append(("Persistence", "Other measures", "Estimated, in [0, 3]", "No target; non-negative (no oscillation)"))
    pathways = pd.read_csv(CONFIG / "dynamic_pathway_priors.csv")
    for _, r in pathways.iterrows():
        sign = {"1": "non-negative", "-1": "non-positive", "both": "either sign"}[str(r.sign)]
        rows.append((f"Pathway prior, {r.evidence_tier} (centre, scale)", f"{LABELS[r.source]} -> {LABELS[r.target]}",
                     f"{r.prior_mean:+.2f}, {r.prior_tau:.1f}", sign))
    opt, shrink = OptimConfig(), ShrinkageConfig()
    rows += [
        ("Evidence constraint", "Total AHS -> PrEP response", f"{opt.evidence_horizon} steps, tolerance {opt.evidence_tau}",
         "Matched to each state's access-adjusted prevalence ratio (AMIS 2017-2021; scripts/2_evidence/01), per draw"),
        ("Objective weight", "Trajectory fit", f"{opt.wX}", ""),
        ("Objective weight", "Covariance consistency", f"{opt.wR}",
         f"Innovation SD {opt.sigma_eta_scale} x each measure's SD"),
        ("Objective weight", "Stability penalty", f"{opt.w_stab:g}", "Above spectral radius 0.999 (soft)"),
        ("Geographic shrinkage", "Division -> nation", f"{shrink.wJ}", ""),
        ("Geographic shrinkage", "State -> division", f"{shrink.wJ_state} x {shrink.state_n_ref:g} / n", "n = state sample size"),
        ("Bounds", "Off-diagonal pathways", f"{opt.bounds}", "Within the pathway's sign"),
        ("Bounds", "Drift", f"+/-{opt.drift_bounds}", "Centred-logit units per two-year step"),
        ("Optimizer", "L-BFGS-B", f"maxiter {opt.maxiter}, ftol {opt.ftol:g}, gtol {opt.gtol:g}", ""),
    ]
    for _, r in pd.read_csv(CONSTANTS).iterrows():
        if r.constant.startswith("reference_"):
            continue
        se = "" if pd.isna(r.se) else f" (SE {r.se:.3f})"
        rows.append(("Data-derived constant", r.constant, f"{r.value:.3f}{se}", r.definition))
    return pd.DataFrame(rows, columns=["component", "item", "value", "note"])


PATHWAY_POS = {"stigma_gss": (0.0, 1.0), "stigma_family": (0.0, -1.0), "stigma_ahs": (1.6, 0.35),
       "out_gid": (3.5, 1.75), "seehcp": (3.5, -0.75), "hivtest12": (3.5, -2.2),
       "prep_used": (5.25, 0.9), "risk_behavior": (5.25, -1.3)}
PATHWAY_NAME = {"stigma_gss": "General social\nstigma", "stigma_family": "Family stigma", "stigma_ahs": "Anticipated\nhealthcare stigma",
        "out_gid": "Disclosure to\nprovider", "seehcp": "Healthcare\nprovider visit", "hivtest12": "HIV testing",
        "prep_used": "PrEP use", "risk_behavior": "HIV-related\nrisk behavior"}
PATHWAY_NOTE = {"stigma_ahs": "drift", "prep_used": "drift; persistence 0.85", "risk_behavior": "drift",
        "seehcp": "persistence 0.95", "stigma_gss": "persistence 0.85", "stigma_family": "persistence 0.85"}
PATHWAY_SIGN = {"1": style.CONTINUED, "-1": "#D55E00", "both": "#8c8c8c"}
PATHWAY_TIER = {"strong": dict(ls="-", lw=2.0), "moderate": dict(ls=(0, (5, 2)), lw=1.3), "weak": dict(ls=(0, (1.2, 1.6)), lw=1.4)}
PATHWAY_KEY = {("stigma_ahs", "prep_used"), ("prep_used", "stigma_ahs")}
PATHWAY_RAD = {("out_gid", "hivtest12"): -0.55, ("stigma_ahs", "prep_used"): 0.0, ("prep_used", "stigma_ahs"): 0.12}
BOX_W, BOX_H, BOX_GAP = 1.22, 0.62, 0.06


def edge_point(c, o):
    """Where the ray from box centre c toward o leaves the box, plus a small gap."""
    c, o = np.array(c), np.array(o)
    d = o - c
    t = min((BOX_W / 2 + BOX_GAP) / abs(d[0]) if d[0] else np.inf, (BOX_H / 2 + BOX_GAP) / abs(d[1]) if d[1] else np.inf)
    return tuple(c + t * d)


def pathway_figure() -> plt.Figure:
    """Permitted pathways from config/dynamic_pathway_priors.csv: colour = sign constraint, line style =
    prior tier; persistence and drift noted in the boxes; the AHS -> PrEP and PrEP -> AHS links highlighted."""
    style.apply()
    P = pd.read_csv(CONFIG / "dynamic_pathway_priors.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    pairs = {(r.source, r.target) for r in P.itertuples()}
    for r in P.itertuples():
        s, t = PATHWAY_POS[r.source], PATHWAY_POS[r.target]
        rad = PATHWAY_RAD.get((r.source, r.target), 0.2 if (r.target, r.source) in pairs else 0.0)
        a, b = edge_point(s, t), edge_point(t, s)
        if (r.source, r.target) in PATHWAY_KEY:
            ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-", lw=8,
                                         color="#F0E442", alpha=0.45, zorder=1))
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=9,
                                     color=PATHWAY_SIGN[str(r.sign)], zorder=2, **PATHWAY_TIER[r.evidence_tier]))
    for v, (x, y) in PATHWAY_POS.items():
        key = v in ("stigma_ahs", "prep_used")
        ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="white", ec=style.INK if key else style.MUTED, lw=1.4 if key else 0.8, zorder=3))
        ax.text(x, y + (0.07 if v in PATHWAY_NOTE else 0), PATHWAY_NAME[v], ha="center", va="center", fontsize=6.9 if key else 7.2, zorder=4,
                fontweight="bold" if key else "normal", linespacing=1.0)
        if v in PATHWAY_NOTE:
            ax.text(x, y - BOX_H / 2 + 0.06, PATHWAY_NOTE[v], ha="center", va="bottom", fontsize=5.8, color=style.MUTED, zorder=4)
    for x, label in [(0, "Upstream stigma"), (1.6, "Healthcare stigma"), (3.5, "Healthcare engagement"), (5.25, "Prevention")]:
        ax.text(x, 2.5, label, ha="center", fontsize=7.8, color=style.MUTED, fontweight="bold")
    ax.set_xlim(-0.75, 6.0)
    ax.set_ylim(-2.65, 2.75)
    ax.axis("off")
    h = [plt.Line2D([], [], color=PATHWAY_SIGN["1"], lw=1.6, label="Positive"),
         plt.Line2D([], [], color=PATHWAY_SIGN["-1"], lw=1.6, label="Negative"),
         plt.Line2D([], [], color=PATHWAY_SIGN["both"], lw=1.6, label="Either sign"),
         plt.Line2D([], [], color=style.INK, **PATHWAY_TIER["strong"], label="Strong prior"),
         plt.Line2D([], [], color=style.INK, **PATHWAY_TIER["moderate"], label="Moderate prior"),
         plt.Line2D([], [], color=style.INK, **PATHWAY_TIER["weak"], label="Weak prior")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=6.8, bbox_to_anchor=(0.5, 0.0), handlelength=2.4)
    fig.legend(handles=[plt.Line2D([], [], color="#F0E442", alpha=0.6, lw=7)],
               labels=["AHS → PrEP: total response matched to survey evidence (aPR 0.55).  "
                       "PrEP → AHS: continued or not continued in projections."],
               loc="lower center", fontsize=6.8, bbox_to_anchor=(0.5, -0.045), handlelength=2.4)
    return fig


def dynamic_fit() -> pd.DataFrame:
    z = np.load(BUNDLE, allow_pickle=True)
    v_names, units, years = z["sem_v_names"].astype(str).tolist(), z["geo_ids"].astype(str).tolist(), z["sem_obs_years"]
    J, drift, anchor = z["sem_J_samples"], z["sem_drift_samples"], z["sem_anchor_logits"]   # draws x units (x ...)
    obs = np.clip(z["sem_obs_samples"], LOGIT_EPS, 1 - LOGIT_EPS)                                        # draws x units x m x T
    X = logit(obs, LOGIT_EPS) - anchor[..., None]
    fitted_logit = np.einsum("duij,dujt->duit", J, X[..., :-1]) + drift[..., None] + anchor[..., None]
    fitted = inverse_logit(fitted_logit)
    rows = []
    for state in primary_states():
        u = units.index(state)
        for i, measure in enumerate(v_names):
            for t in range(1, len(years)):
                q = np.quantile(fitted[:, u, i, t - 1], [0.025, 0.5, 0.975])
                residual = np.quantile(obs[:, u, i, t] - fitted[:, u, i, t - 1], [0.025, 0.975])
                rows.append({"state": state, "measure": measure, "year": int(years[t]),
                             "observed": float(z["sem_obs"][u, i, t]), "observed_sd": float(obs[:, u, i, t].std()),
                             "fitted_median": q[1], "fitted_low": q[0], "fitted_high": q[2],
                             "covered": bool(residual[0] <= 0 <= residual[1])})
    return pd.DataFrame(rows)


def dynamic_fit_figure(fit: pd.DataFrame) -> plt.Figure:
    style.apply()
    measures = list(dict.fromkeys(fit["measure"]))
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.8))
    colors = {2020: style.NOT_CONTINUED, 2022: style.ACCENT, 2024: style.CONTINUED}
    for ax, measure in zip(axes.flat, measures):
        d = fit[fit["measure"].eq(measure)]
        for year, color in colors.items():
            e = d[d["year"].eq(year)]
            ax.errorbar(100 * e["observed"], 100 * e["fitted_median"],
                        yerr=[100 * (e["fitted_median"] - e["fitted_low"]), 100 * (e["fitted_high"] - e["fitted_median"])],
                        fmt="o", ms=2.5, color=color, elinewidth=0.4, alpha=0.8, label=str(year))
        lo = 100 * min(d["observed"].min(), d["fitted_low"].min())
        hi = 100 * max(d["observed"].max(), d["fitted_high"].max())
        ax.plot([lo, hi], [lo, hi], color=style.MUTED, lw=0.8, ls="--", zorder=0)
        mae = 100 * np.mean(np.abs(d["observed"] - d["fitted_median"]))
        ax.set_title(f"{LABELS[measure]}\nMAE {mae:.1f} pp (sampling SD {100 * d['observed_sd'].median():.1f})\n"
                     f"{100 * d['covered'].mean():.0f}% consistent with the fit", loc="left", fontsize=7)
        ax.tick_params(labelsize=7)
    fig.supxlabel("Observed (%)", fontsize=9)
    fig.supylabel("One-step-ahead fitted (%)", fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def epi_calibration() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(EPI_FIT)
    d = d[d["state"].isin(primary_states()) & d["observed"].notna()]
    summary = d.groupby(["outcome", "period"]).agg(
        states=("state", "size"), coverage_90=("covered_90pct", "mean"),
        median_abs_error_pct=("relative_error_median", lambda x: 100 * x.abs().median())).reset_index()
    return d, summary


def epi_figure(d: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    style.apply()
    colors = {"2017-2019": style.NOT_CONTINUED, "2021-2022": style.ACCENT, "2023-2024": style.CONTINUED}
    titles = {"diagnoses": "HIV diagnoses", "incidence": "Estimated HIV incidence"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    for ax, outcome in zip(axes, titles):
        e = d[d["outcome"].eq(outcome)]
        for period, color in colors.items():
            p = e[e["period"].eq(period)]
            if p.empty:
                continue
            ax.errorbar(p["observed"], p["posterior_median"],
                        yerr=[p["posterior_median"] - p["posterior_p05"], p["posterior_p95"] - p["posterior_median"]],
                        fmt="o", ms=3, color=color, elinewidth=0.5, label=period.replace("-", "–"))
        lo, hi = e[["observed", "posterior_p05"]].min().min() * 0.8, e[["observed", "posterior_p95"]].max().max() * 1.2
        ax.plot([lo, hi], [lo, hi], color=style.MUTED, lw=0.8, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        s = summary[summary["outcome"].eq(outcome)]
        note = "\n".join(f"{r.period.replace('-', '–')}: {100 * r.coverage_90:.0f}% of states in the 90% interval"
                         for r in s.itertuples())
        ax.set_title(f"{titles[outcome]}\n{note}", loc="left", fontsize=7.5)
        ax.set_xlabel("Observed (CDC), period mean per year")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Model posterior (median, 90% interval)")
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specification_table().to_csv(OUT / "tableS_model_specification.csv", index=False)
    style.save(pathway_figure(), OUT / "figS_pathways")
    fit = dynamic_fit()
    fit.to_csv(OUT / "dynamic_fit.csv", index=False)
    style.save(dynamic_fit_figure(fit), OUT / "figS_dynamic_fit")
    epi, summary = epi_calibration()
    summary.to_csv(OUT / "epi_calibration_summary.csv", index=False)
    style.save(epi_figure(epi, summary), OUT / "figS_epi_calibration")
    fit["abs_error_pp"] = 100 * (fit["observed"] - fit["fitted_median"]).abs()
    print(fit.groupby("measure")[["abs_error_pp", "covered"]].mean().round(2))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
