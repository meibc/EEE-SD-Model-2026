from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/supplementary_figures"
OUT.mkdir(parents=True, exist_ok=True)

ROBUST = ["CA", "FL", "GA", "IL", "IN", "MA", "MD", "MI", "NC", "NJ", "NY", "OH", "PA", "TN", "TX", "VA", "WA"]

LABELS = {
    "stigma_ahs": "Anticipated\nhealthcare stigma",
    "stigma_gss": "General social\nstigma",
    "stigma_family": "Family\nstigma",
    "out_gid": "Disclosure to\nhealthcare provider",
    "seehcp": "Saw healthcare\nprovider",
    "risk_behavior": "HIV-related\nrisk behavior",
    "prep_used": "PrEP use",
    "hivtest12": "HIV testing\nfrequency",
}

BLUE = "#3E6FA8"
ORANGE = "#D98235"
PURPLE = "#7B5AA6"
DARK = "#262626"
GRID = "#D9D9D9"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
})


def dynamic_plot_data():
    path = ROOT / "output/amis_joint_bootstrap_direct_v2_n500/standardized_input_v9_direct_bootstrap.npz"
    z = np.load(path, allow_pickle=True)
    units = np.asarray(z["geo_ids"]).astype(str).tolist()
    variables = np.asarray(z["sem_v_names"]).astype(str).tolist()
    years = np.asarray(z["sem_obs_years"], dtype=int)
    observed = np.asarray(z["sem_obs"], dtype=float)
    fitted = np.asarray(z["sem_pred"], dtype=float)

    rows = []
    for state in ROBUST:
        g = units.index(state)
        for k, variable in enumerate(variables):
            for t, year in enumerate(years):
                rows.append({
                    "state": state,
                    "year": int(year),
                    "variable": variable,
                    "observed": observed[g, k, t],
                    "fitted": fitted[g, k, t],
                })
    data = pd.DataFrame(rows).dropna(subset=["observed", "fitted"])
    data["residual_pp"] = 100 * (data["fitted"] - data["observed"])
    return data, variables


def plot_dynamic_diagnostics():
    data, variables = dynamic_plot_data()
    data.to_csv(OUT / "figure_S3_dynamic_fit_plotting_data.csv", index=False)

    fig = plt.figure(figsize=(8.5, 7.5), facecolor="white")
    outer = fig.add_gridspec(
        2, 1, height_ratios=[2.25, 1.0],
        left=0.10, right=0.98, top=0.87, bottom=0.10, hspace=0.42,
    )
    scatter_grid = outer[0].subgridspec(2, 4, wspace=0.34, hspace=0.48)

    fig.text(0.10, 0.965, "A. Observed and fitted AMIS construct values", fontsize=11, weight="bold", va="top")

    for idx, variable in enumerate(variables):
        ax = fig.add_subplot(scatter_grid[idx // 4, idx % 4])
        d = data[data.variable == variable]
        lo = min(d.observed.min(), d.fitted.min())
        hi = max(d.observed.max(), d.fitted.max())
        pad = max((hi - lo) * 0.09, 0.015)
        lo = max(0, lo - pad)
        hi = min(1, hi + pad)

        ax.scatter(d.observed, d.fitted, s=16, color=BLUE, alpha=0.55, edgecolor="none")
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#666666")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(LABELS[variable], weight="bold", pad=5)
        ax.grid(color=GRID, lw=0.5, alpha=0.65)
        mae = 100 * np.mean(np.abs(d.fitted - d.observed))
        ax.text(0.04, 0.94, f"MAE = {mae:.1f} pp", transform=ax.transAxes, va="top", fontsize=7.5, color=DARK)
        if idx // 4 == 1:
            ax.set_xlabel("Observed")
        if idx % 4 == 0:
            ax.set_ylabel("Fitted")
        ax.spines[["top", "right"]].set_visible(False)

    axr = fig.add_subplot(outer[1])
    residual_position = axr.get_position()
    axr.set_position([0.21, residual_position.y0, 0.77, residual_position.height])
    fig.text(0.10, 0.355, "B. Fitted-minus-observed residuals by construct", fontsize=11, weight="bold", va="top")
    residuals = [data.loc[data.variable == v, "residual_pp"].to_numpy() for v in variables]
    bp = axr.boxplot(
        residuals, vert=False, patch_artist=True, widths=0.55, showfliers=True,
        medianprops={"color": DARK, "linewidth": 1.2},
        boxprops={"facecolor": "#BFD2E8", "edgecolor": BLUE, "linewidth": 0.9},
        whiskerprops={"color": BLUE, "linewidth": 0.9},
        capprops={"color": BLUE, "linewidth": 0.9},
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": BLUE, "markeredgecolor": "none", "alpha": 0.35},
    )
    axr.axvline(0, color="#666666", ls="--", lw=1)
    axr.set_yticks(np.arange(1, len(variables) + 1), [LABELS[v].replace("\n", " ") for v in variables])
    axr.invert_yaxis()
    axr.set_xlabel("Residual (percentage points)")
    axr.grid(axis="x", color=GRID, lw=0.6)
    axr.spines[["top", "right", "left"]].set_visible(False)
    axr.tick_params(axis="y", length=0)

    png = OUT / "figure_S3_dynamic_pathway_model_fit_diagnostics.png"
    pdf = OUT / "figure_S3_dynamic_pathway_model_fit_diagnostics.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_epidemiological_diagnostics():
    data_dir = ROOT / "output/supplementary_results_data"
    incidence = pd.read_csv(data_dir / "S9_incidence_calibration_ppc.csv")
    diagnosis = pd.read_csv(data_dir / "S9_diagnosis_calibration_ppc.csv")
    incidence = incidence[incidence.state.isin(ROBUST)].copy()
    diagnosis = diagnosis[diagnosis.state.isin(ROBUST)].copy()

    fig = plt.figure(figsize=(8.5, 5.4), facecolor="white")
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.22],
        left=0.09, right=0.98, top=0.91, bottom=0.17, wspace=0.36,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel A: posterior-predictive incidence versus surveillance target.
    included = incidence.year != 2020
    for mask, color, marker, label, alpha in [
        (included, BLUE, "o", "Calibration years", 0.62),
        (~included, "#999999", "s", "2020 (excluded)", 0.70),
    ]:
        d = incidence.loc[mask]
        x = d.inc_observed.to_numpy() / 1000
        y = d.inc_ppc_median.to_numpy() / 1000
        yerr = np.vstack([
            y - d.inc_ppc_p05.to_numpy() / 1000,
            d.inc_ppc_p95.to_numpy() / 1000 - y,
        ])
        ax1.errorbar(x, y, yerr=yerr, fmt=marker, ms=4.0, color=color, ecolor=color,
                     elinewidth=0.7, alpha=alpha, capsize=0, label=label)

    max_inc = max(incidence.inc_observed.max(), incidence.inc_ppc_p95.max()) / 1000 * 1.04
    ax1.plot([0, max_inc], [0, max_inc], ls="--", lw=1, color="#666666")
    coverage = np.mean(
        (incidence.loc[included, "inc_observed"] >= incidence.loc[included, "inc_ppc_p05"])
        & (incidence.loc[included, "inc_observed"] <= incidence.loc[included, "inc_ppc_p95"])
    )
    ax1.text(0.04, 0.96, f"90% predictive coverage = {coverage:.0%}", transform=ax1.transAxes,
             ha="left", va="top", fontsize=8)
    ax1.set_xlim(0, max_inc)
    ax1.set_ylim(0, max_inc)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("Observed incidence (thousands)")
    ax1.set_ylabel("Posterior-predicted incidence (thousands)")
    ax1.set_title("A. Annual HIV incidence", loc="left", weight="bold", pad=9)
    ax1.grid(color=GRID, lw=0.5, alpha=0.65)
    ax1.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel B: state-level period-average diagnosis posterior predictions.
    state_order = (
        diagnosis.groupby("state")["observed_period_mean"].mean().sort_values().index.tolist()
    )
    ybase = np.arange(len(state_order))
    period_specs = [("2017-2019", BLUE, -0.16), ("2021-2022", ORANGE, 0.16)]
    for period, color, offset in period_specs:
        d = diagnosis[diagnosis.period == period].set_index("state").loc[state_order]
        med = d.predictive_median.to_numpy() / 1000
        lo = d.predictive_p05.to_numpy() / 1000
        hi = d.predictive_p95.to_numpy() / 1000
        obs = d.observed_period_mean.to_numpy() / 1000
        yy = ybase + offset
        ax2.errorbar(
            med, yy, xerr=np.vstack([med - lo, hi - med]), fmt="o", ms=4.2,
            color=color, ecolor=color, elinewidth=1.0, capsize=2, alpha=0.82,
        )
        ax2.scatter(obs, yy, marker="|", s=70, linewidth=1.5, color=DARK, zorder=4)

    ax2.set_yticks(ybase, state_order)
    ax2.set_xlabel("Period-average diagnoses (thousands)")
    ax2.set_title("B. Period-average HIV diagnoses", loc="left", weight="bold", pad=9)
    ax2.grid(axis="x", color=GRID, lw=0.6)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, lw=1, label="2017-2019 prediction", markersize=4),
            Line2D([0], [0], marker="o", color=ORANGE, lw=1, label="2021-2022 prediction", markersize=4),
            Line2D([0], [0], marker="|", color=DARK, lw=0, label="Observed target", markersize=9),
        ],
        frameon=False, loc="lower right", fontsize=7.5,
    )

    fig.text(
        0.09, 0.045,
        "Points show posterior-predictive medians; bars show 90% posterior-predictive intervals.\n"
        "Incidence observations from 2020 are displayed for context but were excluded from calibration.",
        fontsize=7.4, color="#444444", ha="left", va="bottom",
    )

    png = OUT / "figure_S4_epidemiological_calibration_diagnostics.png"
    pdf = OUT / "figure_S4_epidemiological_calibration_diagnostics.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


if __name__ == "__main__":
    outputs = [*plot_dynamic_diagnostics(), *plot_epidemiological_diagnostics()]
    for output in outputs:
        print(output)
