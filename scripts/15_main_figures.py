#!/usr/bin/env python3
from __future__ import annotations

"""Main-text figures and Table 1, collected in results/paper/.

Fig. 1  Observed 2018-2024 change in the eight modeled measures, primary states.
Fig. 2  State heterogeneity in permitted pathway coefficients (primary fit).
Fig. 3  AHS-PrEP coupling strength, routes, and PrEP gain (copied from scripts/13).
Fig. 4  Demographic stigma and PrEP trajectories (copied from scripts/14).
Table 1 State outcomes under the AHS-reduction scenario, ordered by coupling strength.
"""

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

VARIABLES = ["stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "risk_behavior", "prep_used", "hivtest12"]
VARIABLE_LABELS = {
    "stigma_ahs": "Anticipated\nhealthcare\nstigma", "stigma_gss": "General\nsocial\nstigma",
    "stigma_family": "Family\nstigma", "out_gid": "Disclosure\nto provider", "seehcp": "Saw\nhealthcare\nprovider",
    "risk_behavior": "Risk\nbehavior", "prep_used": "PrEP\nuse", "hivtest12": "HIV\ntesting",
}
REGIONS = {
    "Northeast": ["MA", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH"],
    "South": ["FL", "GA", "MD", "NC", "TN", "TX", "VA"],
    "West": ["CA", "WA"],
}
# (source, target, label, family) for every permitted pathway touching stigma or prevention.
PATHWAYS = [
    ("stigma_gss", "stigma_ahs", "General social stigma → AHS", "Upstream stigma → AHS"),
    ("stigma_family", "stigma_ahs", "Family stigma → AHS", "Upstream stigma → AHS"),
    ("stigma_ahs", "prep_used", "AHS → PrEP use", "AHS → prevention"),
    ("stigma_ahs", "hivtest12", "AHS → HIV testing", "AHS → prevention"),
    ("stigma_ahs", "out_gid", "AHS → disclosure", "AHS ↔ engagement"),
    ("stigma_ahs", "seehcp", "AHS → saw provider", "AHS ↔ engagement"),
    ("seehcp", "stigma_ahs", "Saw provider → AHS", "AHS ↔ engagement"),
    ("out_gid", "prep_used", "Disclosure → PrEP use", "Engagement → prevention"),
    ("out_gid", "hivtest12", "Disclosure → HIV testing", "Engagement → prevention"),
    ("seehcp", "hivtest12", "Saw provider → HIV testing", "Engagement → prevention"),
    ("hivtest12", "prep_used", "HIV testing → PrEP use", "Engagement → prevention"),
    ("prep_used", "hivtest12", "PrEP use → HIV testing", "Engagement → prevention"),
]
FAMILY_COLORS = {
    "Upstream stigma → AHS": "#2a78d6", "AHS → prevention": "#eb6834",
    "AHS ↔ engagement": "#1baf7a", "Engagement → prevention": "#eda100",
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def figure1(workbook: Path, states: list[str], out: Path) -> pd.DataFrame:
    trends = pd.read_excel(workbook, sheet_name="trends")
    wide = trends[trends.Geography.isin(states)].pivot_table(index=["Geography", "Indicator"], columns="Year", values="Value")
    change = (100 * (wide[2024] - wide[2018])).unstack("Indicator")[VARIABLES]
    order = [s for region in REGIONS.values() for s in sorted(region) if s in states]
    table = pd.concat([change.loc[order], change.median().to_frame("Median").T])

    cmap = LinearSegmentedColormap.from_list("green_grey_purple", ["#006D2C", "#F2F2F0", "#762A83"], N=256)
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    im = ax.imshow(table.values, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-30, vmax=30), aspect="auto")
    ax.set_xticks(range(len(VARIABLES)), [VARIABLE_LABELS[v] for v in VARIABLES], fontsize=10)
    ax.set_yticks(range(len(table)), table.index, fontsize=10)
    ax.tick_params(length=0)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.iat[i, j]:+.0f}", ha="center", va="center", fontsize=8.5, color=INK)
    for x in (2.5, 4.5):
        ax.axvline(x, color=INK, lw=1.2)
    start = 0
    for region, members in REGIONS.items():
        n = sum(s in states for s in members)
        ax.axhline(start + n - 0.5, color=INK, lw=1.0)
        ax.text(-1.2, start + (n - 1) / 2, region, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
        start += n
    for x, label in ((1, "Stigma"), (3.5, "Disclosure and healthcare engagement"), (6, "Prevention and behavior")):
        ax.text(x, -1.9, label, ha="center", va="center", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Percentage-point change, 2018–2024\n(green = decrease, purple = increase)", fontsize=10)
    save(fig, out, "fig1_observed_change")
    return table


def figure2(fit_path: Path, out: Path) -> pd.DataFrame:
    fit = np.load(fit_path, allow_pickle=True)
    v = fit["v_names"].astype(str).tolist()
    units = fit["unit_order"].astype(str).tolist()
    states = fit["selected_states"].astype(str).tolist()
    rows = []
    for source, target, label, family in PATHWAYS:
        for state in states:
            vals = fit["Jmeans_stack"][:, units.index(state), v.index(target), v.index(source)].astype(float)
            q = np.quantile(vals, [0.25, 0.5, 0.75])
            rows.append({"pathway": label, "family": family, "state": state, "p25": q[0], "p50": q[1], "p75": q[2],
                         "share_at_zero": float(np.mean(vals == 0))})
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    labels = [p[2] for p in PATHWAYS][::-1]
    rng = np.random.default_rng(12)
    for y, label in enumerate(labels):
        sub = data[data.pathway == label]
        color = FAMILY_COLORS[sub.family.iloc[0]]
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F4F3F0", zorder=-5)
        ax.scatter(sub.p50, y + rng.uniform(-0.12, 0.12, len(sub)), s=12, color=color, alpha=0.45, edgecolor="none")
        q25, med, q75 = sub.p50.quantile([0.25, 0.5, 0.75])
        ax.plot([q25, q75], [y, y], color=color, lw=4, solid_capstyle="round")
        ax.scatter([med], [y], s=45, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Pathway coefficient (centered-logit scale; state medians across 500 refits)", fontsize=9, color=INK)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=6, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    fig.text(0.01, -0.06, "Small points: states. Large points and bars: cross-state median and IQR. AHS = anticipated healthcare stigma.",
             fontsize=8, color=MUTED)
    save(fig, out, "fig2_pathway_heterogeneity")
    return data


def table1(coupling: Path, epi_draws: Path, out: Path) -> pd.DataFrame:
    summary = pd.read_csv(coupling).set_index("state")
    pop = pd.read_csv(ROOT / "inputs/msm_state_population.csv").set_index("state")["rounded"]
    e = pd.read_csv(epi_draws)
    e = e[e["scenario"].eq("s1_reduce_ahs") & e["year"].eq(2036)]

    def q(series: pd.Series, scale: float = 1.0) -> str:
        lo, med, hi = (scale * series).quantile([0.25, 0.5, 0.75])
        return f"{med:.1f} ({lo:.1f} to {hi:.1f})"

    rows = []
    for state, row in summary.iterrows():
        cum = e[e.unit_id.eq(state) & e.variable.eq("incidence_cumulative")]
        inc = e[e.unit_id.eq(state) & e.variable.eq("incidence")]
        rows.append({
            "State": state,
            "Coupling strength (IQR)": f"{-row.total_p50:.2f} ({-row.total_p75:.2f} to {-row.total_p25:.2f})",
            "PrEP gain 2032, pp": f"{row.ahs_scenario_prep_gain_pp_2032:.1f}",
            "PrEP gain 2036, pp": f"{row.ahs_scenario_prep_gain_pp_2036:.1f}",
            "Cumulative incidence change 2025–2036, % (IQR)": q(cum["pct_change"]),
            "Infections averted per 1,000 MSM, 2025–2036 (IQR)": q(-cum["delta"] / pop[state] * 1000),
            "Incidence change in 2036, % (IQR)": q(inc["pct_change"]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "table1_state_outcomes_ahs_scenario.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workbook", type=Path, default=RESULTS / "01_weights/state_model_inputs.xlsx")
    parser.add_argument("--fit", type=Path, default=RESULTS / "03_fits/primary/fits/primary.npz")
    parser.add_argument("--coupling", type=Path, default=RESULTS / "07_structural/state_coupling_summary.csv")
    parser.add_argument("--epi-draws", type=Path, default=RESULTS / "04_scenarios/primary/scenario_endpoint_epi_draws.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "paper")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    states = np.load(args.fit, allow_pickle=True)["selected_states"].astype(str).tolist()
    figure1(args.workbook, states, args.output_dir).to_csv(args.output_dir / "fig1_data.csv")
    figure2(args.fit, args.output_dir).to_csv(args.output_dir / "fig2_data.csv", index=False)
    print(table1(args.coupling, args.epi_draws, args.output_dir).to_string(index=False))
    for src, stem in [(RESULTS / "07_structural/state_coupling", "fig3_coupling_strength"),
                      (RESULTS / "08_demographic/group_trajectories", "fig4_demographic_trajectories")]:
        for ext in (".png", ".pdf"):
            shutil.copyfile(src.with_suffix(ext), args.output_dir / f"{stem}{ext}")
    print(f"Wrote main-text outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
