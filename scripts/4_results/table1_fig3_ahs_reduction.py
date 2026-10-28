#!/usr/bin/env python3
from __future__ import annotations

"""Results, section 3: projected effects of reducing anticipated healthcare stigma (AHS) (Table 1, Fig. 3).

Scenario PrEP changes come from the respondent-level AHS-PrEP prevalence ratio applied at each state's
projected levels (scripts/3_model/05 --constant-pr-prep), with the fitted PrEP -> AHS relationship continued
(primary) or not continued in the projections.

Table 1  Pooled over the primary states: infections averted 2025-2036 (number and percent of baseline)
         and the change in 2036 incidence, median and 95% interval, for halving AHS from its 2024 level,
         stigma-resilient PrEP delivery (half of the PrEP gain from reducing AHS to the 5th-percentile
         benchmark, AHS unchanged) and halving general social and family stigma. (Reducing AHS to the
         benchmark itself is in the sensitivity table: nearly identical to halving.)
Fig. 3   Where halving AHS helps most, and why, by state (2025-2036): (A) infections averted per 100,000 MSM
         per year; (B) baseline infections per 100,000 MSM per year; (C) AHS percentage points removed by
         halving (2036); (D) PrEP gain (percentage points, 2036) per AHS point removed. Per draw,
         A = B x C x D x infections averted per PrEP point, the last nearly constant across states. A and C
         by projection; B and D shown for the primary projection (nearly identical in both). Median and
         middle 50% of draws; states ordered by A (primary).
Fig. S   By state, relationship continued and not continued: (A) projected 2036 AHS without intervention,
         with observed 2023-24 AHS, shading above the highest AHS observed in any primary state and period;
         (B) reduction in 2036 incidence from halving AHS. Median, middle 50% and 95% of draws, axes cut
         at 60% (A) and 50% (B) with an arrowhead where the 95% range continues.
Table S  By state and projection, halving AHS: change in 2036 incidence (%) and infections averted
         2025-2036 (number, percent of baseline, per 100,000 MSM per year), median and 95% interval; and
         the share of draws whose projected 2036 AHS without intervention exceeds 30% and 50% (above any
         level observed in a state). Rate denominators: fixed state MSM population
         (inputs/msm_state_population.csv) x 12 years.

Outputs (results/sections/ahs_reduction/): table1_pooled.csv; fig3_state_breakdown.{png,pdf};
figS_state_effects.{png,pdf}; tableS_state_effects.csv; state_breakdown.csv (all measures and factors,
median, interquartile range and 95% interval); breakdown_variance_shares.csv (share of the across-state
variance in log rate attributable to each factor); state_effects.csv; projected_ahs.csv; association.csv.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reporting import sources, style  # noqa: E402
from reporting.summaries import END_YEAR, interval, observed_paths, pooled_effect, primary_states, projected_ahs, state_effects  # noqa: E402

OUT = sources.SECTIONS / "ahs_reduction"
SCENARIOS = {
    "halve_ahs": "Halve AHS from its 2024 level",
    "resilient_prep_delivery": "Stigma-resilient PrEP delivery",
    "halve_upstream_stigma": "Halve general social and family stigma",
}
PROJECTIONS = {"continued": sources.CONTINUED, "not_continued": sources.NOT_CONTINUED}


def table1() -> pd.DataFrame:
    rows = []
    for code, label in SCENARIOS.items():
        for projection, path in PROJECTIONS.items():
            rows.append({"scenario": label, "projection": projection, **pooled_effect(path, code)})
    return pd.DataFrame(rows)


def association() -> pd.DataFrame:
    """Respondent-level AHS-PrEP prevalence ratio: pooled and the range of state estimates (primary states)."""
    ev = pd.read_csv(sources.AHS_PREP_EVIDENCE).set_index("state")
    pooled = ev.loc["POOLED"]
    states = ev.loc[primary_states()]
    return pd.DataFrame([{"pooled_apr": pooled["apr"], "pooled_low": pooled["apr_low"], "pooled_high": pooled["apr_high"],
                          "state_pooled_min": states["apr_pooled"].min(), "state_pooled_max": states["apr_pooled"].max(),
                          "state_unpooled_min": states["apr"].min(), "state_unpooled_max": states["apr"].max()}])


YEARS = END_YEAR - 2024    # projection years 2025-2036
FACTORS = {"rate": "Infections averted per 100,000 MSM per year", "burden": "Baseline infections per 100,000 MSM per year",
           "gap": "AHS points removed by halving", "prep_per_point": "PrEP gain per AHS point removed"}


def state_draws(path: Path, scenario: str = "halve_ahs") -> pd.DataFrame:
    """Per state and draw: effects on three scales and the factors of the per-capita rate."""
    states = primary_states()
    e = pd.read_csv(path / "scenario_endpoint_epi_draws.csv.gz")
    e = e[e["scenario"].eq(scenario) & e["year"].eq(END_YEAR) & e["unit_id"].isin(states)].set_index(["unit_id", "draw"])
    s = pd.read_csv(path / "scenario_endpoint_dynamic_draws.csv.gz")
    s = s[s["scenario"].eq(scenario) & s["year"].eq(END_YEAR) & s["unit_id"].isin(states)].set_index(["unit_id", "draw"])
    cum, inc = e[e["variable"].eq("incidence_cumulative")], e[e["variable"].eq("incidence")]
    ahs, prep = s[s["variable"].eq("stigma_ahs")], s[s["variable"].eq("prep_used")]
    person_years = pd.read_csv(sources.MSM_POPULATION).set_index("state")["rounded"].reindex(
        cum.index.get_level_values("unit_id")).to_numpy() * YEARS
    averted = cum["baseline"] - cum["scenario_value"]
    gap = 100 * (ahs["baseline"] - ahs["scenario_value"])
    return pd.DataFrame({
        "incidence_pct": inc["pct_change"], "averted": averted, "averted_pct": 100 * averted / cum["baseline"],
        "rate": 1e5 * averted / person_years, "burden": 1e5 * cum["baseline"] / person_years, "gap": gap,
        "prep_per_point": 100 * (prep["scenario_value"] - prep["baseline"]) / gap,
    })


def state_table(draws: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {(p, state, col): interval(g[col]) for p, d in draws.items() for state, g in d.groupby(level="unit_id")
            for col in d.columns}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.names = ["projection", "state", "measure"]
    return out


TABLE_MEASURES = {"incidence_pct": "incidence_change_2036_pct", "averted": "infections_averted",
                  "averted_pct": "infections_averted_pct", "rate": "averted_per_100k_msm_year"}


def supplement_table(table: pd.DataFrame, projected: pd.DataFrame) -> pd.DataFrame:
    """One row per projection and state: effects (median, 95% interval) and projected-AHS shares."""
    rows = []
    for projection in PROJECTIONS:
        t = table.loc[projection]
        for state in primary_states():
            row = {"projection": projection, "state": state}
            for measure, name in TABLE_MEASURES.items():
                row.update({f"{name}_{k}": t.loc[(state, measure), k] for k in ("median", "low", "high")})
            row["share_ahs_2036_above_30"] = projected.loc[state, f"{projection}_share_above_30"]
            row["share_ahs_2036_above_50"] = projected.loc[state, f"{projection}_share_above_50"]
            rows.append(row)
    return pd.DataFrame(rows)


def variance_shares(table: pd.DataFrame) -> pd.DataFrame:
    """Share of the across-state variance in log rate (medians) attributable to each factor (covariance
    with the total over its variance); the remainder is infections averted per PrEP point."""
    out = {}
    for projection in PROJECTIONS:
        m = np.log(table.loc[projection, "median"].unstack("measure")[list(FACTORS)].clip(lower=1e-6))
        out[projection] = {f: float(np.cov(m[f], m["rate"])[0, 1] / m["rate"].var()) for f in list(FACTORS)[1:]}
    return pd.DataFrame(out)


SHORT = {"continued": "Relationship continued", "not_continued": "Relationship not continued"}


def _paired(ax, table: dict[str, pd.DataFrame], order: list[str], cap: float) -> None:
    """Median (marker), middle 50% (thick bar) and 95% (thin line) of draws for both projections by state,
    cut at ``cap`` with an arrowhead where the 95% range continues."""
    y = np.arange(len(order))
    for projection, dy in (("not_continued", 0.18), ("continued", -0.18)):
        e, color, yy = table[projection].loc[order], style.COLORS[projection], y + dy
        ax.hlines(yy, e["low"], np.minimum(e["high"], cap), color=color, lw=0.8, alpha=0.6)
        ax.hlines(yy, e["p25"], np.minimum(e["p75"], cap), color=color, lw=3.2, alpha=0.9)
        over = (e["high"] > cap).to_numpy()
        ax.scatter(np.full(over.sum(), cap), yy[over], marker=">", s=12, color=color, alpha=0.6, lw=0)
        ax.scatter(e["median"], yy, marker=style.MARKERS[projection], s=24, color="white", edgecolor=color, lw=1.2,
                   zorder=3, label=SHORT[projection])
    ax.set_xlim(0, cap * 1.03)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.grid(axis="y", visible=False)


def figure(effects: dict[str, pd.DataFrame], projected: pd.DataFrame, observed_max: float) -> plt.Figure:
    """A: projected 2036 AHS without intervention; B: reduction in 2036 incidence from halving AHS; same state order."""
    style.apply()
    reduction = {p: pd.DataFrame({"median": -e["median"], "p25": -e["p75"], "p75": -e["p25"], "low": -e["high"],
                                  "high": -e["low"]}) for p, e in effects.items()}
    order = list(reduction["continued"]["median"].sort_values(ascending=False).index)   # largest reduction at the top
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 6.4), sharey=True, gridspec_kw={"wspace": 0.08})
    ahs = {p: projected[[f"{p}_2036", f"{p}_2036_p25", f"{p}_2036_p75", f"{p}_2036_low", f"{p}_2036_high"]]
           .set_axis(["median", "p25", "p75", "low", "high"], axis=1) for p in PROJECTIONS}
    _paired(a, ahs, order, cap=60)
    a.scatter(projected.loc[order, "observed_2024"], np.arange(len(order)), marker="D", s=14, color=style.OBSERVED,
              zorder=4, label="Observed 2023–24")
    a.axvspan(observed_max, 62, color="#f2f2f2", zorder=0)
    a.text(observed_max + 1, len(order) - 0.4, "Above any observed\nstate level", fontsize=7, color=style.MUTED, va="bottom")
    a.set_xlabel(f"AHS in {END_YEAR} without intervention (%)")
    a.set_title("A", loc="left", fontweight="bold", fontsize=11)
    _paired(b, reduction, order, cap=50)
    b.set_xlabel(f"Reduction in {END_YEAR} HIV incidence\nfrom halving AHS (%)")
    b.set_title("B", loc="left", fontweight="bold", fontsize=11)
    a.invert_yaxis()
    handles, labels = a.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=style.MUTED, lw=3.2), plt.Line2D([], [], color=style.MUTED, lw=0.8)]
    labels += ["Middle 50% of draws", "95% of draws (▸ continues)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3, fontsize=7.5)
    return fig


def breakdown_figure(table: pd.DataFrame) -> plt.Figure:
    style.apply()
    q = {p: table.loc[p].unstack("measure") for p in PROJECTIONS}
    order = list(q["continued"]["median"]["rate"].sort_values(ascending=False).index)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 6.0), sharey=True, gridspec_kw={"width_ratios": [1.5, 0.9, 1.2, 0.9], "wspace": 0.22})
    panels = [("rate", "A  Infections averted\nper 100,000 MSM\nper year", True), ("burden", "B  Baseline\ninfections per\n100,000 MSM/year", False),
              ("gap", "C  AHS points\nremoved by\nhalving", True), ("prep_per_point", "D  PrEP gain\nper AHS point\nremoved", False)]
    for ax, (col, title, both) in zip(axes, panels):
        for projection, dy in ((("not_continued", 0.17), ("continued", -0.17)) if both else (("continued", 0.0),)):
            d = {k: q[projection][k][col].loc[order] for k in ("median", "p25", "p75")}
            color = style.COLORS[projection] if both else style.OBSERVED
            ax.hlines(y + dy, d["p25"], d["p75"], color=color, lw=2.6, alpha=0.5)
            ax.scatter(d["median"], y + dy, marker=style.MARKERS[projection] if both else "D", s=20 if both else 14,
                       color=color, edgecolor="white", lw=0.5, zorder=3)
        ax.set_title(title, loc="left", fontsize=7.8, fontweight="bold" if col == "rate" else "normal")
        ax.set_xlim(0, None)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=style.COLORS[p], marker=style.MARKERS[p], ls="", ms=5) for p in ("continued", "not_continued")]
    handles += [plt.Line2D([], [], color=style.OBSERVED, marker="D", ls="", ms=4), plt.Line2D([], [], color=style.MUTED, lw=2.6, alpha=0.5)]
    fig.legend(handles, [SHORT["continued"], SHORT["not_continued"], "Primary projection (B, D)", "Middle 50% of draws"],
               loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = table1()
    t1.to_csv(OUT / "table1_pooled.csv", index=False)
    association().to_csv(OUT / "association.csv", index=False)
    effects = {p: state_effects(path, "halve_ahs") for p, path in PROJECTIONS.items()}
    pd.concat(effects, names=["projection"]).to_csv(OUT / "state_effects.csv")
    table = state_table({p: state_draws(path) for p, path in PROJECTIONS.items()})
    table.to_csv(OUT / "state_breakdown.csv")
    shares = variance_shares(table)
    shares.to_csv(OUT / "breakdown_variance_shares.csv", index_label="factor")
    style.save(breakdown_figure(table), OUT / "fig3_state_breakdown")
    projected = projected_ahs()
    projected.to_csv(OUT / "projected_ahs.csv")
    supplement_table(table, projected).to_csv(OUT / "tableS_state_effects.csv", index=False)
    observed_max = float(observed_paths(("stigma_ahs",))["level"].max())
    style.save(figure(effects, projected, observed_max), OUT / "figS_state_effects")
    cols = ["scenario", "projection", "averted_median", "averted_low", "averted_high", "averted_pct_median",
            "incidence_pct_median", "incidence_pct_low", "incidence_pct_high"]
    print(t1[cols].round(2).to_string(index=False))
    print(shares.round(2).to_string())


if __name__ == "__main__":
    main()
