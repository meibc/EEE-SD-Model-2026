#!/usr/bin/env python3
"""Results, section 3: projected effects of reducing anticipated healthcare stigma (AHS) (Table 1, Fig. 3).

Every result is given for the three PrEP -> AHS feedback values (reporting/sources.py FEEDBACK): -0.075
(affirming care), 0 (the reference) and +0.15 (best fit to the wording-adjusted trends).

Table 1  Pooled over the primary states: infections averted 2025-2036 (number and percent of baseline) and the
         change in 2036 incidence, median and 95% interval, for halving AHS from its 2024 level, holding AHS at
         or below its 2024 level, stigma-resilient services (half the PrEP gap for men with AHS closed, AHS
         unchanged), a PrEP stall (PrEP use among men without AHS held at its 2024 level) and a worsening stigma
         climate (general social and family stigma raised by a sustained 2022-sized rise).
Fig. 3   Where halving AHS helps most, and why, by state: (A) infections averted per 100,000 MSM per year,
         2025-2036, each feedback value, median and middle 50% of draws; (B-D, one per feedback value) the same
         against AHS percentage points removed by halving (2036), shaded by baseline infections per 100,000 MSM
         per year, on shared axes; the states with the most infections averted per capita labelled.
Fig. S   Breakdown of Fig. 3A by state (2025-2036): (A) infections averted per 100,000 MSM per year;
         (B) baseline infections per 100,000 MSM per year; (C) AHS percentage points removed by halving
         (2036); (D) PrEP gain (percentage points, 2036) per AHS point removed. Per draw,
         A = B x C x D x infections averted per PrEP point, the last nearly constant across states. A and C
         by feedback value; B and D for the reference (no feedback). Median and middle 50% of draws.
Table S  By state and feedback value, halving AHS: change in 2036 incidence (%) and infections averted
         2025-2036 (number, percent of baseline, per 100,000 MSM per year), median and 95% interval; observed
         2023-24 AHS and projected 2036 AHS without intervention (median, 95% interval), and the share of draws
         whose projected 2036 AHS exceeds 30% and 50% (above any level observed in a state). Rate denominators: fixed state MSM population
         (inputs/msm_state_population.csv) x 12 years.

Outputs: results/paper/main/: table1_effects.csv, fig3_state_effects.{png,pdf}. results/paper/supplement/:
figS_state_breakdown.{png,pdf}, tableS_state_effects.csv. results/paper/source_data/:
fig3_state_breakdown.csv (all measures and factors, median, interquartile range and 95% interval);
fig3_variance_shares.csv (share of the across-state variance in log rate attributable to each factor);
projected_ahs_by_state.csv; ahs_prep_association.csv.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reporting import sources, style  # noqa: E402
from reporting.summaries import END_YEAR, interval, pooled_effect, primary_states, projected_ahs  # noqa: E402

MAIN, SUPPLEMENT, DATA = sources.MAIN, sources.SUPPLEMENT, sources.SOURCE_DATA
SCENARIOS = {
    "halve_ahs": "Halve AHS from its 2024 level",
    "ahs_no_worsening": "Hold AHS at or below its 2024 level",
    "resilient_services": "Stigma-resilient services",
    "prep_stall": "PrEP stall",
    "worsening_climate": "Worsening stigma climate",
}
FEEDBACK = sources.FEEDBACK
REFERENCE = "0"


def offsets(n: int, step: float = 0.25) -> np.ndarray:
    """Vertical offsets for n feedback values around each state's row (first value on top)."""
    return step * (np.arange(n) - (n - 1) / 2)


def table1() -> pd.DataFrame:
    rows = []
    for code, label in SCENARIOS.items():
        for feedback, path in FEEDBACK.items():
            rows.append({"scenario": label, "feedback": sources.FEEDBACK_VALUES[feedback], **pooled_effect(path, code)})
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
    out.index.names = ["feedback", "state", "measure"]
    return out.sort_index()


TABLE_MEASURES = {"incidence_pct": "incidence_change_2036_pct", "averted": "infections_averted",
                  "averted_pct": "infections_averted_pct", "rate": "averted_per_100k_msm_year"}


def supplement_table(table: pd.DataFrame, projected: pd.DataFrame) -> pd.DataFrame:
    """One row per feedback value and state: effects (median, 95% interval) and projected-AHS shares."""
    rows = []
    for feedback in FEEDBACK:
        t = table.loc[feedback]
        for state in primary_states():
            row = {"feedback": sources.FEEDBACK_VALUES[feedback], "state": state}
            for measure, name in TABLE_MEASURES.items():
                row.update({f"{name}_{k}": t.loc[(state, measure), k] for k in ("median", "low", "high")})
            row["ahs_2024_observed"] = projected.loc[state, "observed_2024"]
            row.update({f"ahs_2036_{k}": projected.loc[state, f"{feedback}_2036{sfx}"]
                        for k, sfx in (("median", ""), ("low", "_low"), ("high", "_high"))})
            row["share_ahs_2036_above_30"] = projected.loc[state, f"{feedback}_share_above_30"]
            row["share_ahs_2036_above_50"] = projected.loc[state, f"{feedback}_share_above_50"]
            rows.append(row)
    return pd.DataFrame(rows)


def variance_shares(table: pd.DataFrame) -> pd.DataFrame:
    """Share of the across-state variance in log rate (medians) attributable to each factor (covariance
    with the total over its variance); the remainder is infections averted per PrEP point."""
    out = {}
    for feedback in FEEDBACK:
        m = np.log(table.loc[feedback, "median"].unstack("measure")[list(FACTORS)].clip(lower=1e-6))
        out[feedback] = {f: float(np.cov(m[f], m["rate"])[0, 1] / m["rate"].var()) for f in list(FACTORS)[1:]}
    return pd.DataFrame(out)


N_LABELLED = 3     # states labelled in Fig. 3B-D: the most infections averted per capita (reference)


def state_effects_figure(table: pd.DataFrame) -> plt.Figure:
    """A: infections averted per 100,000 MSM per year by state, each feedback value; B-D: the same against AHS
    points removed, one panel per feedback value on shared axes, shaded by baseline infections per 100,000."""
    style.apply()
    q = {p: table.loc[p].unstack("measure") for p in FEEDBACK}
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(len(FEEDBACK), 2, width_ratios=[1, 1.15], wspace=0.3, hspace=0.45)
    a = fig.add_subplot(gs[:, 0])
    order = list(q[REFERENCE]["median"]["rate"].sort_values(ascending=False).index)
    y = np.arange(len(order))
    for p, dy in zip(FEEDBACK, offsets(len(FEEDBACK))):
        d = {k: q[p][k]["rate"].loc[order] for k in ("median", "p25", "p75")}
        a.hlines(y + dy, d["p25"], d["p75"], color=style.COLORS[p], lw=2.2, alpha=0.5)
        a.scatter(d["median"], y + dy, marker=style.MARKERS[p], s=18, color=style.COLORS[p], edgecolor="white", lw=0.5, zorder=3)
    a.set_yticks(y)
    a.set_yticklabels(order)
    a.invert_yaxis()
    a.set_xlim(0, None)
    a.grid(axis="y", visible=False)
    a.set_title("A  Infections averted per 100,000\nMSM per year, 2025–2036", loc="left", fontsize=7.8, fontweight="bold")
    a.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], color=style.COLORS[p], marker=style.MARKERS[p], ls="", ms=5) for p in FEEDBACK]
    handles += [plt.Line2D([], [], color=style.MUTED, lw=2.2, alpha=0.5)]
    a.legend(handles, [style.LABELS[p] for p in FEEDBACK] + ["Middle 50% of draws"], loc="lower right", fontsize=6.3)

    burden = q[REFERENCE]["median"]["burden"]
    cmap = LinearSegmentedColormap.from_list("burden", ["#d9d9d9", style.INK])
    norm = Normalize(burden.min(), burden.max())
    xmax = 1.15 * max(q[p]["median"]["gap"].max() for p in FEEDBACK)
    ymax = 1.15 * max(q[p]["median"]["rate"].max() for p in FEEDBACK)
    labelled = order[:N_LABELLED]
    axes = []
    for i, p in enumerate(FEEDBACK):
        b = fig.add_subplot(gs[i, 1])
        axes.append(b)
        m = q[p]["median"]
        b.scatter(m["gap"], m["rate"], c=cmap(norm(burden.loc[m.index])), marker=style.MARKERS[p], s=28,
                  edgecolor=style.COLORS[p], lw=0.9, zorder=3)
        b.set_xlim(0, xmax)
        b.set_ylim(0, ymax)
        for st in labelled:
            b.annotate(st, (m.loc[st, "gap"], m.loc[st, "rate"]), xytext=(4, 2), textcoords="offset points",
                       fontsize=6.5, color=style.INK)
        b.tick_params(labelsize=7)
        b.set_ylabel("Infections averted\nper 100,000 per year", fontsize=6.8)
        b.set_title(f"{'BCD'[i]}  {style.LABELS[p]}", loc="left", fontsize=7.6, color=style.COLORS[p], fontweight="bold")
    axes[-1].set_xlabel(f"AHS percentage points removed by halving ({END_YEAR})", fontsize=7.2)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal", fraction=0.03,
                      pad=0.11, aspect=30)
    cb.set_label("Baseline infections per 100,000 MSM per year", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.5)
    fig.text(0.5, 0.05, "B–D: state medians, same axes. Infections averted are 2025–"
             f"{END_YEAR} averages; AHS points removed are at {END_YEAR}.", ha="center", fontsize=6.3, color=style.MUTED)
    return fig


def breakdown_figure(table: pd.DataFrame) -> plt.Figure:
    style.apply()
    q = {p: table.loc[p].unstack("measure") for p in FEEDBACK}
    order = list(q[REFERENCE]["median"]["rate"].sort_values(ascending=False).index)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 6.0), sharey=True, gridspec_kw={"width_ratios": [1.5, 0.9, 1.2, 0.9], "wspace": 0.22})
    panels = [("rate", "A  Infections averted\nper 100,000 MSM\nper year", True), ("burden", "B  Baseline\ninfections per\n100,000 MSM/year", False),
              ("gap", "C  AHS points\nremoved by\nhalving", True), ("prep_per_point", "D  PrEP gain\nper AHS point\nremoved", False)]
    for ax, (col, title, both) in zip(axes, panels):
        runs = zip(FEEDBACK, offsets(len(FEEDBACK))) if both else ((REFERENCE, 0.0),)
        for feedback, dy in runs:
            d = {k: q[feedback][k][col].loc[order] for k in ("median", "p25", "p75")}
            color = style.COLORS[feedback] if both else style.OBSERVED
            ax.hlines(y + dy, d["p25"], d["p75"], color=color, lw=2.2, alpha=0.5)
            ax.scatter(d["median"], y + dy, marker=style.MARKERS[feedback] if both else "D", s=16 if both else 14,
                       color=color, edgecolor="white", lw=0.5, zorder=3)
        ax.set_title(title, loc="left", fontsize=7.8, fontweight="bold" if col == "rate" else "normal")
        ax.set_xlim(0, None)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=style.COLORS[p], marker=style.MARKERS[p], ls="", ms=5) for p in FEEDBACK]
    handles += [plt.Line2D([], [], color=style.OBSERVED, marker="D", ls="", ms=4), plt.Line2D([], [], color=style.MUTED, lw=2.2, alpha=0.5)]
    fig.legend(handles, [style.LABELS[p] for p in FEEDBACK] + ["No feedback (B, D)", "Middle 50% of draws"],
               loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.04))
    return fig


def main() -> None:
    for d in (MAIN, SUPPLEMENT, DATA):
        d.mkdir(parents=True, exist_ok=True)
    t1 = table1()
    t1.to_csv(MAIN / "table1_effects.csv", index=False)
    association().to_csv(DATA / "ahs_prep_association.csv", index=False)
    table = state_table({p: state_draws(path) for p, path in FEEDBACK.items()})
    table.to_csv(DATA / "fig3_state_breakdown.csv")
    shares = variance_shares(table)
    shares.to_csv(DATA / "fig3_variance_shares.csv", index_label="factor")
    style.save(state_effects_figure(table), MAIN / "fig3_state_effects")
    style.save(breakdown_figure(table), SUPPLEMENT / "figS_state_breakdown")
    projected = projected_ahs()
    projected.to_csv(DATA / "projected_ahs_by_state.csv")
    supplement_table(table, projected).to_csv(SUPPLEMENT / "tableS_state_effects.csv", index=False)
    cols = ["scenario", "feedback", "averted_median", "averted_low", "averted_high", "averted_pct_median",
            "incidence_pct_median", "incidence_pct_low", "incidence_pct_high"]
    print(t1[cols].round(2).to_string(index=False))
    print(shares.round(2).to_string())


if __name__ == "__main__":
    main()
