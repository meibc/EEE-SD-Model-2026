#!/usr/bin/env python3
from __future__ import annotations

"""Results, section 2: projections over time (Fig. 2).

Fig. 2   (A) Anticipated healthcare stigma (AHS), across-state median, 2024-2036: no intervention with the
         fitted PrEP -> AHS relationship continued and not continued, and AHS halved; the shading is the gap
         halving closes. (B) Reduction in annual HIV incidence, pooled over the primary states, relative to
         the paired baseline, for halving AHS, stigma-resilient PrEP delivery and halving general social and
         family stigma; relationship continued and not continued; median and 95% interval across draws.
         Projections start from observed 2023-24 values; interventions ramp in from 2026 to full strength
         in 2032. Values are reported at each two-year step of the model.

Outputs (results/sections/projections/): fig2_projections.{png,pdf}; ahs_trajectories.csv;
incidence_trajectories.csv.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reporting import sources, style  # noqa: E402
from reporting.summaries import interval, primary_states, projected_ahs  # noqa: E402

OUT = sources.SECTIONS / "projections"
START_YEAR = 2024
SCENARIOS = {"halve_ahs": "Halve AHS", "resilient_prep_delivery": "Stigma-resilient PrEP delivery",
             "halve_upstream_stigma": "Halve social and family stigma"}
PROJECTIONS = {"continued": sources.CONTINUED, "not_continued": sources.NOT_CONTINUED}
SHORT = {"continued": "Relationship continued", "not_continued": "Relationship not continued"}


def _summarize(values: pd.Series, keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame.from_dict({k: interval(g) for k, g in values.groupby(keys)}, orient="index").rename_axis(keys)


def ahs_trajectories() -> pd.DataFrame:
    """Across-state median AHS (percent) per draw, then summarized across draws, by projection, year and arm
    (no intervention, AHS halved)."""
    states = primary_states()
    out = {}
    for projection, path in PROJECTIONS.items():
        s = pd.read_csv(path / "scenario_endpoint_sem_draws.csv")
        s = s[s["unit_id"].isin(states) & s["variable"].eq("stigma_ahs") & s["scenario"].eq("halve_ahs")]
        per_draw = 100 * s.groupby(["year", "draw"])[["baseline", "scenario_value"]].median()
        out[projection] = pd.concat({arm: _summarize(per_draw[col], ["year"]) for arm, col in
                                     {"no_intervention": "baseline", "ahs_halved": "scenario_value"}.items()}, names=["arm"])
    return pd.concat(out, names=["projection"])


def incidence_trajectories() -> pd.DataFrame:
    """Reduction (percent) in annual incidence summed over the primary states, per draw relative to the paired
    baseline, by projection, scenario and year."""
    states = primary_states()
    out = {}
    for projection, path in PROJECTIONS.items():
        e = pd.read_csv(path / "scenario_endpoint_epi_draws.csv")
        e = e[e["unit_id"].isin(states) & e["variable"].eq("incidence") & e["scenario"].isin(SCENARIOS)]
        t = e.groupby(["scenario", "year", "draw"])[["baseline", "scenario_value"]].sum()
        out[projection] = _summarize(100 * (1 - t["scenario_value"] / t["baseline"]), ["scenario", "year"])
    return pd.concat(out, names=["projection"])


def figure(ahs: pd.DataFrame, incidence: pd.DataFrame, observed: float) -> plt.Figure:
    style.apply()
    colors = {"continued": style.CONTINUED, "not_continued": style.NOT_CONTINUED}
    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.75, 1], left=0.09, right=0.98, top=0.93, bottom=0.17,
                            hspace=0.75, wspace=0.12)
    a = fig.add_subplot(grid[0, :])
    for projection in PROJECTIONS:
        base = ahs.loc[(projection, "no_intervention")]["median"]
        halved = ahs.loc[(projection, "ahs_halved")]["median"]
        x = [START_YEAR] + list(base.index)
        yb, yh = [observed] + list(base), [observed] + list(halved)
        a.fill_between(x, yh, yb, color=colors[projection], alpha=0.10, lw=0)
        a.plot(x, yb, color=colors[projection], lw=1.8, marker=style.MARKERS[projection], ms=3.5)
        a.plot(x, yh, color=colors[projection], lw=1.2, ls="--", marker=style.MARKERS[projection], ms=3, mfc="white")
        a.text(x[-1] + 0.4, yb[-1], "No intervention", fontsize=7, va="center", color=colors[projection])
    a.text(x[-1] + 0.4, yh[-1], "AHS halved", fontsize=7, va="center", color=style.MUTED)
    a.set_xlim(START_YEAR - 0.5, 2038.3)
    a.set_xticks(range(START_YEAR, 2037, 2))
    a.set_ylim(0, 16)
    a.set_ylabel("AHS (%)")
    a.set_title("A  Anticipated healthcare stigma, across-state median", loc="left", fontweight="bold")
    axes = [fig.add_subplot(grid[1, i]) for i in range(len(SCENARIOS))]
    for ax, (code, label) in zip(axes, SCENARIOS.items()):
        for projection in PROJECTIONS:
            d = incidence.loc[(projection, code)]
            x = [START_YEAR] + list(d.index)
            ax.fill_between(x, [0] + list(d["low"]), [0] + list(d["high"]), color=colors[projection], alpha=0.12, lw=0)
            ax.plot(x, [0] + list(d["median"]), color=colors[projection], lw=1.8, marker=style.MARKERS[projection], ms=3.5)
        ax.set_title(label, loc="left", fontsize=8)
        ax.set_ylim(0, 15)
        ax.set_xticks(range(START_YEAR, 2037, 4))
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("Reduction in annual\nHIV incidence (%)")
    fig.text(0.01, 0.575, "B  Effect on HIV incidence, pooled across states", fontweight="bold", fontsize=9)
    handles = [plt.Line2D([], [], color=colors[p], marker=style.MARKERS[p], lw=1.8, ms=4) for p in PROJECTIONS]
    fig.legend(handles, [SHORT[p] for p in PROJECTIONS], loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.05))
    fig.text(0.5, 0.01, "Medians; B shaded: 95% of draws. Interventions ramp in from 2026 and reach full strength in 2032.",
             ha="center", fontsize=7, color=style.MUTED)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ahs, incidence = ahs_trajectories(), incidence_trajectories()
    ahs.to_csv(OUT / "ahs_trajectories.csv")
    incidence.to_csv(OUT / "incidence_trajectories.csv")
    observed = float(projected_ahs()["observed_2024"].median())
    style.save(figure(ahs, incidence, observed), OUT / "fig2_projections")
    print(incidence["median"].unstack("year").round(1).to_string())


if __name__ == "__main__":
    main()
