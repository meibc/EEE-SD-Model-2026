#!/usr/bin/env python3
"""Results, section 1: observed stigma and prevention trajectories (Fig. 1).

Fig. 1A  2018 and 2024 levels of the eight modeled measures in the primary states, each on its own scale
         (AHS adjusted for the 2022 item change, as in the model). One line per state, colored when the
         95% interval of the 2018-2024 change across respondent-bootstrap draws excludes zero; the black
         line joins the across-state medians.
Fig. 1B  National anticipated healthcare stigma (AHS) by AMIS cycle as measured and as adjusted for the
         2022 item change (the model's input), with experienced healthcare stigma, whose items did not
         change (scripts/2_evidence/02); survey-weighted 95% intervals.

Outputs (results/sections/trajectories/): fig1_trajectories.{png,pdf}; state_changes.csv (level in 2018
and 2024 and the change with its 95% interval, by measure and state); summary.csv (across-state median
and the number of states with an increase, a decrease or no clear change, by measure).
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reporting import sources, style  # noqa: E402
from reporting.summaries import observed_change  # noqa: E402

OUT = sources.SECTIONS / "trajectories"
MEASURES = ["stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "risk_behavior", "prep_used", "hivtest12"]
SHORT_LABELS = {"stigma_ahs": "Anticipated healthcare\nstigma\n(wording-adjusted)", "stigma_gss": "General social\nstigma",
                "stigma_family": "Family\nstigma", "out_gid": "Disclosure to\nprovider", "seehcp": "Healthcare provider\nvisit (past year)",
                "risk_behavior": "HIV-related\nrisk behavior", "prep_used": "Past-year\nPrEP use", "hivtest12": "HIV\ntesting"}
SHARED_SCALE = False   # True puts every panel of Fig. 1A on 0-100% (author prefers separate scales)
DIRECTION_COLORS = {"increase": "#D55E00", "decrease": style.CONTINUED, "no clear change": "#c8c8c8"}


def state_changes() -> pd.DataFrame:
    return pd.concat({m: observed_change(m) for m in MEASURES}, names=["measure"]).reset_index()


def summarize(changes: pd.DataFrame) -> pd.DataFrame:
    counts = changes.groupby(["measure", "direction"]).size().unstack(fill_value=0)
    medians = changes.groupby("measure")[["level_2018", "level_2024", "change"]].median().add_prefix("median_")
    out = medians.join(counts).reindex(MEASURES)
    return out.rename(columns=lambda c: c.replace(" ", "_"))


def figure(changes: pd.DataFrame, by_year: pd.DataFrame) -> plt.Figure:
    style.apply()
    fig = plt.figure(figsize=(7.2, 6.6))
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.95], hspace=0.6, wspace=0.3)
    for i, m in enumerate(MEASURES):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        d = changes[changes["measure"].eq(m)]
        for direction in ["no clear change", "decrease", "increase"]:            # colored lines drawn on top
            for _, r in d[d["direction"].eq(direction)].iterrows():
                ax.plot([2018, 2024], [r["level_2018"], r["level_2024"]], color=DIRECTION_COLORS[direction],
                        lw=0.9, alpha=0.9, solid_capstyle="round")
        ax.plot([2018, 2024], [d["level_2018"].median(), d["level_2024"].median()], color=style.INK, lw=2.2)
        ax.set_title(SHORT_LABELS[m], loc="left", fontsize=8.5)
        ax.set_xticks([2018, 2024])
        ax.set_xlim(2017.4, 2024.6)
        if SHARED_SCALE:
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            if i % 4:
                ax.set_yticklabels([])
        else:
            ax.set_ylim(bottom=0)
        if i % 4 == 0:
            ax.set_ylabel("Percent")
        ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], color=DIRECTION_COLORS[k], lw=1.5, label=k.capitalize()) for k in ["increase", "decrease", "no clear change"]]
    handles.append(plt.Line2D([], [], color=style.INK, lw=2.2, label="Across-state median"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995), fontsize=8)
    fig.text(0.01, 0.985, "A", fontweight="bold", fontsize=11)

    ax = fig.add_subplot(grid[2, :])
    ax.axvspan(2021.5, 2024.5, color="#f2f2f2", zorder=0)
    adjusted = by_year[by_year["survey_year"].ge(2021)]                 # identical to the measured series before 2022
    series = [(by_year, "stigma_ahs", dict(color=style.CONTINUED, marker="o", lw=1.6), "AHS, measured", -0.06),
              (adjusted, "stigma_ahs_adjusted", dict(color=style.CONTINUED, marker="o", lw=1.2, ls="--", mfc="white"),
               "AHS, wording-adjusted (as in A)", 0.06),
              (by_year, "ehs", dict(color=style.ACCENT, marker="s", lw=1.6), "Experienced healthcare stigma", 0.0)]
    for data, col, kw, label, dx in series:
        x, mid = data["survey_year"] + dx, 100 * data[col]
        ax.errorbar(x, mid, yerr=[mid - 100 * data[f"{col}_low"], 100 * data[f"{col}_high"] - mid], ms=4, capsize=0,
                    elinewidth=0.8, label=label, **kw)
    ax.set_xticks([2017, 2018, 2019, 2021, 2022, 2023, 2024])
    ax.set_xlim(2016.6, 2024.5)
    ax.set_ylim(0, None)
    ax.set_ylabel("Percent, national")
    ax.text(2023, 0.6, "AHS items reworded", fontsize=7, color=style.MUTED, ha="center", va="bottom")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5, ncol=3, handlelength=2.4, columnspacing=1.2)
    fig.text(0.01, 0.31, "B", fontweight="bold", fontsize=11)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    changes = state_changes()
    changes.to_csv(OUT / "state_changes.csv", index=False)
    summary = summarize(changes)
    summary.to_csv(OUT / "summary.csv")
    by_year = pd.read_csv(sources.WORDING / "items_by_year.csv")
    style.save(figure(changes, by_year), OUT / "fig1_trajectories")
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
