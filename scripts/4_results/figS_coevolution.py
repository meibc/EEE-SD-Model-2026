#!/usr/bin/env python3
"""Supplement: PrEP-AHS co-evolution behind the fitted relationship (Supplementary figure).

Fig. S   Observed past-year PrEP use and anticipated healthcare stigma (AHS) by state over the four
         periods (2017-18 to 2023-24; median across respondent-bootstrap draws), one panel per state, all
         panels on the same axes.
         (Projected AHS by state is the supplementary figure in scripts/4_results/table1_fig3_ahs_reduction.py.)

Also the fitted PrEP -> AHS relationship by state (transition-matrix entry, centered-logit units per
two-year step) in the primary fit and under the AHS temporal placebo (periods of AHS permuted).

Outputs (results/sections/coevolution/): figS_coevolution.{png,pdf}; observed_paths.csv;
prep_to_ahs.csv (by state, primary and placebo: median, 95% interval, share of
draws above zero); prep_to_ahs_overall.csv.
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
from reporting.summaries import interval, observed_paths, primary_states  # noqa: E402

OUT = sources.SECTIONS / "coevolution"
PERIOD_LABELS = {2018: "2017–18", 2020: "2019–20", 2022: "2021–22", 2024: "2023–24"}


def prep_to_ahs(fit_path: Path) -> pd.DataFrame:
    """Fitted PrEP -> AHS entry by primary state: median, 95% interval, share of draws above zero."""
    z = np.load(fit_path, allow_pickle=True)
    v, units = z["v_names"].astype(str).tolist(), z["unit_ids"].astype(str).tolist()
    coef = z["J"][:, :, v.index("stigma_ahs"), v.index("prep_used")].astype(float)
    rows = {s: {**interval(coef[:, units.index(s)]), "share_positive": float((coef[:, units.index(s)] > 0).mean())}
            for s in primary_states()}
    overall = {**interval(coef[:, [units.index(s) for s in primary_states()]].ravel()),
               "share_positive": float((coef[:, [units.index(s) for s in primary_states()]] > 0).mean())}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("state"), overall


def figure(paths: pd.DataFrame) -> plt.Figure:
    style.apply()
    states = sorted(primary_states())
    fig = plt.figure(figsize=(7.2, 5.2))
    top = fig.add_gridspec(4, 6, left=0.08, right=0.98, top=0.91, bottom=0.1, hspace=0.35, wspace=0.12)
    wide = paths.pivot_table(index=["state", "year"], columns="measure", values="level").reset_index()
    xmax, ymax = wide["prep_used"].max() * 1.08, wide["stigma_ahs"].max() * 1.12
    for i, s in enumerate(states):
        ax = fig.add_subplot(top[i // 6, i % 6])
        d = wide[wide["state"].eq(s)].sort_values("year")
        ax.plot(d["prep_used"], d["stigma_ahs"], color=style.MUTED, lw=1, zorder=1)
        for (x0, y0), (x1, y1) in zip(d[["prep_used", "stigma_ahs"]].values[:-1], d[["prep_used", "stigma_ahs"]].values[1:]):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=style.MUTED, lw=1,
                                                                         mutation_scale=5, shrinkA=2, shrinkB=2))
        ax.scatter(d["prep_used"].iloc[1:-1], d["stigma_ahs"].iloc[1:-1], s=9, color="#9a9a9a", zorder=2, lw=0)
        ax.scatter(d["prep_used"].iloc[[0]], d["stigma_ahs"].iloc[[0]], s=14, color="white", edgecolor=style.INK, zorder=3, lw=0.8)
        ax.scatter(d["prep_used"].iloc[[-1]], d["stigma_ahs"].iloc[[-1]], s=14, color=style.INK, zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.text(0.04, 0.93, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        if i % 6:
            ax.set_yticklabels([])
        if i < 18:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=7)
    fig.text(0.53, 0.02, "Past-year PrEP use (%)", ha="center")
    fig.text(0.012, 0.5, "Anticipated healthcare stigma (%)", rotation=90, va="center")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc="white", mec=style.INK, ms=4, label="2017–18"),
                        plt.Line2D([], [], marker="o", ls="", color="#9a9a9a", ms=3, label="2019–20, 2021–22"),
                        plt.Line2D([], [], marker="o", ls="", color=style.INK, ms=4, label="2023–24")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.55, 0.995), fontsize=8)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = observed_paths(("prep_used", "stigma_ahs"))
    paths.to_csv(OUT / "observed_paths.csv", index=False)
    primary, primary_all = prep_to_ahs(sources.PRIMARY_FIT)
    placebo, placebo_all = prep_to_ahs(sources.SPECIFICATION_FITS / "placebo_ahs.npz")
    pd.concat({"primary": primary, "placebo_ahs": placebo}, names=["fit"]).to_csv(OUT / "prep_to_ahs.csv")
    overall = pd.DataFrame({"primary": primary_all, "placebo_ahs": placebo_all}).T
    overall.to_csv(OUT / "prep_to_ahs_overall.csv", index_label="fit")
    style.save(figure(paths), OUT / "figS_coevolution")
    print(overall.round(3).to_string())


if __name__ == "__main__":
    main()
