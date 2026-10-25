#!/usr/bin/env python3
from __future__ import annotations

"""Weighted stigma and PrEP trajectories, and cross-sectional associations, by demographic group.

Descriptive companion to the state analysis; no group-specific dynamic models.

- Trajectories: national age-PrEP weighted estimates for six marginal groups
  (Black non-Hispanic, Hispanic/Latino, White non-Hispanic; ages <=24, 25-34,
  >=35) over the observed AMIS waves, with 95% intervals from the 500
  respondent-bootstrap replicates (scripts/06_group_bootstrap.R). Estimates are the ESS-weighted
  local logit smooth used for these groups, so endpoint years borrow from the
  adjacent wave.
- Change 2017 -> 2024 with bootstrap intervals.
- Cross-sectional associations: survey-weighted quasibinomial regressions within
  each group (scripts/05_group_associations.R), reported as odds ratios.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

GROUPS = {
    "Race and ethnicity": ["black_nh", "hispanic_latino", "white_nh"],
    "Age": ["age_under_24", "age_25_34", "age_35_plus"],
}
LABELS = {
    "black_nh": "Black, non-Hispanic", "hispanic_latino": "Hispanic/Latino", "white_nh": "White, non-Hispanic",
    "age_under_24": "Age ≤24", "age_25_34": "Age 25–34", "age_35_plus": "Age ≥35",
}
VARIABLES = {
    "prep_used": "Past-year PrEP use",
    "stigma_ahs": "Anticipated healthcare stigma",
    "stigma_gss": "General social stigma",
    "stigma_family": "Family stigma",
}
RELATIONSHIPS = [
    "stigma_ahs -> prep_used", "stigma_ahs -> out_gid", "stigma_ahs -> hivtest12",
    "stigma_ahs -> seehcp", "out_gid -> prep_used", "hivtest12 -> prep_used",
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def trajectories(draws: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    q = draws.groupby(["population_id", "variable", "Year"])["value"].quantile([0.025, 0.975]).unstack()
    q.columns = ["p025", "p975"]
    est = points.set_index(["population_id", "variable", "Year"])["value"].rename("estimate")
    return pd.concat([est, q], axis=1).reset_index()


def changes(draws: pd.DataFrame, points: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    wide = draws.pivot_table(index=["population_id", "variable", "draw"], columns="Year", values="value")
    delta = 100 * (wide[last] - wide[first])
    q = delta.groupby(["population_id", "variable"]).quantile([0.025, 0.975]).unstack()
    q.columns = ["change_p025", "change_p975"]
    pts = points.pivot_table(index=["population_id", "variable"], columns="Year", values="value")
    out = pd.DataFrame({f"estimate_{first}": 100 * pts[first], f"estimate_{last}": 100 * pts[last],
                        "change_pp": 100 * (pts[last] - pts[first])}).join(q)
    out["interval_excludes_zero"] = (out["change_p025"] > 0) | (out["change_p975"] < 0)
    return out.reset_index()


def associations(workbook: Path) -> pd.DataFrame:
    terms = pd.read_excel(workbook, sheet_name="key_terms")
    terms = terms[terms["relationship"].isin(RELATIONSHIPS)]
    return terms[["population_id", "population_label", "relationship", "odds_ratio", "conf_low_or",
                  "conf_high_or", "p.value", "n_complete"]].sort_values(["relationship", "population_id"])


def plot(traj: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(len(GROUPS), len(VARIABLES), figsize=(15, 7.5), sharex=True)
    for r, (row_label, pops) in enumerate(GROUPS.items()):
        for c, (var, title) in enumerate(VARIABLES.items()):
            ax = axes[r, c]
            ends = []
            for pop, color in zip(pops, COLORS):
                t = traj[(traj.population_id == pop) & (traj.variable == var)].sort_values("Year")
                ax.fill_between(t.Year, 100 * t.p025, 100 * t.p975, color=color, alpha=0.15, linewidth=0)
                ax.plot(t.Year, 100 * t.estimate, color=color, linewidth=2, marker="o", markersize=4)
                ends.append([100 * t.estimate.iloc[-1], LABELS[pop], t.Year.iloc[-1]])
            # Spread end labels so they do not overlap (minimum gap in data units).
            lo, hi = ax.get_ylim()
            gap = 0.06 * (hi - lo)
            ends.sort()
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
            for y, label, year in ends:
                ax.annotate(label, (year, y), xytext=(5, 0), textcoords="offset points",
                            fontsize=8, color=INK, va="center")
            if r == 0:
                ax.set_title(title, loc="left", fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"{row_label}\n(weighted %)", color=INK)
            ax.set_xlim(2016.5, 2027.5)
            ax.set_xticks([2017, 2019, 2021, 2024])
            ax.grid(color=GRID, linewidth=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID)
    fig.text(0.01, 0.01, "Lines: weighted estimates (local smooth across adjacent AMIS waves; no 2020 wave); "
             "bands: 95% intervals across 500 respondent-bootstrap replicates.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-dir", type=Path, default=RESULTS / "02_inputs/group_bootstrap")
    parser.add_argument("--regressions", type=Path,
                        default=RESULTS / "02_inputs/group_associations.xlsx")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "08_demographic")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pops = [p for group in GROUPS.values() for p in group]
    draws = pd.read_csv(args.bootstrap_dir / "group_draws.csv.gz")
    points = pd.read_csv(args.bootstrap_dir / "group_trajectories.csv")
    draws = draws[draws.population_id.isin(pops)]
    points = points[points.population_id.isin(pops)]

    traj = trajectories(draws, points)
    years = sorted(points.Year.unique())
    change = changes(draws, points, years[0], years[-1])
    assoc = associations(args.regressions)

    traj.to_csv(args.output_dir / "group_trajectories.csv", index=False)
    change.to_csv(args.output_dir / "group_change_2017_2024.csv", index=False)
    assoc.to_csv(args.output_dir / "group_cross_sectional_associations.csv", index=False)
    plot(traj, args.output_dir / "group_trajectories.png")

    show = change[change.variable.isin(VARIABLES)].copy()
    show["change"] = show.apply(lambda r: f"{r.change_pp:+.1f} ({r.change_p025:+.1f} to {r.change_p975:+.1f})", axis=1)
    print(show.pivot(index="population_id", columns="variable", values="change").loc[pops].to_string())
    print(assoc.pivot(index="population_id", columns="relationship", values="odds_ratio").loc[pops].round(2).to_string())
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
