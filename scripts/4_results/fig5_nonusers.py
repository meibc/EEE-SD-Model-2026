#!/usr/bin/env python3
"""Results: who remains without PrEP as it expands (Fig. 5).

With PrEP use among men with AHS a fraction r of that among men without (r < 1), men reporting AHS make up a
growing share of the men still not using PrEP as PrEP expands:
    Pr(AHS | no PrEP) = A (1 - r P0) / (1 - P),   P = P0 (1 - A (1 - r))      (models/dynamic/mixture.py)
With r held constant the widening follows from the mixture; what the projection adds is its size. The observed
national AMIS cycles (scripts/2_evidence/05) show whether it has been widening: reported in paper_numbers, not
plotted.

Fig. 5   Primary states combined (MSM-weighted, per draw, each draw's own r): AHS among all men and among PrEP
         non-users, 2017-18 to 2023-24 from the observed inputs (wording-adjusted) and projected to 2036 without
         intervention, against PrEP use; the reference (no PrEP -> AHS feedback). Medians across draws; the
         other feedback values and the AHS-halved scenario are in fig5_nonusers_projected.csv (and paper_numbers).

Outputs: results/paper/main/fig5_nonusers.{png,pdf}; in results/paper/source_data/, fig5_nonusers_observed.csv
(national AMIS, by cycle) and fig5_nonusers_projected.csv (by run, year and measure: median and intervals).
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.projection import N_SAMPLES, SEED  # noqa: E402
from reporting import sources, style  # noqa: E402
from reporting.summaries import interval, primary_states  # noqa: E402

MAIN, DATA = sources.MAIN, sources.SOURCE_DATA
OBSERVED = sources.RESULTS / "02_inputs/ahs_among_nonusers.csv"
START_YEAR = 2024
PERIOD_LABELS = {2018: "2017–18", 2020: "2019–20", 2022: "2021–22", 2024: "2023–24"}


def composition(A: np.ndarray, P: np.ndarray, r: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per draw (rows), pooled over states (columns, weights w): AHS among all men and among PrEP non-users."""
    p0 = P / (1 - A * (1 - r))
    ahs_nonuser = A * (1 - r * p0)                 # share of all men who report AHS and do not use PrEP
    return (A * w).sum(1) / w.sum(), (ahs_nonuser * w).sum(1) / ((1 - P) * w).sum(1)


def projected() -> pd.DataFrame:
    """AHS among all men and among non-users (percent), their ratio and PrEP, primary states combined, per draw
    and year: observed periods (the same for every run), then each run's projection."""
    bundle = np.load(sources.REFERENCE_BUNDLE, allow_pickle=True)
    geo, v = bundle["geo_ids"].astype(str).tolist(), bundle["sem_v_names"].astype(str).tolist()
    years = [int(y) for y in bundle["sem_obs_years"]]
    states = primary_states()
    sx = [geo.index(s) for s in states]
    w = pd.read_csv(sources.MSM_POPULATION).set_index("state")["median"].reindex(states).to_numpy()
    rng = np.random.default_rng(SEED)              # the scenario runner's pairing (pipeline/projection.py)
    n_dyn = bundle["sem_J_samples"].shape[0]
    d_idx = rng.choice(n_dyn, size=N_SAMPLES, replace=N_SAMPLES > n_dyn)
    r = np.asarray(bundle["sem_ahs_prep_apr"], float)[d_idx][:, sx]                       # sample x state
    obs = np.asarray(bundle["sem_obs_samples"], float)[d_idx][:, sx]                      # sample x state x m x T
    series = {("observed", y): (obs[:, :, v.index("stigma_ahs"), t], obs[:, :, v.index("prep_used"), t])
              for t, y in enumerate(years)}
    for f, path in sources.FEEDBACK.items():
        d = pd.read_csv(path / "scenario_endpoint_dynamic_draws.csv.gz")
        d = d[d["scenario"].eq("halve_ahs") & d["unit_id"].isin(states)]
        for arm, col in (("no_intervention", "baseline"), ("ahs_halved", "scenario_value")):
            for y, g in d.groupby("year"):
                wide = g.pivot_table(index="draw", columns=["variable", "unit_id"], values=col)
                series[(f"{f}_{arm}", int(y))] = (wide["stigma_ahs"][states].to_numpy(), wide["prep_used"][states].to_numpy())
    rows = []
    for (run, year), (A, P) in series.items():
        a_all, a_non = composition(A, P, r, w)
        for name, values in (("ahs_all", 100 * a_all), ("ahs_nonusers", 100 * a_non), ("ratio", a_non / a_all),
                             ("prep", 100 * (P * w).sum(1) / w.sum())):
            rows.append({"run": run, "year": year, "measure": name, **interval(values)})
    return pd.DataFrame(rows)


def figure(proj: pd.DataFrame) -> plt.Figure:
    style.apply()
    p = proj.set_index(["run", "measure", "year"])["median"].sort_index()

    def path(run, measure):
        """PrEP use (x) and the measure (y): observed periods, then the run's projection."""
        x = list(p.loc[("observed", "prep")].values) + list(p.loc[(run, "prep")].values)
        y = list(p.loc[("observed", measure)].values) + list(p.loc[(run, measure)].values)
        years = list(p.loc[("observed", "prep")].index) + list(p.loc[(run, "prep")].index)
        return x, y, years

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    c = style.COLORS["0"]
    xa, ya, years = path("0_no_intervention", "ahs_all")
    xn, yn, _ = path("0_no_intervention", "ahs_nonusers")
    k = years.index(START_YEAR)
    ax.axvspan(0, xa[k], color="#f4f4f4", zorder=0, lw=0)
    ax.fill_between(xa, ya, yn, color=c, alpha=0.15, lw=0)
    ax.plot(xa, ya, color=c, lw=1.6, marker="o", ms=4, mfc="white")
    ax.plot(xn, yn, color=c, lw=2.2, marker="o", ms=4)
    for x, y, yr in zip(xn, yn, years):            # years along the non-user line
        if yr in (2018, 2024, 2030, 2036):
            ax.text(x, y + 0.6, PERIOD_LABELS.get(yr, str(yr)), fontsize=style.FONT, color=style.MUTED, ha="center", va="bottom")
    ax.text(xn[-1] + 1.0, yn[-1], "Among PrEP non-users", fontsize=style.FONT, color=c, va="center", fontweight="bold")
    ax.text(xa[-1] + 1.0, ya[-1], "Among all men", fontsize=style.FONT, color=c, va="center")
    ax.text(xa[k] / 2, 1.0, "Observed", fontsize=style.FONT, color=style.MUTED, ha="center")
    ax.text((xa[k] + xa[-1]) / 2, 1.0, "Projected, no intervention", fontsize=style.FONT, color=style.MUTED, ha="center")
    ax.set_xlim(0, 71)
    ax.spines["bottom"].set_bounds(0, 60)
    ax.set_xticks(range(0, 61, 10))
    ax.set_ylim(0, None)
    ax.set_xlabel("PrEP use (%)")
    ax.set_ylabel("Anticipated healthcare stigma (%)")
    ax.grid(axis="x", visible=False)
    return fig


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    pd.read_csv(OBSERVED).to_csv(DATA / "fig5_nonusers_observed.csv", index=False)
    proj = projected()
    proj.to_csv(DATA / "fig5_nonusers_projected.csv", index=False)
    style.save(figure(proj), MAIN / "fig5_nonusers")
    t = proj[proj["year"].isin([2018, 2024, 2036])].pivot_table(index=["run", "year"], columns="measure", values="median", sort=False)
    print(t.round(2).to_string())


if __name__ == "__main__":
    main()
