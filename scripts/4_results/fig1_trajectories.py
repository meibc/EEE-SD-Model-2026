#!/usr/bin/env python3
"""Results, section 1: observed stigma and prevention trajectories (Fig. 1).

Fig. 1   National trends in the eight modeled measures by AMIS cycle, 2017-2024 (national weights,
         scripts/1_data/07; survey-weighted 95% intervals; scripts/2_evidence/02), one panel per measure, grouped
         as stigma, disclosure and care, and prevention and risk. Every panel spans the same number of percentage
         points (SPAN), centred on its own series, so slopes are comparable across measures; the panels are in
         the order of the groups (stigma; disclosure and care; prevention and risk). The AHS panel shows
         AHS as measured and adjusted for the 2022 item change (the model's input), with experienced healthcare
         stigma, whose items did not change; the shading marks the reworded cycles. Each panel gives the change
         2017 to 2024 in percentage points.
Fig. S   Change 2017-18 to 2023-24 in the eight measures (AHS wording-adjusted, as in the model) by primary
         state: state x measure heatmap, cells colored when the 95% interval of the change across
         respondent-bootstrap draws excludes zero and grey otherwise; bottom row the median across states.

Outputs: results/paper/main/fig1_trajectories.{png,pdf}, results/paper/supplement/figS_state_changes.{png,pdf};
in results/paper/source_data/, fig1_national.csv (the plotted national series), fig1_state_changes.csv (level in
2018 and 2024 and the change with its 95% interval, by measure and state) and fig1_summary.csv (across-state median
and the number of states with an increase, a decrease or no clear change, by measure).
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reporting import sources, style  # noqa: E402
from reporting.summaries import observed_change  # noqa: E402

MAIN, SUPP, DATA = sources.MAIN, sources.SUPPLEMENT, sources.SOURCE_DATA
MEASURES = ["stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "risk_behavior", "prep_used", "hivtest12"]
SHORT_LABELS = {"stigma_ahs": "Anticipated\nhealthcare\nstigma", "stigma_gss": "General\nsocial\nstigma",
                "stigma_family": "Family\nstigma", "out_gid": "Disclosure\nto provider", "seehcp": "Provider\nvisit\n(past year)",
                "risk_behavior": "HIV-related\nrisk\nbehavior", "prep_used": "Past-year\nPrEP use", "hivtest12": "HIV\ntesting"}
PANEL_TITLES = {"stigma_ahs": "Anticipated\nhealthcare stigma", "stigma_gss": "General\nsocial stigma",
                "stigma_family": "Family\nstigma", "out_gid": "Disclosure\nto provider",
                "seehcp": "Provider visit\n(past year)", "risk_behavior": "HIV-related\nrisk behavior",
                "prep_used": "PrEP use\n(past year)", "hivtest12": "HIV testing\n(past year)"}
GROUPS = [("Stigma", 0, 3), ("Disclosure and care", 3, 5), ("Prevention and risk", 5, 8)]
SPAN = 30                                 # percentage points shown in every Fig. 1 panel
YEARS = [2017, 2018, 2019, 2021, 2022, 2023, 2024]
YEAR_LABELS = {2017: "’17", 2019: "’19", 2021: "’21", 2024: "’24"}


def signed(x: float) -> str:
    return f"{x:+.1f}".replace("-", "−")
REGIONS = {"Northeast": ["CT", "MA", "NJ", "NY", "PA"], "Midwest": ["IL", "IN", "MI", "MN", "MO", "OH", "WI"],
           "South": ["AL", "FL", "GA", "KY", "LA", "MD", "NC", "SC", "TN", "TX", "VA"], "West": ["AZ", "CA", "CO", "OR", "UT", "WA"]}
INCREASE, DECREASE, UNCLEAR = style.VERMILLION, style.BLUE, "#ececec"
CMAP = LinearSegmentedColormap.from_list("change", [DECREASE, "white", INCREASE])
NORM = TwoSlopeNorm(0, -25, 25)


# =============================================================================
# Fig. 1: national trends
# =============================================================================

def national() -> pd.DataFrame:
    """National series (percent) by survey year: each measure, wording-adjusted AHS and experienced healthcare
    stigma, with 95% intervals."""
    return 100 * pd.read_csv(sources.WORDING / "items_by_year.csv").set_index("survey_year")


def _series(ax, d: pd.DataFrame, col: str, **kw) -> None:
    mid = d[col]
    ax.errorbar(d.index, mid, yerr=[mid - d[f"{col}_low"], d[f"{col}_high"] - mid], capsize=0, elinewidth=0.7,
                **{"ms": 3.5, "marker": "o", "lw": 1.5, "color": style.INK, **kw})


def _limits(d: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    """A SPAN-point window centred on the series (with intervals), kept within 0-100."""
    lo = min(d[f"{c}_low"].min() for c in cols)
    hi = max(d[f"{c}_high"].max() for c in cols)
    centre = (lo + hi) / 2
    bottom = min(max(centre - SPAN / 2, 0), 100 - SPAN)
    return bottom, bottom + SPAN


def national_figure(d: pd.DataFrame) -> plt.Figure:
    style.apply()
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    for ax, m in zip(axes.flat, MEASURES):
        if m == "stigma_ahs":
            ax.axvspan(2021.5, 2024.5, color="#f2f2f2", zorder=0, lw=0)
            _series(ax, d, "ehs", color=style.GREEN, marker="s", ms=3, lw=1.2)
            _series(ax, d.loc[2021:], "stigma_ahs_adjusted", ls="--", mfc="white", lw=1.2)
            _series(ax, d, "stigma_ahs")
            ax.text(2023, 1.0, "Reworded", fontsize=6, color=style.MUTED, ha="center", va="bottom")
            ax.text(2017, d.loc[2017, "stigma_ahs"] + 3.2, "Measured", fontsize=6.2, color=style.INK)
            ax.text(2021.6, 8.6, "Adjusted", fontsize=6.2, color=style.INK, ha="left", va="top")
            ax.text(2017, d.loc[2017, "ehs"] + 2.0, "Experienced", fontsize=6.2, color=style.GREEN)
            ax.set_ylim(*_limits(d, ["stigma_ahs", "ehs"]))
            change = (f"{signed(d.loc[2024, 'stigma_ahs'] - d.loc[2017, 'stigma_ahs'])} measured,\n"
                      f"{signed(d.loc[2024, 'stigma_ahs_adjusted'] - d.loc[2017, 'stigma_ahs'])} adjusted")
            top = True
        else:
            _series(ax, d, m)
            lo, hi = _limits(d, [m])
            ax.set_ylim(lo, hi)
            change = f"{signed(d.loc[2024, m] - d.loc[2017, m])} points"
            top = d[m].mean() < (lo + hi) / 2          # label where the data are not
        ax.set_title(PANEL_TITLES[m], loc="left", fontsize=7.4)
        ax.text(0.03, 0.97 if top else 0.03, f"2017–24: {change}", transform=ax.transAxes, fontsize=6.2,
                color=style.MUTED, va="top" if top else "bottom")
        ax.set_xlim(2016.5, 2024.5)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([YEAR_LABELS.get(y, "") for y in YEARS], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="x", visible=False)
    fig.supylabel("Percent, national", fontsize=8.5, x=0.04)
    fig.text(0.5, -0.01, f"National weights, 95% intervals. Every panel spans {SPAN} percentage points. AMIS did not run in 2020.",
             ha="center", fontsize=6.5, color=style.MUTED)
    return fig


# =============================================================================
# Fig. S: state changes
# =============================================================================

def state_changes() -> pd.DataFrame:
    return pd.concat({m: observed_change(m) for m in MEASURES}, names=["measure"]).reset_index()


def summarize(changes: pd.DataFrame) -> pd.DataFrame:
    counts = changes.groupby(["measure", "direction"]).size().unstack(fill_value=0)
    medians = changes.groupby("measure")[["level_2018", "level_2024", "change"]].median().add_prefix("median_")
    out = medians.join(counts).reindex(MEASURES)
    return out.rename(columns=lambda c: c.replace(" ", "_"))


def heatmap(ax: plt.Axes, changes: pd.DataFrame, states: list[str]) -> None:
    """State x measure change heatmap with a median row (all primary states) and region labels."""
    d = changes.set_index(["state", "measure"])
    regions = {r: [s for s in st if s in states] for r, st in REGIONS.items()}
    order = [s for st in regions.values() for s in st]
    n = len(order)
    for i, s in enumerate(order):
        for j, m in enumerate(MEASURES):
            r = d.loc[(s, m)]
            v, clear = r["change"], r["direction"] != "no clear change"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=CMAP(NORM(v)) if clear else UNCLEAR, lw=0))
            ink = "white" if clear and abs(v) > 14 else style.INK if clear else style.MUTED
            ax.text(j + 0.5, i + 0.5, "0" if round(v) == 0 else f"{v:+.0f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if clear else "normal", color=ink)
    medians = changes.groupby("measure")["change"].median()
    for j, m in enumerate(MEASURES):
        v = medians[m]
        ax.text(j + 0.5, n + 0.75, "0" if round(v) == 0 else f"{v:+.0f}", ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=style.INK)
    ax.plot([0, len(MEASURES)], [n + 0.2, n + 0.2], color=style.MUTED, lw=0.6)
    y = 0
    for r, st in regions.items():
        if st:
            ax.text(-0.72, y + len(st) / 2, r, rotation=90, ha="center", va="center", fontsize=7.5, fontweight="bold")
            y += len(st)
            if y < n:
                ax.axhline(y, color="white", lw=2.5)
    for _, a, _ in GROUPS[1:]:
        ax.axvline(a, color="white", lw=2.5)
    for g, a, b in GROUPS:
        ax.text((a + b) / 2, -0.85, g, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.plot([a + 0.1, b - 0.1], [-0.7, -0.7], color=style.MUTED, lw=0.8, clip_on=False)
    ax.set_xlim(0, len(MEASURES))
    ax.set_ylim(n + 1.3, 0)
    ax.set_yticks([i + 0.5 for i in range(n)] + [n + 0.75])
    ax.set_yticklabels(order + [f"Median, all {changes['state'].nunique()}"], fontsize=7.5)
    ax.set_xticks([j + 0.5 for j in range(len(MEASURES))])
    ax.set_xticklabels([SHORT_LABELS[m] for m in MEASURES], fontsize=7.5)
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def supplement_figure(changes: pd.DataFrame) -> plt.Figure:
    style.apply()
    states = sorted(changes["state"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(states) + 2.2))
    heatmap(ax, changes, states)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, fraction=0.03, pad=0.02,
                      ticks=[-20, -10, 0, 10, 20])
    cb.set_label("Change, percentage points", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    return fig


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    d = national()
    d.to_csv(DATA / "fig1_national.csv")
    style.save(national_figure(d), MAIN / "fig1_trajectories")
    changes = state_changes()
    changes.to_csv(DATA / "fig1_state_changes.csv", index=False)
    summary = summarize(changes)
    summary.to_csv(DATA / "fig1_summary.csv")
    style.save(supplement_figure(changes), SUPP / "figS_state_changes")
    cols = [c for c in d.columns if not c.endswith(("_low", "_high"))]
    print(d.loc[[2017, 2024], cols].round(1).to_string())


if __name__ == "__main__":
    main()
