#!/usr/bin/env python3
"""Results, section 3: projected effects of reducing anticipated healthcare stigma (AHS) (Table 1, Fig. 4).

Every result is given for the three PrEP -> AHS feedback values (reporting/sources.py FEEDBACK): -0.075
(affirming care), 0 (the reference) and +0.15 (best fit to the wording-adjusted trends).

Table 1  Pooled over the primary states: infections averted 2025-2036 (number and percent of baseline) and the
         change in 2036 incidence, median and 95% interval, for halving AHS from its 2024 level, holding AHS at
         or below its 2024 level, stigma-resilient services (half the PrEP gap for men with AHS closed, AHS
         unchanged), a PrEP stall (PrEP use among men without AHS held at its 2024 level) and a worsening stigma
         climate (general social and family stigma raised by a sustained 2022-sized rise).
Fig. 4   Where halving AHS helps most, and why, by state: (A) infections averted per 100,000 MSM per year,
         2025-2036, each feedback value, median and middle 50% of draws, states ordered by the reference;
         (B) the same (no feedback) against the reduction in AHS by 2036 (percentage points), shaded by baseline
         infections per 100,000 MSM per year; every state labelled.
Fig. S   Breakdown of Fig. 4A by state (2025-2036): (A) infections averted per 100,000 MSM per year;
         (B) baseline infections per 100,000 MSM per year; (C) AHS percentage points removed by halving
         (2036); (D) PrEP gain (percentage points, 2036) per AHS point removed. Per draw,
         A = B x C x D x infections averted per PrEP point, the last nearly constant across states. A and C
         by feedback value; B and D for the reference (no feedback). Median and middle 50% of draws.
Table S  By state and feedback value, halving AHS: change in 2036 incidence (%) and infections averted
         2025-2036 (number, percent of baseline, per 100,000 MSM per year), median and 95% interval; observed
         2023-24 AHS and projected 2036 AHS without intervention (median, 95% interval), and the share of draws
         whose projected 2036 AHS exceeds 30% and 50% (above any level observed in a state). Rate denominators: fixed state MSM population
         (inputs/msm_state_population.csv) x 12 years.

Outputs: results/paper/main/: table1_effects.csv, fig4_state_effects.{png,pdf}. results/paper/supplement/:
figS_state_breakdown.{png,pdf}, tableS_state_effects.csv. results/paper/source_data/:
fig4_state_breakdown.csv (all measures and factors, median, interquartile range and 95% interval);
fig4_variance_shares.csv (share of the across-state variance in log rate attributable to each factor);
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
    """Offsets for n feedback values around each state's position (first value first)."""
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


def state_effects_figure(table: pd.DataFrame) -> plt.Figure:
    """A: infections averted per 100,000 MSM per year by state (states along x, most first), each feedback value,
    median and middle 50% of draws; B: the same (no feedback) against the reduction in AHS by 2036, shaded by
    baseline infections per 100,000 MSM per year, every state labelled."""
    style.apply()
    q = {p: table.loc[p].unstack("measure") for p in FEEDBACK}
    fig = plt.figure(figsize=(style.WIDTH, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1], hspace=0.3)
    a = fig.add_subplot(gs[0])
    order = list(q[REFERENCE]["median"]["rate"].sort_values(ascending=False).index)
    x = np.arange(len(order))
    for p, dx in zip(FEEDBACK, offsets(len(FEEDBACK))):
        d = {k: q[p][k]["rate"].loc[order] for k in ("median", "p25", "p75")}
        a.vlines(x + dx, d["p25"], d["p75"], color=style.COLORS[p], lw=2.0, alpha=0.5)
        a.scatter(x + dx, d["median"], marker=style.MARKERS[p], s=16, color=style.COLORS[p], edgecolor="white", lw=0.5, zorder=3)
    a.set_xticks(x)
    a.set_xticklabels(order)
    a.set_xlim(-0.6, len(order) - 0.4)
    a.set_ylim(0, None)
    a.grid(axis="x", visible=False)
    a.tick_params(axis="x", length=0)
    a.set_ylabel("Infections averted per\n100,000 MSM per year")
    a.set_title("a  Infections averted by state", loc="left", fontweight="bold")
    handles = [plt.Line2D([], [], color=style.COLORS[p], marker=style.MARKERS[p], ls="", ms=5) for p in FEEDBACK]
    handles += [plt.Line2D([], [], color=style.MUTED, lw=2.0, alpha=0.5)]
    a.legend(handles, [style.LABELS[p] for p in FEEDBACK] + ["Middle 50% of draws"], loc="upper right",
             title="PrEP→AHS feedback", title_fontsize=style.FONT, alignment="left")

    m = q[REFERENCE]["median"]
    cmap = LinearSegmentedColormap.from_list("burden", ["#d9d9d9", style.INK])
    norm = Normalize(m["burden"].min(), m["burden"].max())
    b = fig.add_subplot(gs[1])
    b.scatter(m["gap"], m["rate"], c=cmap(norm(m["burden"])), marker="o", s=40, edgecolor=style.OBSERVED, lw=0.7, zorder=3)
    b.set_xlim(0, 1.08 * m["gap"].max())
    b.set_ylim(-1.5, 1.1 * m["rate"].max())                  # room below zero for labels
    b.spines["left"].set_bounds(0, 25)
    b.set_yticks(range(0, 26, 5))
    labels = m.drop(index=list(SHARED_LABELS.values()))           # states on the same spot share one label
    label_all(b, labels["gap"].to_numpy(), labels["rate"].to_numpy(),
              [f"{s}, {SHARED_LABELS[s]}" if s in SHARED_LABELS else s for s in labels.index], fixed=FIXED_LABELS)
    b.set_xlabel(f"Reduction in AHS by {END_YEAR} (percentage points)")
    b.set_ylabel("Infections averted per\n100,000 MSM per year")
    b.set_title("b  Infections averted by AHS reduction and baseline incidence (no feedback)", loc="left", fontweight="bold")
    cax = b.inset_axes([0.09, 0.80, 0.3, 0.045])                       # the empty upper left
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="horizontal")
    cb.set_label("Baseline infections per 100,000 MSM per year", labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    return fig


SHARED_LABELS = {"MO": "WA"}      # Fig. 4B: WA sits on MO (within 0.1 on both axes), labelled together
FIXED_LABELS: dict[str, tuple[float, float]] = {}   # hand-set label offsets (points) for crowded states in Fig. 4B


LABEL_POSITIONS = [(5, 0), (-5, 0), (0, 6), (0, -6), (5, 5), (-5, 5), (5, -5), (-5, -5),
                   (12, 0), (-12, 0), (0, 12), (0, -12), (10, 9), (-10, 9), (10, -9), (-10, -9)]   # offsets, points


def label_all(ax, xs, ys, names, fontsize: float = style.FONT, marker_pts: float = 4.5, fixed: dict | None = None) -> None:
    """Label every point: for each (largest y first), the first candidate offset whose text box overlaps no
    placed label and no marker; a leader line when the label sits away from the point. ``fixed``: offsets set by
    hand for crowded points (placed first)."""
    fixed = fixed or {}
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    r = marker_pts * scale
    placed = []
    order = sorted(range(len(names)), key=lambda i: (names[i] not in fixed, -ys[i]))
    for i in order:
        (px, py), name = pts[i], names[i]
        probe = ax.text(0, 0, name, fontsize=fontsize)
        bb = probe.get_window_extent(renderer)
        probe.remove()
        w, h = bb.width + 2, bb.height + 1
        best = None
        for dx, dy in ([fixed[name]] if name in fixed else LABEL_POSITIONS):
            cx, cy = px + dx * scale, py + dy * scale
            x0 = cx if dx > 0 else cx - w if dx < 0 else cx - w / 2
            y0 = cy - h / 2 if dy == 0 else cy if dy > 0 else cy - h
            box = (x0, y0, x0 + w, y0 + h)
            hits_label = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed)
            hits_point = any(box[0] - r < qx < box[2] + r and box[1] - r < qy < box[3] + r
                             for j, (qx, qy) in enumerate(pts) if j != i)
            if name in fixed or (not hits_label and not hits_point):
                best = (dx, dy, box)
                break
        if best is None:
            dx, dy = LABEL_POSITIONS[0]
            best = (dx, dy, (px + dx * scale, py - h / 2, px + dx * scale + w, py + h / 2))
        dx, dy, box = best
        placed.append(box)
        far = abs(dx) > 6 or abs(dy) > 6
        ax.annotate(name, (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                    color=style.INK, va="center" if dy == 0 else "bottom" if dy > 0 else "top",
                    ha="left" if dx > 0 else "right" if dx < 0 else "center",
                    arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.5, shrinkA=0, shrinkB=3) if far else None)


def breakdown_figure(table: pd.DataFrame) -> plt.Figure:
    style.apply()
    q = {p: table.loc[p].unstack("measure") for p in FEEDBACK}
    order = list(q[REFERENCE]["median"]["rate"].sort_values(ascending=False).index)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 4, figsize=(style.WIDTH, 6.0), sharey=True, gridspec_kw={"width_ratios": [1.5, 0.9, 1.2, 0.9], "wspace": 0.22})
    panels = [("rate", "a  Infections averted\nper 100,000 MSM\nper year", True), ("burden", "b  Baseline\ninfections per\n100,000 MSM/year", False),
              ("gap", "c  AHS points\nremoved by\nhalving", True), ("prep_per_point", "d  PrEP gain\nper AHS point\nremoved", False)]
    for ax, (col, title, both) in zip(axes, panels):
        runs = zip(FEEDBACK, offsets(len(FEEDBACK))) if both else ((REFERENCE, 0.0),)
        for feedback, dy in runs:
            d = {k: q[feedback][k][col].loc[order] for k in ("median", "p25", "p75")}
            color = style.COLORS[feedback]
            ax.hlines(y + dy, d["p25"], d["p75"], color=color, lw=2.2, alpha=0.5)
            ax.scatter(d["median"], y + dy, marker=style.MARKERS[feedback], s=16, color=color, edgecolor="white",
                       lw=0.5, zorder=3)
        ax.set_title(title, loc="left", fontsize=style.FONT, fontweight="bold" if col == "rate" else "normal")
        ax.set_xlim(0, None)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=style.FONT)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=style.COLORS[p], marker=style.MARKERS[p], ls="", ms=5) for p in FEEDBACK]
    handles += [plt.Line2D([], [], color=style.MUTED, lw=2.2, alpha=0.5)]
    fig.legend(handles, [style.LABELS[p] for p in FEEDBACK] + ["Middle 50% of draws"],
               title="PrEP→AHS feedback (b, d: no feedback only)", title_fontsize=style.FONT, loc="lower center", ncol=4, fontsize=style.FONT, bbox_to_anchor=(0.5, -0.04))
    return fig


def main() -> None:
    for d in (MAIN, SUPPLEMENT, DATA):
        d.mkdir(parents=True, exist_ok=True)
    t1 = table1()
    t1.to_csv(MAIN / "table1_effects.csv", index=False)
    association().to_csv(DATA / "ahs_prep_association.csv", index=False)
    table = state_table({p: state_draws(path) for p, path in FEEDBACK.items()})
    table.to_csv(DATA / "fig4_state_breakdown.csv")
    shares = variance_shares(table)
    shares.to_csv(DATA / "fig4_variance_shares.csv", index_label="factor")
    style.save(state_effects_figure(table), MAIN / "fig4_state_effects")
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
