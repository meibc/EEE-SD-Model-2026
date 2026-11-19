#!/usr/bin/env python3
"""Results, section 2: projections over time (Fig. 2).

Fig. 2   (A) Anticipated healthcare stigma (AHS), the primary states combined (state values weighted by MSM
         population, the weights of the epidemic model): observed 2017-18 to 2023-24 (wording-adjusted, as in the
         model; median and 95% interval across respondent-bootstrap draws), then 2024-2036 without intervention
         for each PrEP -> AHS
         feedback value (-0.075 affirming care, 0 the reference, +0.15 the best fit to the wording-adjusted
         trends; median and 95% interval across draws), and AHS halved from its 2024 level (the same path for
         every feedback value: the intervention holds AHS on its target); arrows give the AHS points halving
         removes in 2036. Also observed AHS as measured (not adjusted for the 2022 item change; its projection is
         in the sensitivity analyses). (B) Change in
         annual HIV incidence, pooled over the primary states, relative to the paired baseline, for halving AHS,
         holding AHS at or below its 2024 level and stigma-resilient services (half the PrEP gap for men with
         AHS closed, AHS unchanged); each feedback value; median and 95% interval across draws.
         Projections start from observed 2023-24 values; interventions ramp in over 2026 and reach full strength
         in 2028. Values are reported at each two-year step of the model.

Outputs: results/paper/main/fig2_projections.{png,pdf}; in results/paper/source_data/, fig2_ahs_observed.csv,
fig2_ahs_trajectories.csv and fig2_incidence_trajectories.csv.
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
from data.state_inputs import StateInputs  # noqa: E402
from reporting.summaries import interval, primary_states  # noqa: E402

MAIN, DATA = sources.MAIN, sources.SOURCE_DATA
START_YEAR = 2024
FIRST_YEAR = 2018
PERIOD_LABELS = {2018: "2017–18", 2020: "2019–20", 2022: "2021–22", 2024: "2023–24"}
SCENARIOS = {"halve_ahs": "Halve AHS", "ahs_no_worsening": "Hold AHS at its 2024 level",
             "resilient_services": "Stigma-resilient services"}


def _summarize(values: pd.Series, keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame.from_dict({k: interval(g) for k, g in values.groupby(keys)}, orient="index").rename_axis(keys)


def msm_weights() -> pd.Series:
    """MSM population by primary state (inputs/msm_state_population.csv, median)."""
    return pd.read_csv(sources.MSM_POPULATION).set_index("state")["median"].reindex(primary_states())


def pooled(values: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """MSM-weighted mean of ``cols`` across states (unit_id) within ``keys``."""
    w = values["unit_id"].map(msm_weights())
    weighted = values[cols].mul(w, axis=0).assign(w=w, **{k: values[k] for k in keys}).groupby(keys).sum()
    return weighted[cols].div(weighted["w"], axis=0)


def observed_ahs(path: Path = sources.INPUT_DRAWS) -> pd.DataFrame:
    """Observed AHS (percent; wording-adjusted, or as measured from the unadjusted inputs) for the primary states
    combined (MSM-weighted) per respondent-bootstrap draw, summarized across draws, by period."""
    inputs = StateInputs.load(path)
    states = primary_states()
    cols = [inputs.unit_ids.index(s) for s in states]
    x = 100 * inputs.observed_draws[:, cols, inputs.v_names.index("stigma_ahs")]       # draw x state x period
    w = msm_weights().to_numpy()
    pooled_draws = np.einsum("dsp,s->dp", x, w / w.sum())
    return pd.DataFrame({int(y): interval(pooled_draws[:, i]) for i, y in enumerate(inputs.years)}).T.rename_axis("year")


def ahs_trajectories() -> pd.DataFrame:
    """AHS (percent) for the primary states combined (MSM-weighted) per draw, then summarized across draws,
    by feedback value, year and arm (no intervention, AHS halved)."""
    states = primary_states()
    out = {}
    for feedback, path in sources.FEEDBACK.items():
        s = pd.read_csv(path / "scenario_endpoint_dynamic_draws.csv.gz")
        s = s[s["unit_id"].isin(states) & s["variable"].eq("stigma_ahs") & s["scenario"].eq("halve_ahs")]
        per_draw = 100 * pooled(s, ["year", "draw"], ["baseline", "scenario_value"])
        out[feedback] = pd.concat({arm: _summarize(per_draw[col], ["year"]) for arm, col in
                                   {"no_intervention": "baseline", "ahs_halved": "scenario_value"}.items()}, names=["arm"])
    return pd.concat(out, names=["feedback"])


def incidence_trajectories() -> pd.DataFrame:
    """Change (percent) in annual incidence summed over the primary states, per draw relative to the paired
    baseline, by feedback value, scenario and year."""
    states = primary_states()
    out = {}
    for feedback, path in sources.FEEDBACK.items():
        e = pd.read_csv(path / "scenario_endpoint_epi_draws.csv.gz")
        e = e[e["unit_id"].isin(states) & e["variable"].eq("incidence") & e["scenario"].isin(SCENARIOS)]
        t = e.groupby(["scenario", "year", "draw"])[["baseline", "scenario_value"]].sum()
        out[feedback] = _summarize(100 * (t["scenario_value"] / t["baseline"] - 1), ["scenario", "year"])
    return pd.concat(out, names=["feedback"])


def figure(ahs: pd.DataFrame, incidence: pd.DataFrame, obs: pd.DataFrame, obs_unadjusted: pd.DataFrame) -> plt.Figure:
    style.apply()
    order = list(sources.FEEDBACK)
    observed = float(obs.loc[START_YEAR, "median"])
    c_un = style.UNADJUSTED
    fig = plt.figure(figsize=(7.2, 5.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.8, 1], left=0.09, right=0.98, top=0.93, bottom=0.17,
                            hspace=0.7, wspace=0.12)
    a = fig.add_subplot(grid[0, :])
    a.axvspan(FIRST_YEAR - 0.5, START_YEAR, color="#f4f4f4", zorder=0, lw=0)
    a.errorbar(obs.index, obs["median"], yerr=[obs["median"] - obs["low"], obs["high"] - obs["median"]], color=style.OBSERVED,
               marker="D", ms=3.5, lw=1.6, elinewidth=0.8, capsize=0, zorder=4)
    o = obs_unadjusted
    a.errorbar(o.index + 0.15, o["median"], yerr=[o["median"] - o["low"], o["high"] - o["median"]], color=c_un,
               marker="s", ms=3.2, mfc="white", lw=1.2, ls="--", elinewidth=0.7, capsize=0, zorder=4)
    a.text(START_YEAR - 0.2, o.loc[START_YEAR, "median"] + 0.6, "As measured", fontsize=6.5, color=c_un,
           ha="right", va="bottom")
    a.text(START_YEAR - 0.2, observed - 1.4, "Wording-adjusted\n(model input)", fontsize=6.5, color=style.OBSERVED,
           ha="right", va="top")
    a.text(2021, 0.6, "Observed", fontsize=7, color=style.MUTED, ha="center")
    a.text(2025.2, 0.6, "Projected", fontsize=7, color=style.MUTED, ha="left")
    halved = ahs.loc[("0", "ahs_halved")]
    x = [START_YEAR] + list(halved.index)
    a.plot(x, [observed] + list(halved["median"]), color=style.MUTED, lw=1.3, ls="--")
    a.text(2030, halved.loc[2030, "median"] - 1.2, "AHS halved (every feedback value)", fontsize=7, color=style.MUTED,
           ha="center", va="top")
    for k, feedback in enumerate(order):
        c, base = style.COLORS[feedback], ahs.loc[(feedback, "no_intervention")]
        yb = [observed] + list(base["median"])
        a.fill_between(x, [observed] + list(base["low"]), [observed] + list(base["high"]), color=c, alpha=0.12, lw=0)
        a.plot(x, yb, color=c, lw=2, marker=style.MARKERS[feedback], ms=3.5, label=style.LABELS[feedback])
        xb = 2036.5 + 0.7 * k                                   # AHS points removed in 2036
        a.annotate("", xy=(xb, halved["median"].iloc[-1]), xytext=(xb, yb[-1]),
                   arrowprops=dict(arrowstyle="->", color=c, lw=1.2, shrinkA=0, shrinkB=0))
        a.text(xb, yb[-1] + 0.4, f"−{yb[-1] - halved['median'].iloc[-1]:.0f}", fontsize=6.5, color=c, ha="center", va="bottom")
    a.text(2037.2, halved["median"].iloc[-1] - 0.6, "Points\nremoved", fontsize=6.5, color=style.MUTED, va="top", ha="center")
    a.set_xlim(FIRST_YEAR - 0.5, 2038.3)
    a.set_xticks(range(FIRST_YEAR, 2037, 2))
    a.set_xticklabels([PERIOD_LABELS.get(y, str(y)) for y in range(FIRST_YEAR, 2037, 2)], fontsize=7)
    a.set_ylim(0, None)
    a.set_ylabel("AHS (%)")
    a.set_title("A  Anticipated healthcare stigma (AHS), primary states combined", loc="left", fontweight="bold")
    axes = [fig.add_subplot(grid[1, i]) for i in range(len(SCENARIOS))]
    lo = min(incidence["low"].min(), 0)
    hi = max(incidence["high"].max(), 0)
    for ax, (code, label) in zip(axes, SCENARIOS.items()):
        ax.axhline(0, color=style.MUTED, lw=0.8)
        for feedback in order:
            d, c = incidence.loc[(feedback, code)], style.COLORS[feedback]
            x = [START_YEAR] + list(d.index)
            ax.fill_between(x, [0] + list(d["low"]), [0] + list(d["high"]), color=c, alpha=0.12, lw=0)
            ax.plot(x, [0] + list(d["median"]), color=c, lw=1.8, marker=style.MARKERS[feedback], ms=3.5)
        ax.set_title(label, loc="left", fontsize=8)
        ax.set_ylim(lo * 1.05, hi * 1.05 + 0.5)
        ax.set_xticks(range(START_YEAR, 2037, 4))
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("Change in annual\nHIV incidence (%)")
    fig.text(0.01, 0.56, "B  Effect on HIV incidence, primary states combined", fontweight="bold", fontsize=9)
    handles = [plt.Line2D([], [], color=style.COLORS[f], marker=style.MARKERS[f], lw=1.8, ms=4) for f in order]
    fig.legend(handles, [style.LABELS[f] for f in order], title="PrEP→AHS feedback (wording-adjusted AHS)",
               title_fontsize=7.5, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.0, "Medians, shaded 95% of draws; A weighted by state MSM population.",
             ha="center", fontsize=7, color=style.MUTED)
    return fig


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ahs, incidence = ahs_trajectories(), incidence_trajectories()
    obs, obs_unadjusted = observed_ahs(), observed_ahs(sources.INPUT_DRAWS_UNADJUSTED)
    pd.concat({"adjusted": obs, "unadjusted": obs_unadjusted}, names=["ahs"]).to_csv(DATA / "fig2_ahs_observed.csv")
    ahs.to_csv(DATA / "fig2_ahs_trajectories.csv")
    incidence.to_csv(DATA / "fig2_incidence_trajectories.csv")
    style.save(figure(ahs, incidence, obs, obs_unadjusted), MAIN / "fig2_projections")
    print(incidence["median"].unstack("year").round(1).to_string())


if __name__ == "__main__":
    main()
