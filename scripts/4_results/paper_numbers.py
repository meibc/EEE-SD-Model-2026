#!/usr/bin/env python3
"""Every number quoted in the paper, in paper order, formatted as it is written in the text.

Reads the outputs of the other results scripts and the demographic stage, so run_pipeline.sh runs it last.
Sections: (1) observed trajectories by state and demographic group; (2) the AHS-PrEP association, the
fitted PrEP -> AHS relationship and projected AHS; (3) effects of reducing AHS, pooled, by state and in
the sensitivity analyses; then Methods counts and supplement diagnostics. Each line names its source file.
Intervals are 95% unless marked IQR.

Output: results/sections/paper_numbers.md
"""

from __future__ import annotations

from pathlib import Path
import sys

import arviz as az
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.fit import BOOTSTRAP_DRAWS, FIT_DRAWS  # noqa: E402
from config.projection import N_SAMPLES  # noqa: E402
from reporting import sources  # noqa: E402
from reporting.style import MEASURE_LABELS  # noqa: E402

S = sources.SECTIONS
IN = sources.RESULTS / "02_inputs"
GROUPS = {"black_nh": "Black, non-Hispanic", "hispanic_latino": "Hispanic/Latino", "white_nh": "White, non-Hispanic",
          "age_under_24": "age ≤24", "age_25_34": "age 25–34", "age_35_plus": "age ≥35"}
PROJECTIONS = {"continued": "relationship continued", "not_continued": "relationship not continued"}
lines: list[str] = []


def out(text: str = "") -> None:
    lines.append(text)


def pct(x: float, signed: bool = False) -> str:
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def pp(x: float) -> str:
    return f"{x:+.1f} points"


def ci(m: float, lo: float, hi: float, f=lambda v: f"{v:.1f}") -> str:
    return f"{f(m)} (95% {f(lo)} to {f(hi)})"


def count(x: float) -> str:
    return f"{x:,.0f}"


def trajectories() -> None:
    out("## 1. Observed trajectories")
    out("")
    sc = pd.read_csv(S / "trajectories/state_changes.csv")
    out(f"Primary states: {sc.state.nunique()}. Levels are medians across states; changes 2018 to 2024 in "
        "percentage points; 'clear' = 95% interval of the change excludes zero. AHS is wording-adjusted. "
        "[trajectories/state_changes.csv]")
    out("")
    for m, g in sc.groupby("measure", sort=False):
        low, high = g.loc[g.level_2024.idxmin()], g.loc[g.level_2024.idxmax()]
        down, up = g.loc[g.change.idxmin()], g.loc[g.change.idxmax()]
        clear = "; ".join(f"{d} in all {len(h)} states" if len(h) == len(g) else
                          f"{d} in {', '.join(f'{r.state} ({pp(r.change)})' for r in h.itertuples())}"
                          for d, h in g[g.direction != "no clear change"].groupby("direction"))
        out(f"- **{MEASURE_LABELS[m]}:** {pct(g.level_2018.median())} in 2018, {pct(g.level_2024.median())} in 2024; "
            f"2024 range {low.state} {pct(low.level_2024)} to {high.state} {pct(high.level_2024)}. "
            f"Median change {pp(g.change.median())} ({down.state} {pp(down.change)} to {up.state} {pp(up.change)}). "
            f"Clear change: {clear or 'none'}.")
    out("")
    gc = pd.read_csv(sources.RESULTS / "08_demographic/group_change_2017_2024.csv")
    out("Demographic groups, national weights, 2017 to 2024 (AHS as measured, not wording-adjusted). "
        "* = 95% interval excludes zero. [08_demographic/group_change_2017_2024.csv]")
    out("")
    for v in ("prep_used", "stigma_ahs", "stigma_gss", "stigma_family"):
        g = gc[gc.variable == v].set_index("population_id").loc[list(GROUPS)]
        items = ", ".join(f"{GROUPS[p]} {r.estimate_2017:.1f}% to {r.estimate_2024:.1f}% "
                          f"({pp(r.change_pp)}{'*' if r.interval_excludes_zero else ''})" for p, r in g.iterrows())
        out(f"- **{MEASURE_LABELS[v]}:** {items}.")
    out("")


def relationship() -> None:
    out("## 2. The AHS–PrEP association and the fitted relationship")
    out("")
    a = pd.read_csv(S / "ahs_reduction/association.csv").iloc[0]
    out(f"- Pooled prevalence ratio of PrEP use, AHS vs none (AMIS 2017–2021): {ci(a.pooled_apr, a.pooled_low, a.pooled_high, lambda v: f'{v:.2f}')}; "
        f"states after pooling {a.state_pooled_min:.2f} to {a.state_pooled_max:.2f} "
        f"(unpooled {a.state_unpooled_min:.2f} to {a.state_unpooled_max:.2f}). [ahs_reduction/association.csv]")
    cl = pd.read_csv(IN / "climate_associations.csv").query("term == 'climate_ahs'").set_index(["model", "contrast"])
    out("- AHS climate, prevalence ratio per 10 points: " + "; ".join(
        f"{c.replace('_', ' ')}{' with social and family climate' if m == 'all_climates' else ''} "
        f"{ci(r.pr, r.low, r.high, lambda v: f'{v:.2f}')}" for (m, c), r in cl.iterrows()) + ". [02_inputs/climate_associations.csv]")
    w = pd.read_csv(IN / "wording/shift_2022.csv").set_index("measure")
    out(f"- 2022 item change, AMIS 2022 vs 2021 (log-odds): AHS {w.loc['ahs', 'estimate']:.3f} (SE {w.loc['ahs', 'se']:.3f}); "
        f"unchanged items: experienced healthcare stigma {w.loc['ehs', 'estimate']:.2f}, general social {w.loc['gss', 'estimate']:.2f}, "
        f"family {w.loc['family', 'estimate']:.2f}. [02_inputs/wording/shift_2022.csv]")
    po = pd.read_csv(S / "coevolution/prep_to_ahs_overall.csv").set_index("fit")
    st = pd.read_csv(S / "coevolution/prep_to_ahs.csv").query("fit == 'primary'")
    strong = st[st.share_positive >= 0.9]
    for fit, label in (("primary", "primary fit"), ("placebo_ahs", "AHS temporal placebo")):
        r = po.loc[fit]
        out(f"- Fitted PrEP → AHS ({label}, centred-logit units per two-year step): median {r['median']:.2f} "
            f"(IQR {r.p25:.2f} to {r.p75:.2f}), {100 * r.share_positive:.0f}% of draws positive.")
    out(f"- States with ≥90% of draws positive: {', '.join(f'{r.state} ({100 * r.share_positive:.0f}%)' for r in strong.itertuples())}. "
        "[coevolution/prep_to_ahs*.csv]")
    t = pd.read_csv(S / "projections/ahs_trajectories.csv").set_index(["projection", "arm", "year"])
    for p, label in PROJECTIONS.items():
        b0, b1, h = t.loc[(p, "no_intervention", 2026)], t.loc[(p, "no_intervention", 2036)], t.loc[(p, "ahs_halved", 2036)]
        out(f"- Across-state median AHS, {label}: {pct(b0['median'])} in 2026 to {ci(b1['median'], b1.low, b1.high, pct)} in 2036 "
            f"without intervention; {pct(h['median'])} with AHS halved. [projections/ahs_trajectories.csv]")
    pa = pd.read_csv(S / "ahs_reduction/projected_ahs.csv")
    top = pa[pa.continued_share_above_30 >= 0.3].sort_values("continued_2036", ascending=False)
    out(f"- Projected 2036 AHS without intervention, median state: {pct(pa.continued_2036.median())} continued, "
        f"{pct(pa.not_continued_2036.median())} not continued (observed 2024: {pct(pa.observed_2024.median())}).")
    out("- States with ≥30% of draws above 30% AHS in 2036 (continued): " + ", ".join(
        f"{r.state} {pct(r.observed_2024)} → {pct(r.continued_2036)} ({100 * r.continued_share_above_30:.0f}% of draws; "
        f"not continued {pct(r.not_continued_2036)}, {100 * r.not_continued_share_above_30:.0f}%)" for r in top.itertuples())
        + ". [ahs_reduction/projected_ahs.csv]")
    out("")


def effects() -> None:
    out("## 3. Effects of reducing AHS")
    out("")
    t1 = pd.read_csv(S / "ahs_reduction/table1_pooled.csv")
    out(f"Pooled over the primary states; infections averted 2025–2036. Baseline infections 2025–2036: "
        f"{count(t1.baseline_cumulative_median.iloc[0])} (continued). [ahs_reduction/table1_pooled.csv]")
    out("")
    for r in t1.itertuples():
        out(f"- **{r.scenario}, {PROJECTIONS[r.projection]}:** {ci(r.averted_median, r.averted_low, r.averted_high, count)} "
            f"averted, {pct(r.averted_pct_median)} of baseline ({pct(r.averted_pct_low)} to {pct(r.averted_pct_high)}); "
            f"2036 incidence {ci(r.incidence_pct_median, r.incidence_pct_low, r.incidence_pct_high, lambda v: pct(v, True))}.")
    out("")
    ts = pd.read_csv(S / "ahs_reduction/tableS_state_effects.csv")
    out("By state, halving AHS (medians; 2036 incidence change, infections averted, per 100,000 MSM per year). "
        "[ahs_reduction/tableS_state_effects.csv]")
    out("")
    for p, label in PROJECTIONS.items():
        g = ts[ts.projection == p]
        inc = g.sort_values("incidence_change_2036_pct_median")
        def top(col, fmt):
            return ", ".join(f"{r.state} {fmt(r[col])}" for _, r in g.nlargest(3, col).iterrows())
        out(f"- **{label.capitalize()}:** median state {pct(inc.incidence_change_2036_pct_median.median(), True)}; range "
            f"{inc.state.iloc[0]} {pct(inc.incidence_change_2036_pct_median.iloc[0], True)} to "
            f"{inc.state.iloc[-1]} {pct(inc.incidence_change_2036_pct_median.iloc[-1], True)}. "
            f"Most averted: {top('infections_averted_median', count)}. "
            f"Per 100,000 MSM per year: {top('averted_per_100k_msm_year_median', lambda x: f'{x:.1f}')}.")
    v = pd.read_csv(S / "ahs_reduction/breakdown_variance_shares.csv").set_index("factor")
    names = {"burden": "baseline infection rate", "gap": "AHS points removed", "prep_per_point": "PrEP gain per AHS point"}
    for p, label in PROJECTIONS.items():
        out(f"- Share of between-state variation in infections averted per MSM ({label}): "
            + ", ".join(f"{names[f]} {100 * v.loc[f, p]:.0f}%" for f in names) + ". [ahs_reduction/breakdown_variance_shares.csv]")
    out("")
    out("Sensitivity, halving AHS, pooled. [sensitivity/table_sensitivity.csv]")
    out("")
    for r in pd.read_csv(S / "sensitivity/table_sensitivity.csv").itertuples():
        out(f"- **{r.analysis}:** {ci(r.averted, r.averted_low, r.averted_high, count)} averted ({pct(r.averted_pct)}); "
            f"2036 incidence {ci(r.incidence_pct, r.incidence_pct_low, r.incidence_pct_high, lambda x: pct(x, True))}.")
    out("")


def methods_and_supplement() -> None:
    out("## Methods counts")
    out("")
    screen = pd.read_csv(IN / "state_screen/state_effective_n_screen.csv")
    spec = pd.read_csv(S / "supplement/tableS_model_specification.csv")
    idata = az.from_netcdf(IN / "epi_fit/posterior.nc")
    post = idata.posterior.sizes
    out(f"- States passing the primary screen: {int(screen.primary_precision_screen.sum())}; stricter screen: "
        f"{int(screen.strict_precision_screen.sum())}. [02_inputs/state_screen/state_effective_n_screen.csv]")
    out(f"- Respondent-bootstrap replicates: {BOOTSTRAP_DRAWS}; dynamic-model fits per specification: {FIT_DRAWS}; "
        f"paired simulations per scenario run: {N_SAMPLES}. [config/fit.py, config/projection.py]")
    out(f"- Permitted off-diagonal pathways: {spec.component.str.startswith('Pathway prior').sum()}. "
        "[supplement/tableS_model_specification.csv]")
    out(f"- Epidemic-model posterior: {post['chain']} chains × {post['draw']} draws = {post['chain'] * post['draw']:,}.")
    out("")
    out("## Supplement diagnostics")
    out("")
    diag = az.summary(idata, kind="diagnostics")
    out(f"- Epidemic fit: {int(idata.sample_stats.diverging.sum())} divergences, maximum R-hat {diag.r_hat.max():.2f}, "
        f"minimum bulk ESS {diag.ess_bulk.min():.0f}, minimum tail ESS {diag.ess_tail.min():.0f}. [02_inputs/epi_fit/posterior.nc]")
    cal = pd.read_csv(S / "supplement/epi_calibration_summary.csv")
    out("- Epidemic calibration, observed in the 90% interval (median absolute error): " + "; ".join(
        f"{r.outcome} {r.period} {100 * r.coverage_90:.0f}% ({r.median_abs_error_pct:.1f}%)" for r in cal.itertuples())
        + ". [supplement/epi_calibration_summary.csv]")
    d = pd.read_csv(S / "supplement/dynamic_fit.csv")
    d["error"] = 100 * (d.fitted_median - d.observed).abs()
    by = d.groupby("measure").agg(mae=("error", "mean"), covered=("covered", "mean")).loc[list(MEASURE_LABELS)]
    out("- Dynamic-model fit, mean absolute error in points (share of observed values consistent with the fit): "
        + "; ".join(f"{MEASURE_LABELS[m]} {r.mae:.1f} ({100 * r.covered:.0f}%)" for m, r in by.iterrows())
        + f"; overall {100 * d.covered.mean():.0f}%. [supplement/dynamic_fit.csv]")


def main() -> None:
    out("# Paper numbers")
    out("")
    out("Generated by scripts/4_results/paper_numbers.py; sources are relative to results/sections/ unless they start "
        "with 02_inputs/ or 08_demographic/ (relative to results/).")
    out("")
    trajectories()
    relationship()
    effects()
    methods_and_supplement()
    (S / "paper_numbers.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {S / 'paper_numbers.md'}")


if __name__ == "__main__":
    main()
