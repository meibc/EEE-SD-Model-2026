#!/usr/bin/env python3
"""Every number quoted in the paper, in paper order, formatted as it is written in the text.

Reads the outputs of the other results scripts and the demographic stage, so run_pipeline.sh runs it last.
Sections: (1) observed trajectories by state and demographic group; (2) the AHS-PrEP association, the
PrEP -> AHS feedback profile and projected AHS; (3) effects of reducing AHS, pooled, by state and in
the sensitivity analyses; then Methods counts and supplement diagnostics. Results are given for each main
feedback value (-0.075, 0 the reference, +0.15). Each line names its source file.
Intervals are 95% unless marked IQR.

Output: results/paper/paper_numbers.md
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

PAPER, DATA = sources.PAPER, sources.SOURCE_DATA
IN = sources.RESULTS / "02_inputs"
GROUPS = {"black_nh": "Black, non-Hispanic", "hispanic_latino": "Hispanic/Latino", "white_nh": "White, non-Hispanic",
          "age_under_24": "age ≤24", "age_25_34": "age 25–34", "age_35_plus": "age ≥35"}
FEEDBACK = {"neg075": "feedback −0.075 (affirming care)", "0": "no feedback (reference)", "pos15": "feedback +0.15 (best fit)"}
VALUES = {"neg075": -0.075, "0": 0.0, "pos15": 0.15}
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
    nat = pd.read_csv(DATA / "fig1_national.csv").set_index("survey_year")
    out("National (Fig. 1), AMIS 2017 to 2024, national weights, 95% intervals. [source_data/fig1_national.csv]")
    out("")
    series = {"stigma_ahs": "Anticipated healthcare stigma, as measured",
              "stigma_ahs_adjusted": "Anticipated healthcare stigma, wording-adjusted",
              "ehs": "Experienced healthcare stigma"} | {m: MEASURE_LABELS[m] for m in MEASURE_LABELS if m != "stigma_ahs"}
    for col, label in series.items():
        a, b = nat.loc[2017], nat.loc[2024]
        out(f"- **{label}:** {ci(a[col], a[f'{col}_low'], a[f'{col}_high'], pct)} in 2017, "
            f"{ci(b[col], b[f'{col}_low'], b[f'{col}_high'], pct)} in 2024 ({pp(b[col] - a[col])}); "
            f"2022 {pct(nat.loc[2022, col])} (2021 {pct(nat.loc[2021, col])}).")
    out("")
    sc = pd.read_csv(DATA / "fig1_state_changes.csv")
    out(f"By state (supplement heatmap), primary states: {sc.state.nunique()}. Levels are medians across states; changes "
        "2018 to 2024 in percentage points; 'clear' = 95% interval of the change excludes zero. AHS is wording-adjusted. "
        "[source_data/fig1_state_changes.csv]")
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
        "* = 95% interval excludes zero. [results/08_demographic/group_change_2017_2024.csv]")
    out("")
    for v in ("prep_used", "stigma_ahs", "stigma_gss", "stigma_family"):
        g = gc[gc.variable == v].set_index("population_id").loc[list(GROUPS)]
        items = ", ".join(f"{GROUPS[p]} {r.estimate_2017:.1f}% to {r.estimate_2024:.1f}% "
                          f"({pp(r.change_pp)}{'*' if r.interval_excludes_zero else ''})" for p, r in g.iterrows())
        out(f"- **{MEASURE_LABELS[v]}:** {items}.")
    out("")


def relationship() -> None:
    out("## 2. The AHS–PrEP association, the PrEP → AHS feedback profile and projected AHS")
    out("")
    a = pd.read_csv(DATA / "ahs_prep_association.csv").iloc[0]
    out(f"- Pooled prevalence ratio of PrEP use, AHS vs none (AMIS 2017–2021): {ci(a.pooled_apr, a.pooled_low, a.pooled_high, lambda v: f'{v:.2f}')}; "
        f"states after pooling {a.state_pooled_min:.2f} to {a.state_pooled_max:.2f} "
        f"(unpooled {a.state_unpooled_min:.2f} to {a.state_unpooled_max:.2f}). [source_data/ahs_prep_association.csv]")
    cl = pd.read_csv(IN / "climate_associations.csv").query("term == 'climate_ahs'").set_index(["model", "contrast"])
    out("- AHS climate, prevalence ratio per 10 points: " + "; ".join(
        f"{c.replace('_', ' ')}{' with social and family climate' if m == 'all_climates' else ''} "
        f"{ci(r.pr, r.low, r.high, lambda v: f'{v:.2f}')}" for (m, c), r in cl.iterrows()) + ". [results/02_inputs/climate_associations.csv]")
    w = pd.read_csv(IN / "wording/shift_2022.csv").set_index("measure")
    out(f"- 2022 item change, AMIS 2022 vs 2021 (log-odds): AHS {w.loc['ahs', 'estimate']:.3f} (SE {w.loc['ahs', 'se']:.3f}); "
        f"unchanged items: experienced healthcare stigma {w.loc['ehs', 'estimate']:.2f}, general social {w.loc['gss', 'estimate']:.2f}, "
        f"family {w.loc['family', 'estimate']:.2f}. [results/02_inputs/wording/shift_2022.csv]")
    bb = pd.read_csv(sources.FEEDBACK_PROFILE / "best_b.csv").set_index("inputs")
    for inputs, label in (("adjusted", "wording-adjusted AHS"), ("unadjusted", "AHS not adjusted for the 2022 item change")):
        r = bb.loc[inputs]
        out(f"- PrEP → AHS feedback best fitting the observed trends ({label}; {r.draws:.0f} draws): "
            f"{ci(r.best_b_median, r.best_b_low, r.best_b_high, lambda v: f'{v:+.2f}')}; "
            f"{100 * r.share_b_le_0:.0f}% of draws at or below 0. [results/05_sensitivity/feedback_profile/best_b.csv]")
    t = pd.read_csv(DATA / "fig2_ahs_trajectories.csv", dtype={"feedback": str}).set_index(["feedback", "arm", "year"])
    for f, label in FEEDBACK.items():
        b0, b1, h = t.loc[(f, "no_intervention", 2026)], t.loc[(f, "no_intervention", 2036)], t.loc[(f, "ahs_halved", 2036)]
        out(f"- AHS, primary states combined (MSM-weighted; Fig. 2A), {label}: {pct(b0['median'])} in 2026 to "
            f"{ci(b1['median'], b1.low, b1.high, pct)} in 2036 without intervention; {pct(h['median'])} with AHS halved "
            f"({b1['median'] - h['median']:.1f} points removed). [source_data/fig2_ahs_trajectories.csv]")
    pa = pd.read_csv(DATA / "projected_ahs_by_state.csv")
    out(f"- Projected 2036 AHS without intervention, median state (observed 2024: {pct(pa.observed_2024.median())}): "
        + "; ".join(f"{label} {pct(pa[f'{f}_2036'].median())}" for f, label in FEEDBACK.items()) + ".")
    top = pa[pa.pos15_share_above_30 >= 0.3].sort_values("pos15_2036", ascending=False)
    out("- States with ≥30% of draws above 30% AHS in 2036 (feedback +0.15): " + (", ".join(
        f"{r['state']} {pct(r['observed_2024'])} → {pct(r['pos15_2036'])} ({100 * r['pos15_share_above_30']:.0f}% of draws; "
        f"no feedback {pct(r['0_2036'])})" for _, r in top.iterrows()) or "none") + ". [source_data/projected_ahs_by_state.csv]")
    out("")


def effects() -> None:
    out("## 3. Effects of reducing AHS")
    out("")
    t1 = pd.read_csv(sources.MAIN / "table1_effects.csv")
    ref = t1[t1.feedback.eq(0.0)]
    out(f"Pooled over the primary states; infections averted 2025–2036. Baseline infections 2025–2036: "
        f"{count(ref.baseline_cumulative_median.iloc[0])} (no feedback). [main/table1_effects.csv]")
    out("")
    labels = {VALUES[f]: label for f, label in FEEDBACK.items()}
    for r in t1.itertuples():
        out(f"- **{r.scenario}, {labels[r.feedback]}:** {ci(r.averted_median, r.averted_low, r.averted_high, count)} "
            f"averted, {pct(r.averted_pct_median)} of baseline ({pct(r.averted_pct_low)} to {pct(r.averted_pct_high)}); "
            f"2036 incidence {ci(r.incidence_pct_median, r.incidence_pct_low, r.incidence_pct_high, lambda v: pct(v, True))}.")
    out("")
    ts = pd.read_csv(sources.SUPPLEMENT / "tableS_state_effects.csv")
    out("By state, halving AHS (medians; 2036 incidence change, infections averted, per 100,000 MSM per year). "
        "[supplement/tableS_state_effects.csv]")
    out("")
    for p, label in FEEDBACK.items():
        g = ts[ts.feedback == VALUES[p]]
        inc = g.sort_values("incidence_change_2036_pct_median")
        def top(col, fmt):
            return ", ".join(f"{r.state} {fmt(r[col])}" for _, r in g.nlargest(3, col).iterrows())
        out(f"- **{label[0].upper() + label[1:]}:** median state {pct(inc.incidence_change_2036_pct_median.median(), True)}; range "
            f"{inc.state.iloc[0]} {pct(inc.incidence_change_2036_pct_median.iloc[0], True)} to "
            f"{inc.state.iloc[-1]} {pct(inc.incidence_change_2036_pct_median.iloc[-1], True)}. "
            f"Most averted: {top('infections_averted_median', count)}. "
            f"Per 100,000 MSM per year: {top('averted_per_100k_msm_year_median', lambda x: f'{x:.1f}')}.")
    v = pd.read_csv(DATA / "fig3_variance_shares.csv").set_index("factor")
    v.columns = v.columns.astype(str)
    names = {"burden": "baseline infection rate", "gap": "AHS points removed", "prep_per_point": "PrEP gain per AHS point"}
    for p, label in FEEDBACK.items():
        out(f"- Share of between-state variation in infections averted per MSM ({label}): "
            + ", ".join(f"{names[f]} {100 * v.loc[f, p]:.0f}%" for f in names) + ". [source_data/fig3_variance_shares.csv]")
    out("")
    out("Sensitivity, halving AHS, pooled. [supplement/tableS_sensitivity.csv]")
    out("")
    for r in pd.read_csv(sources.SUPPLEMENT / "tableS_sensitivity.csv").itertuples():
        out(f"- **{r.analysis}:** {ci(r.averted, r.averted_low, r.averted_high, count)} averted ({pct(r.averted_pct)}); "
            f"2036 incidence {ci(r.incidence_pct, r.incidence_pct_low, r.incidence_pct_high, lambda x: pct(x, True))}.")
    out("")


def methods_and_supplement() -> None:
    out("## Methods counts")
    out("")
    screen = pd.read_csv(IN / "state_screen/state_effective_n_screen.csv")
    spec = pd.read_csv(sources.SUPPLEMENT / "tableS_model_specification.csv")
    idata = az.from_netcdf(IN / "epi_fit/posterior.nc")
    post = idata.posterior.sizes
    out(f"- States passing the primary screen: {int(screen.primary_precision_screen.sum())}; stricter screen: "
        f"{int(screen.strict_precision_screen.sum())}. [results/02_inputs/state_screen/state_effective_n_screen.csv]")
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
        f"minimum bulk ESS {diag.ess_bulk.min():.0f}, minimum tail ESS {diag.ess_tail.min():.0f}. [results/02_inputs/epi_fit/posterior.nc]")
    cal = pd.read_csv(DATA / "figS_epi_calibration.csv")
    out("- Epidemic calibration, observed in the 90% interval (median absolute error): " + "; ".join(
        f"{r.outcome} {r.period} {100 * r.coverage_90:.0f}% ({r.median_abs_error_pct:.1f}%)" for r in cal.itertuples())
        + ". [source_data/figS_epi_calibration.csv]")
    d = pd.read_csv(DATA / "figS_dynamic_fit.csv")
    d["error"] = 100 * (d.fitted_median - d.observed).abs()
    by = d.groupby("measure").agg(mae=("error", "mean"), covered=("covered", "mean")).loc[list(MEASURE_LABELS)]
    out("- Dynamic-model fit, mean absolute error in points (share of observed values consistent with the fit): "
        + "; ".join(f"{MEASURE_LABELS[m]} {r.mae:.1f} ({100 * r.covered:.0f}%)" for m, r in by.iterrows())
        + f"; overall {100 * d.covered.mean():.0f}%. [source_data/figS_dynamic_fit.csv]")


def main() -> None:
    out("# Paper numbers")
    out("")
    out("Generated by scripts/4_results/paper_numbers.py; sources are relative to results/paper/ unless they start "
        "with results/.")
    out("")
    trajectories()
    relationship()
    effects()
    methods_and_supplement()
    (PAPER / "paper_numbers.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {PAPER / 'paper_numbers.md'}")


if __name__ == "__main__":
    main()
