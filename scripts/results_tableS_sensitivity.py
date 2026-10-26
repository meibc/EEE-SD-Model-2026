#!/usr/bin/env python3
from __future__ import annotations

"""Results, section 4: sensitivity of the effect of halving anticipated healthcare stigma (AHS).

One row per analysis, pooled over the primary states: infections averted 2025-2036 and the change in
2036 incidence (median and 95% interval), with the assumption changed, why it was tested and how to
read it. Analyses: the primary projection; AHS reduced to the 5th-percentile benchmark instead of halved; the fitted PrEP -> AHS relationship not continued; the AHS
temporal placebo (relationship continued and not continued); the respondent-level association from all
AMIS cycles (both AHS wordings); AHS not adjusted for the 2022 item change; lower and higher persistence
priors; and the contextual upper bound (a between-state climate prevalence ratio of 0.80 or 0.70 in place
of the respondent-level one, relationship continued and not continued).

Output (results/sections/sensitivity/): table_sensitivity.csv
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reporting import sources  # noqa: E402
from reporting.summaries import pooled_effect  # noqa: E402

OUT = sources.SECTIONS / "sensitivity"
SPEC = sources.STATIC_SPECIFICATIONS
ANALYSES = [
    ("Primary", "Fitted PrEP-AHS relationship continued", "Reference", sources.CONTINUED,
     "Primary estimate."),
    ("Relationship not continued", "Fitted PrEP-AHS relationship removed from the projections",
     "Projected AHS growth rests on extrapolating the fitted relationship", sources.NOT_CONTINUED,
     "Effect without projected AHS increases; the difference from the primary is the part due to them."),
    ("AHS reduced to the benchmark", "AHS reduced to the 5th-percentile state level instead of halved",
     "Whether halving is a realistic size: the benchmark is a level some states already reach", sources.CONTINUED,
     "Nearly identical to halving: for the typical state, halving reaches about the benchmark.", "ahs_to_benchmark"),
    ("AHS temporal placebo, continued", "AHS periods permuted within each state before fitting",
     "Whether the fitted relationship depends on the timing of AHS", sources.PLACEBO / "feedback_on",
     "Effect when the time order of AHS carries no information."),
    ("AHS temporal placebo, not continued", "As above, relationship removed from the projections",
     "Separates the placebo's effect on the relationship from its effect on baselines", sources.PLACEBO / "feedback_off",
     "Placebo baseline without the fitted relationship."),
    ("Association from all AMIS cycles", "Prevalence ratio from 2017-2024 (both AHS wordings) instead of 2017-2021",
     "The 2022 item change weakened the AHS-PrEP association", SPEC / "evidence_all_years",
     "Effect if the association measured under both wordings applies."),
    ("AHS not adjusted for the 2022 item change", "Measured AHS used throughout; all-cycles association",
     "Unchanged items suggest the 2022 rise was largely real", sources.UNADJUSTED_WORDING,
     "Effect if the 2022 rise in AHS is real rather than a wording artifact."),
    ("Lower persistence", "Targeted persistence priors 0.10 lower", "Persistence shapes the projected baselines",
     SPEC / "persistence_low", "Faster return toward each state's usual levels."),
    ("Higher persistence", "Targeted persistence priors 0.10 higher", "Persistence shapes the projected baselines",
     SPEC / "persistence_high", "Slower return toward each state's usual levels."),
]
for pr in (0.80, 0.70):
    for projection, label in (("continued", "continued"), ("not_continued", "not continued")):
        ANALYSES.append((f"Contextual upper bound, ratio {pr:.2f}, {label}",
                         f"Climate prevalence ratio {pr:.2f} per 10 points of AHS applied to everyone's PrEP use",
                         "Whether AHS acts through the broader stigma climate, beyond the men who report it",
                         sources.CONTEXTUAL[pr] / projection,
                         "Hypothetical upper bound: treats a between-state association as causal."))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, changed, why, path, reading, *scenario in ANALYSES:
        e = pooled_effect(path, scenario[0] if scenario else "halve_ahs")
        rows.append({"analysis": name, "assumption_changed": changed, "why_tested": why,
                     "averted": e["averted_median"], "averted_low": e["averted_low"], "averted_high": e["averted_high"],
                     "averted_pct": e["averted_pct_median"], "incidence_pct": e["incidence_pct_median"],
                     "incidence_pct_low": e["incidence_pct_low"], "incidence_pct_high": e["incidence_pct_high"],
                     "interpretation": reading})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table_sensitivity.csv", index=False)
    print(table[["analysis", "averted", "averted_low", "averted_high", "averted_pct", "incidence_pct",
                 "incidence_pct_low", "incidence_pct_high"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
