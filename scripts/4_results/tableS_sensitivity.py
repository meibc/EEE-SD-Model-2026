#!/usr/bin/env python3
"""Results, section 4: sensitivity of the effect of halving anticipated healthcare stigma (AHS).

One row per analysis, pooled over the primary states: infections averted 2025-2036 and the change in 2036
incidence (median and 95% interval), with the assumption changed, why it was tested and how to read it.
Analyses: the reference (no PrEP -> AHS feedback) and the two other main feedback values (as in Table 1); the
supplement feedback values (+0.30, the upper end of the range consistent with the wording-adjusted trends, and
+0.45 with AHS not adjusted for the 2022 item change, fitted on the unadjusted inputs); and, each on the
reference, the prevalence ratio from all AMIS cycles, lower and higher persistence priors and no geographic
shrinkage (config/specifications.py).

Output: results/paper/supplement/tableS_sensitivity.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reporting import sources  # noqa: E402
from reporting.summaries import pooled_effect  # noqa: E402

SUPPLEMENT = sources.SUPPLEMENT
SENS = sources.SENSITIVITY / "scenarios"
ANALYSES = [
    ("Reference: no PrEP→AHS feedback", "Feedback b = 0", "Reference", sources.FEEDBACK["0"],
     "Reference estimate."),
    ("Feedback −0.075 (affirming care)", "PrEP contact lowers AHS (b = −0.075)",
     "Lower edge of the feedback values consistent with the wording-adjusted trends", sources.FEEDBACK["neg075"],
     "Effect if PrEP care is affirming; AHS falls without intervention, so halving removes fewer points."),
    ("Feedback +0.15 (best fit)", "PrEP use raises AHS (b = +0.15)",
     "Best fit to the wording-adjusted trends", sources.FEEDBACK["pos15"],
     "Effect if AHS keeps rising with PrEP use; halving removes more points."),
    ("Feedback +0.30", "b = +0.30", "Upper end of the feedback values consistent with the wording-adjusted trends",
     sources.FEEDBACK_POS30, "Strong feedback: AHS rises steeply without intervention."),
    ("AHS not adjusted for the 2022 item change", "Measured AHS used throughout, its best-fitting feedback "
     "(b = +0.45) and the prevalence ratio from all AMIS cycles",
     "Unchanged items suggest part of the 2022 rise was real", sources.UNADJUSTED_WORDING,
     "Effect if the 2022 rise in AHS is real rather than a wording artifact."),
    ("Prevalence ratio from all AMIS cycles", "AHS-PrEP prevalence ratio from 2017-2024 (both AHS wordings) "
     "instead of 2017-2021; no feedback", "The 2022 item change weakened the AHS-PrEP association",
     SENS / "evidence_all_years", "Effect if the association measured under both wordings applies."),
    ("Lower persistence", "Persistence priors 0.10 lower; no feedback",
     "Persistence of AHS, testing, risk and disclosure is set by its prior", SENS / "persistence_low",
     "Faster return toward each state's usual levels."),
    ("Higher persistence", "Persistence priors 0.10 higher; no feedback",
     "Persistence of AHS, testing, risk and disclosure is set by its prior", SENS / "persistence_high",
     "Slower return toward each state's usual levels."),
    ("No geographic shrinkage", "States fitted without shrinkage toward their division; no feedback",
     "Each state has three transitions; shrinkage borrows strength from its division", SENS / "no_hierarchy",
     "Effect from each state's own data alone."),
]


def main() -> None:
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, changed, why, path, reading in ANALYSES:
        e = pooled_effect(path, "halve_ahs")
        rows.append({"analysis": name, "assumption_changed": changed, "why_tested": why,
                     "averted": e["averted_median"], "averted_low": e["averted_low"], "averted_high": e["averted_high"],
                     "averted_pct": e["averted_pct_median"], "incidence_pct": e["incidence_pct_median"],
                     "incidence_pct_low": e["incidence_pct_low"], "incidence_pct_high": e["incidence_pct_high"],
                     "interpretation": reading})
    table = pd.DataFrame(rows)
    table.to_csv(SUPPLEMENT / "tableS_sensitivity.csv", index=False)
    print(table[["analysis", "averted", "averted_low", "averted_high", "averted_pct", "incidence_pct",
                 "incidence_pct_low", "incidence_pct_high"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
