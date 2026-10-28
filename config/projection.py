"""Projection and scenario-run settings (pipeline/scenarios.py, scripts/3_model/05, the results scripts)."""

N_SAMPLES = 500                     # paired dynamic-model / epidemic-model draws per run
SEED = 123                          # the same seed for the baseline and every scenario, so contrasts are paired
TARGET_END_YEAR = 2036
INTERVENTION_DURATION_STEPS = 3     # two-year steps over which interventions ramp to full strength (6 years)
CUMULATIVE_START_YEAR = 2025        # cumulative infections are counted from this year
INTERIM_REPORT_YEARS = (2030, 2032)  # reported with the end year unless --report-year is given
CONTEXTUAL_PREP_PRS = (0.80, 0.70)  # contextual upper bound: PrEP prevalence ratio per 10 points of AHS (scripts/2_evidence/03)
