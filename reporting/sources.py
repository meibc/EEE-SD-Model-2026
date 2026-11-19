"""Locations of the model outputs read by the results scripts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# Paper outputs: main-text and supplement figures and tables, and the CSVs behind them.
PAPER = RESULTS / "paper"
MAIN = PAPER / "main"
SUPPLEMENT = PAPER / "supplement"
SOURCE_DATA = PAPER / "source_data"

INPUT_DRAWS = RESULTS / "02_inputs/state_inputs.npz"
INPUT_DRAWS_UNADJUSTED = RESULTS / "02_inputs/state_inputs_ahs_unadjusted.npz"   # AHS not adjusted for the 2022 item change
PRIMARY_STATES = RESULTS / "02_inputs/state_screen/primary_states.csv"
CELL_PRECISION = RESULTS / "02_inputs/state_bootstrap/cell_precision.csv"   # Kish effective n by state, period and measure (scripts/1_data/04)
AHS_PREP_EVIDENCE = RESULTS / "02_inputs/ahs_prep_evidence.csv"
WORDING = RESULTS / "02_inputs/wording"
MSM_POPULATION = ROOT / "inputs/msm_state_population.csv"          # fixed state MSM population (read by scripts/1_data/02 and 03)

PRIMARY_FITS = RESULTS / "03_fits/primary"                                    # fits/, bundles/ <primary_fb_*>
PRIMARY_SCENARIOS = RESULTS / "04_scenarios/primary"                          # <primary_fb_*>/
SENSITIVITY = RESULTS / "05_sensitivity/specifications"                       # fits/, bundles/, scenarios/<specification>/
FEEDBACK_PROFILE = RESULTS / "05_sensitivity/feedback_profile"                # scripts/3_model/06

# Scenario runs (scripts/3_model/05, every two-year step 2026-2036) by PrEP -> AHS feedback value b. Main set in
# the paper's order: affirming care, no feedback (the reference), best fit to the wording-adjusted trends.
FEEDBACK = {"neg075": PRIMARY_SCENARIOS / "primary_fb_neg075", "0": PRIMARY_SCENARIOS / "primary_fb_0",
            "pos15": PRIMARY_SCENARIOS / "primary_fb_pos15"}
FEEDBACK_VALUES = {"neg075": -0.075, "0": 0.0, "pos15": 0.15}
REFERENCE = FEEDBACK["0"]
REFERENCE_BUNDLE = PRIMARY_FITS / "bundles/primary_fb_0.npz"
# Supplement feedback values: upper end of the adjusted range, and no AHS wording adjustment (unadjusted inputs)
FEEDBACK_POS30 = PRIMARY_SCENARIOS / "primary_fb_pos30"
UNADJUSTED_WORDING = SENSITIVITY / "scenarios/unadjusted_fb_pos45"
