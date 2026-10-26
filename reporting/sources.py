"""Locations of the model outputs read by the results scripts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SECTIONS = RESULTS / "sections"

INPUT_DRAWS = RESULTS / "02_inputs/state_input_draws.npz"
PRIMARY_STATES = RESULTS / "02_inputs/state_screen/primary_states.csv"
AHS_PREP_EVIDENCE = RESULTS / "02_inputs/ahs_prep_evidence.csv"
WORDING = RESULTS / "02_inputs/wording"
MSM_POPULATION = ROOT / "inputs/msm_state_population.csv"          # fixed state MSM population (scripts/00)

PRIMARY_FIT = RESULTS / "03_fits/primary/fits/primary.npz"
SPECIFICATION_FITS = RESULTS / "05_sensitivity/model_specification/fits"

# Scenario outputs (scripts/11, --constant-pr-prep, reported at every two-year step 2026-2036): the fitted
# PrEP -> AHS relationship continued in the projections (primary) or not continued
# (--zero-projection-edge prep_used:stigma_ahs).
CONTINUED = RESULTS / "04_scenarios/paper_static"
NOT_CONTINUED = RESULTS / "05_sensitivity/static_no_prep_ahs_feedback"
PLACEBO = RESULTS / "05_sensitivity/static_placebo_ahs"                     # continued / not continued under the AHS placebo
UNADJUSTED_WORDING = RESULTS / "05_sensitivity/ahs_wording_unadjusted/scenarios_static"
STATIC_SPECIFICATIONS = RESULTS / "05_sensitivity/static_specifications"   # <specification>/
# Contextual upper bound: climate prevalence ratio in place of the respondent-level one, <dir>/{continued,not_continued}.
CONTEXTUAL = {0.80: RESULTS / "05_sensitivity/contextual_static_pr080", 0.70: RESULTS / "05_sensitivity/contextual_static_pr070"}
SYSTEM_DECOMPOSITION = RESULTS / "07_structural/system_decomposition.csv"
