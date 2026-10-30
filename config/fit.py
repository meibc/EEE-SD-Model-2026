"""Draws and seeds for fitting the dynamic model (scripts/3_model/01 and 03, pipeline/dynamic_fit.py)."""

BOOTSTRAP_DRAWS = 500       # respondent-bootstrap replicates expected from scripts/1_data/04 (its n_boot)
FIT_DRAWS = 500             # input draws fitted per specification
DRAW_SEED = 20260921        # chooses the input draws when FIT_DRAWS < BOOTSTRAP_DRAWS (the same draws in every specification)
EVIDENCE_SEED = 20261023    # each draw's prevalence ratio, seeded by its input draw (paired across specifications)
