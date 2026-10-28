"""Model specifications: the primary model and the variants used in the sensitivity analyses.

Each variant changes one component of the primary model; pipeline/dynamic_fit.py applies the change.
Change keys:
    wR                 covariance-fit weight (config/optimization.py; primary 0.01)
    shrinkage_off      geographic shrinkage weights set to 0 (config/shrinkage.py)
    drift_mean         drift prior means replaced, by measure
    persistence_shift  targeted persistence prior means shifted (PrEP, contact, GSS, family), capped at PERSISTENCE_CAP
    permuted           measures whose periods are permuted within each unit and draw (timing destroyed)
    evidence           "all_years": evidence target from all AMIS cycles (both AHS wordings)
"""

EVIDENCE_SEED = 20261023    # each draw's prevalence ratio, seeded by its input draw (paired across specifications)
PERSISTENCE_CAP = 0.98

SPECIFICATIONS = {
    # Reported in the sensitivity table
    "primary": {},
    "evidence_all_years": {"evidence": "all_years"},
    "persistence_low": {"persistence_shift": -0.10},
    "persistence_high": {"persistence_shift": 0.10},
    "placebo_ahs": {"permuted": ("stigma_ahs",)},
    # Supplement checks of the baseline projections
    "trajectory_only": {"wR": 0.0},
    "covariance_balanced": {"wR": 0.40},
    "no_hierarchy": {"shrinkage_off": True},
    "zero_prep_drift": {"drift_mean": {"prep_used": 0.0}},
}
