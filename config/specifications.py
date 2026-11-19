"""Model specifications: the primary model at each PrEP -> AHS feedback value, and the sensitivity analyses.

The model itself (PrEP mixture, logistic diffusion toward each state's ceiling, feedback outside J) is described
in pipeline/dynamic_fit.py. Naming: "primary" is always the current model; a replaced model takes the next
version number (v2 was the previous one; its results are archived in results/archive/v2).
Keys:
    feedback           PrEP -> AHS feedback b (every specification): each year AHS (centred logit) moves by b x
                       the centred logit of population PrEP
    shrinkage_off      geographic shrinkage weights set to 0 (config/shrinkage.py)
    persistence_shift  persistence prior means shifted (every measure with a prior), capped at PERSISTENCE_CAP
    evidence           "all_years": AHS-PrEP prevalence ratio from all AMIS cycles (both AHS wordings)
"""

PERSISTENCE_CAP = 0.98

SPECIFICATIONS = {
    # Primary model, one fit per feedback value (results/03_fits/primary). Profiling b on the observed trends
    # (60 draws): wording-adjusted AHS, best b +0.15 (95% range -0.07 to +0.29; 17% of draws b <= 0); unadjusted
    # AHS (the 2022 wording change read as real), +0.44 (+0.24 to +0.58); both barely change with a more
    # persistent AHS. Main set: -0.075 (affirming care: PrEP contact lowers AHS; the lower edge of the adjusted
    # range), 0 (no feedback, the reference) and +0.15 (best fit to the adjusted trends). Supplement: +0.30 (upper
    # end of the adjusted range) and unadjusted_fb_pos45 below.
    "primary_fb_neg075": {"feedback": -0.075},
    "primary_fb_0": {"feedback": 0.0},
    "primary_fb_pos15": {"feedback": 0.15},
    "primary_fb_pos30": {"feedback": 0.30},
    # No AHS wording adjustment: fitted on the unadjusted inputs (run_pipeline.sh, --input-draws) at their
    # best-fitting feedback, with the prevalence ratio from all cycles (both wordings)
    "unadjusted_fb_pos45": {"feedback": 0.45, "evidence": "all_years"},
    # Sensitivity analyses: primary_fb_0 with one change (results/05_sensitivity/specifications)
    "evidence_all_years": {"feedback": 0.0, "evidence": "all_years"},
    "persistence_low": {"feedback": 0.0, "persistence_shift": -0.10},
    "persistence_high": {"feedback": 0.0, "persistence_shift": 0.10},
    "no_hierarchy": {"feedback": 0.0, "shrinkage_off": True},
}
