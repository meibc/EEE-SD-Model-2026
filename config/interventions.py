"""Scenario definitions for the paper.

Intervention codes say which stigma measure changes and how (models/scenarios/intervention.py); scenarios
combine them. Cuts are set relative to each draw's own 2024 (last observed) level and held there, never
above the baseline path, so their size does not depend on the projected baseline. The benchmark is the AHS
reference probability (5th percentile of observed state-period levels). Stigma-resilient services raise the
AHS-PrEP prevalence ratio r toward 1 (ratio_gap_closed, pipeline/projection.py). Changes ramp in over the
--intervention-duration-steps two-year steps.
"""


def _ahs_benchmark() -> float:
    from config.dynamic import DynamicConfig
    config = DynamicConfig()
    return float(config.reference_probs()[config.v_names.index("stigma_ahs")])


INTERVENTION_CODEBOOK = {
    **{f"halve_{short}_from_2024": {"var": var, "delta": -0.5, "effect": "probability_relative_to_last_observed",
                                    "label": f"Reduce {var} to half its 2024 level"}
       for short, var in (("ahs", "stigma_ahs"), ("gss", "stigma_gss"), ("family_stigma", "stigma_family"))},
    "eliminate_ahs_from_2024": {"var": "stigma_ahs", "delta": -1.0, "effect": "probability_relative_to_last_observed",
                                "label": "Eliminate anticipated healthcare stigma (counterfactual)"},
    "hold_ahs_at_2024": {"var": "stigma_ahs", "delta": 0.0, "effect": "probability_relative_to_last_observed",
                         "label": "Hold anticipated healthcare stigma at or below its 2024 level"},
    # Worsening climate: general social and family stigma raised relative to their baseline paths by about the
    # national 2021 -> 2022 rise (scripts/2_evidence/02: social 19% -> 32%, family 20% -> 26%), sustained
    "raise_gss_climate": {"var": "stigma_gss", "delta": 0.6, "effect": "probability_relative_to_baseline",
                          "label": "General social stigma 60% above its baseline path"},
    "raise_family_climate": {"var": "stigma_family", "delta": 0.25, "effect": "probability_relative_to_baseline",
                             "label": "Family stigma 25% above its baseline path"},
    # PrEP stall (mixture fits only, where the PrEP row is P0, PrEP use among men without AHS)
    "hold_prep_at_2024": {"var": "prep_used", "delta": 0.0, "effect": "probability_relative_to_last_observed",
                          "label": "PrEP use among men without AHS held at or below its 2024 level"},
    "ahs_to_benchmark": {"var": "stigma_ahs", "delta": -1.0, "effect": "probability_gap_to_benchmark",
                         "benchmark": _ahs_benchmark(),
                         "label": "Bring anticipated healthcare stigma to the benchmark (5th percentile)"},
}

SCENARIO_CODEBOOK = {
    "halve_ahs": {"intervention_codes": ["halve_ahs_from_2024"],
                  "label": "Halve anticipated healthcare stigma from its 2024 level"},
    "ahs_to_benchmark": {"intervention_codes": ["ahs_to_benchmark"],
                         "label": "Reduce anticipated healthcare stigma to the 5th percentile of observed levels"},
    "halve_upstream_stigma": {"intervention_codes": ["halve_gss_from_2024", "halve_family_stigma_from_2024"],
                              "label": "Halve general social and family stigma from their 2024 levels"},
    "ahs_no_worsening": {"intervention_codes": ["hold_ahs_at_2024"],
                         "label": "Anticipated healthcare stigma held at or below its 2024 level"},
    "eliminate_ahs": {"intervention_codes": ["eliminate_ahs_from_2024"],
                      "label": "Eliminate anticipated healthcare stigma (counterfactual: infections attributable to AHS)"},
    "resilient_services": {"intervention_codes": [], "ratio_gap_closed": 0.5,
                           "label": "Stigma-resilient services: half the PrEP gap for men with AHS closed (1 - r halved), AHS unchanged"},
    "worsening_climate": {"intervention_codes": ["raise_gss_climate", "raise_family_climate"],
                          "label": "Worsening climate: general social and family stigma raised by a sustained 2022-sized rise"},
    "prep_stall": {"intervention_codes": ["hold_prep_at_2024"],
                   "label": "PrEP stall: PrEP use among men without AHS held at its 2024 level"},
}
PAPER_SCENARIOS = ("halve_ahs", "resilient_services", "prep_stall", "ahs_no_worsening", "worsening_climate")
SUPPLEMENT_SCENARIOS = ()
