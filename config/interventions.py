"""Scenario definitions for the paper.

Intervention codes say which stigma measure changes and how (models/shared/intervention.py); scenarios
combine them. Cuts are set relative to each draw's own 2024 (last observed) level and held there, never
above the baseline path, so their size does not depend on the projected baseline. The benchmark is the AHS
reference probability (5th percentile of observed state-period levels). Resilient PrEP delivery is a derived
policy comparison: half of the benchmark scenario's PrEP change, with AHS and every other measure at baseline
(pipeline/projection.py, prep_only_fraction). All changes ramp in over three two-year steps.
"""


def _ahs_benchmark() -> float:
    from config.dynamic import DynamicConfig
    config = DynamicConfig()
    return float(config.reference_probs()[config.v_names.index("stigma_ahs")])


INTERVENTION_CODEBOOK = {
    **{f"halve_{short}_from_2024": {"var": var, "delta": -0.5, "effect": "probability_relative_to_last_observed",
                                    "label": f"Reduce {var} to half its 2024 level"}
       for short, var in (("ahs", "stigma_ahs"), ("gss", "stigma_gss"), ("family_stigma", "stigma_family"))},
    "hold_ahs_at_2024": {"var": "stigma_ahs", "delta": 0.0, "effect": "probability_relative_to_last_observed",
                         "label": "Hold anticipated healthcare stigma at or below its 2024 level"},
    "ahs_to_benchmark": {"var": "stigma_ahs", "delta": -1.0, "effect": "probability_gap_to_benchmark",
                         "benchmark": _ahs_benchmark(),
                         "label": "Bring anticipated healthcare stigma to the benchmark (5th percentile)"},
}

SCENARIO_CODEBOOK = {
    "halve_ahs": {"intervention_codes": ["halve_ahs_from_2024"],
                  "label": "Halve anticipated healthcare stigma from its 2024 level"},
    "ahs_to_benchmark": {"intervention_codes": ["ahs_to_benchmark"],
                         "label": "Reduce anticipated healthcare stigma to the 5th percentile of observed levels"},
    "resilient_prep_delivery": {"intervention_codes": ["ahs_to_benchmark"], "prep_only_fraction": 0.5,
                                "label": "Stigma-resilient PrEP delivery: half the AHS-associated PrEP gap closed, AHS unchanged"},
    "halve_upstream_stigma": {"intervention_codes": ["halve_gss_from_2024", "halve_family_stigma_from_2024"],
                              "label": "Halve general social and family stigma from their 2024 levels"},
    "ahs_no_worsening": {"intervention_codes": ["hold_ahs_at_2024"],
                         "label": "Anticipated healthcare stigma held at or below its 2024 level"},
}
PAPER_SCENARIOS = ("halve_ahs", "ahs_to_benchmark", "resilient_prep_delivery", "halve_upstream_stigma")
SUPPLEMENT_SCENARIOS = ("ahs_no_worsening",)
