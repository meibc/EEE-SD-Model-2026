"""Intervention codebooks for the centered-logit drift model."""

INTERVENTION_CODEBOOK = {
    "reduce_ahs": {
        "var": "stigma_ahs",
        "delta": -0.5,
        "effect": "probability_relative_to_baseline",
        "label": "Reduce anticipated healthcare stigma by 50%",
    },
    "reduce_gss": {
        "var": "stigma_gss",
        "delta": -0.5,
        "effect": "probability_relative_to_baseline",
        "label": "Reduce general social stigma by 50%",
    },
    "reduce_family_stigma": {
        "var": "stigma_family",
        "delta": -0.5,
        "effect": "probability_relative_to_baseline",
        "label": "Reduce family stigma by 50%",
    },
}

REL_CODEBOOK = {
    "weaken_stigma_to_prep": {
        "from": "stigma_ahs",
        "to": "prep_used",
        "delta": 0.5,
        "effect": "attenuate",
        "label": "Weaken AHS -> PrEP pathway by 50%",
    },
    "weaken_stigma_to_disclosure": {
        "from": "stigma_ahs",
        "to": "out_gid",
        "delta": 0.5,
        "effect": "attenuate",
        "label": "Weaken AHS -> outness pathway by 50%",
    },
    "strengthen_outness_to_prep": {
        "from": "out_gid",
        "to": "prep_used",
        "delta": 0.5,
        "effect": "multiplicative",
        "label": "Strengthen outness -> PrEP linkage by 50%",
    },
    "strengthen_outness_to_hivtest": {
        "from": "out_gid",
        "to": "hivtest12",
        "delta": 0.5,
        "effect": "multiplicative",
        "label": "Strengthen outness -> HIV testing linkage by 50%",
    },
    "strengthen_seehcp_to_hivtest": {
        "from": "seehcp",
        "to": "hivtest12",
        "delta": 0.5,
        "effect": "multiplicative",
        "label": "Strengthen healthcare -> HIV testing linkage by 50%",
    },
}

SCENARIO_CODEBOOK = {
    "reduce_ahs_50pct": {
        "state_codes": ["reduce_ahs"],
        "relationship_codes": [],
        "label": "Reduce anticipated healthcare stigma by 50%",
    },
    "reduce_gss_50pct": {
        "state_codes": ["reduce_gss"],
        "relationship_codes": [],
        "label": "Reduce general social stigma by 50%",
    },
    "reduce_family_stigma_50pct": {
        "state_codes": ["reduce_family_stigma"],
        "relationship_codes": [],
        "label": "Reduce family stigma by 50%",
    },
    "weaken_ahs_to_prep_50pct": {
        "state_codes": [],
        "relationship_codes": ["weaken_stigma_to_prep"],
        "label": "Weaken AHS -> PrEP pathway by 50%",
    },
    "combined_stigma_package": {
        "state_codes": ["reduce_ahs", "reduce_gss", "reduce_family_stigma"],
        "relationship_codes": ["weaken_stigma_to_prep"],
        "label": "Combined stigma package",
    },
    "strengthen_outness_to_prep_50pct": {
        "state_codes": [],
        "relationship_codes": ["strengthen_outness_to_prep"],
        "label": "Strengthen outness -> PrEP linkage by 50%",
    },
    "strengthen_outness_to_hivtest_50pct": {
        "state_codes": [],
        "relationship_codes": ["strengthen_outness_to_hivtest"],
        "label": "Strengthen outness -> HIV testing linkage by 50%",
    },
    "strengthen_seehcp_to_hivtest_50pct": {
        "state_codes": [],
        "relationship_codes": ["strengthen_seehcp_to_hivtest"],
        "label": "Strengthen healthcare -> HIV testing linkage by 50%",
    },
    "combined_disclosure_prevention_package": {
        "state_codes": [],
        "relationship_codes": [
            "weaken_stigma_to_disclosure",
            "strengthen_outness_to_prep",
            "strengthen_outness_to_hivtest",
        ],
        "label": "Disclosure-enabled prevention package",
    },
}
