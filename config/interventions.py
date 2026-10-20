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
    "weaken_ahs_to_prep": {
        "from": "stigma_ahs",
        "to": "prep_used",
        "delta": 0.5,
        "effect": "attenuate",
        "label": "Weaken AHS -> PrEP pathway by 50%",
    },
    "weaken_ahs_to_hivtest": {
        "from": "stigma_ahs",
        "to": "hivtest12",
        "delta": 0.5,
        "effect": "attenuate",
        "label": "Weaken AHS -> HIV testing pathway by 50%",
    },
    "strengthen_seehcp_to_lower_ahs": {
        "from": "seehcp",
        "to": "stigma_ahs",
        "delta": 0.5,
        "effect": "multiplicative",
        "label": "Strengthen healthcare -> lower AHS pathway by 50%",
    },
    "attenuate_healthcare_generated_ahs": {
        "from": "seehcp",
        "to": "stigma_ahs",
        "delta": 0.5,
        "effect": "attenuate_positive",
        "label": "Reduce healthcare-generated AHS pathway by 50%",
    },
    "reverse_healthcare_generated_ahs": {
        "from": "seehcp",
        "to": "stigma_ahs",
        "delta": 1.0,
        "effect": "positive_to_target",
        "target": -0.087,
        "label": "Reverse healthcare-generated AHS toward typical protective state effect",
    },
}

SCENARIO_CODEBOOK = {
    "s1_reduce_ahs": {
        "state_codes": ["reduce_ahs"],
        "relationship_codes": [],
        "label": "Reduce anticipated healthcare stigma",
    },
    "s2_reduce_gss": {
        "state_codes": ["reduce_gss"],
        "relationship_codes": [],
        "label": "Reduce general social stigma",
    },
    "s3_reduce_family_stigma": {
        "state_codes": ["reduce_family_stigma"],
        "relationship_codes": [],
        "label": "Reduce family stigma",
    },
    "s4_weaken_ahs_to_prep": {
        "state_codes": [],
        "relationship_codes": ["weaken_ahs_to_prep"],
        "label": "Weaken AHS -> PrEP pathway",
    },
    "s5_weaken_ahs_to_hivtest": {
        "state_codes": [],
        "relationship_codes": ["weaken_ahs_to_hivtest"],
        "label": "Weaken AHS -> HIV testing pathway",
    },
    "s6_strengthen_seehcp_to_lower_ahs": {
        "state_codes": [],
        "relationship_codes": ["strengthen_seehcp_to_lower_ahs"],
        "label": "Strengthen healthcare -> lower AHS pathway",
    },
    "s7_combined_stigma_package": {
        "state_codes": ["reduce_ahs", "reduce_gss", "reduce_family_stigma"],
        "relationship_codes": [],
        "label": "All stigma levels",
    },
    "s8_attenuate_healthcare_generated_ahs": {
        "state_codes": [],
        "relationship_codes": ["attenuate_healthcare_generated_ahs"],
        "label": "Reduce healthcare-generated AHS",
    },
    "s9_reverse_healthcare_generated_ahs": {
        "state_codes": [],
        "relationship_codes": ["reverse_healthcare_generated_ahs"],
        "label": "Reverse healthcare-generated AHS toward typical protective state effect",
    },
}

# Primary paper scenarios; relationship interventions remain separate diagnostics.
SCENARIO_CODEBOOK["s10_upstream_social_stigma"] = {
    "state_codes": ["reduce_gss", "reduce_family_stigma"],
    "relationship_codes": [],
    "label": "Upstream social stigma",
}
PRIMARY_SCENARIOS = ("s1_reduce_ahs", "s10_upstream_social_stigma", "s7_combined_stigma_package")
