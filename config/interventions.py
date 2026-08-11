"""Intervention codebooks (scenario definitions)."""

INTERVENTION_CODEBOOK = {
    "reduce_gss": {
        "var": "stigma_gss",
        "delta": -1.0,
        "mode": "linear",
        "space": "logit",
        "label": "Reduce general social stigma",
    },
    "reduce_family_stigma": {
        "var": "stigma_family",
        "delta": -1.0,
        "mode": "linear",
        "space": "logit",
    },
    "reduce_ahs": {
        "var": "stigma_ahs",
        "delta": -1.0,
        "mode": "linear",
        "space": "logit",
    },
    "increase_seehcp": {
        "var": "seehcp",
        "delta": 0.3,
        "mode": "linear",
        "space": "logit",
    },
    "reduce_risk": {
        "var": "risk_behavior",
        "delta": -0.5,
        "mode": "linear",
        "space": "logit",
    },
}

REL_CODEBOOK = {
    "weaken_stigma_to_care": {
        "from": "stigma_ahs",
        "to": "seehcp",
        "delta": -1.5,
    },
    "weaken_stigma_to_prep": {
        "from": "stigma_ahs",
        "to": "prep_used",
        "delta": -0.5,
    },
    "strengthen_care_feedback": {
        "from": "seehcp",
        "to": "stigma_ahs",
        "delta": -0.2,
    },
    "strengthen_prep_feedback": {
        "from": "prep_used",
        "to": "stigma_ahs",
        "delta": -0.2,
    },
    "weaken_stigma_to_hivtest": {
        "from": "stigma_ahs",
        "to": "hivtest12",
        "delta": -0.5,
    },
}


# Named multi-scenario intervention bundles for compare mode.
# Each scenario references existing state and relationship codes.
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
    "s4_increase_seehcp": {
        "state_codes": ["increase_seehcp"],
        "relationship_codes": [],
        "label": "Increase healthcare contact",
    },
    "s5_reduce_risk": {
        "state_codes": ["reduce_risk"],
        "relationship_codes": [],
        "label": "Reduce risk behavior",
    },
    "s6_weaken_stigma_to_care": {
        "state_codes": [],
        "relationship_codes": ["weaken_stigma_to_care"],
        "label": "Weaken stigma -> care pathway",
    },
    "s7_weaken_stigma_to_prep": {
        "state_codes": [],
        "relationship_codes": ["weaken_stigma_to_prep"],
        "label": "Weaken stigma -> prep pathway",
    },
    "s8_weaken_stigma_to_hivtest": {
        "state_codes": [],
        "relationship_codes": ["weaken_stigma_to_hivtest"],
        "label": "Weaken stigma -> hiv test pathway",
    },
    "s9_combined_stigma_package": {
        "state_codes": ["reduce_ahs", "reduce_gss", "reduce_family_stigma"],
        "relationship_codes": [
            "weaken_stigma_to_care",
            "weaken_stigma_to_prep",
            "weaken_stigma_to_hivtest",
        ],
        "label": "Combined stigma package",
    },
}
