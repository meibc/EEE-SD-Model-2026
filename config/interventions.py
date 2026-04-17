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
