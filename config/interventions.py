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

# Elimination: AHS brought to the benchmark (the AHS reference probability, 5th percentile of
# state-period estimates), closing the whole gap above it with the same ramp.
def _ahs_benchmark() -> float:
    from config.sem import SEMConfig
    config = SEMConfig()
    return float(config.reference_probs()[config.v_names.index("stigma_ahs")])


INTERVENTION_CODEBOOK["eliminate_ahs"] = {
    "var": "stigma_ahs",
    "delta": -1.0,
    "effect": "probability_gap_to_benchmark",
    "benchmark": _ahs_benchmark(),
    "label": "Bring anticipated healthcare stigma to the benchmark (5th percentile)",
}
SCENARIO_CODEBOOK["s11_eliminate_ahs"] = {
    "state_codes": ["eliminate_ahs"],
    "relationship_codes": [],
    "label": "Eliminate anticipated healthcare stigma (to benchmark)",
}

# Paper scenarios (2026-10-24). Cuts are set relative to each draw's own 2024 (last observed)
# level and held there, never above the baseline path, so their size does not depend on the
# projected baseline. The benchmark scenario brings AHS to the 5th percentile of observed
# state-period levels. Resilient PrEP delivery is a derived policy comparison: half of the
# benchmark scenario's PrEP change, with AHS and every other measure at baseline.
for _short, _var in (("ahs", "stigma_ahs"), ("gss", "stigma_gss"), ("family_stigma", "stigma_family")):
    INTERVENTION_CODEBOOK[f"halve_{_short}_from_2024"] = {
        "var": _var,
        "delta": -0.5,
        "effect": "probability_relative_to_last_observed",
        "label": f"Reduce {_var} to half its 2024 level",
    }
INTERVENTION_CODEBOOK["hold_ahs_at_2024"] = {
    "var": "stigma_ahs",
    "delta": 0.0,
    "effect": "probability_relative_to_last_observed",
    "label": "Hold anticipated healthcare stigma at or below its 2024 level",
}
SCENARIO_CODEBOOK.update({
    "halve_ahs": {"state_codes": ["halve_ahs_from_2024"], "relationship_codes": [],
                  "label": "Halve anticipated healthcare stigma from its 2024 level"},
    "ahs_to_benchmark": {"state_codes": ["eliminate_ahs"], "relationship_codes": [],
                         "label": "Reduce anticipated healthcare stigma to the 5th percentile of observed levels"},
    "resilient_prep_delivery": {"state_codes": ["eliminate_ahs"], "relationship_codes": [], "prep_only_fraction": 0.5,
                                "label": "Stigma-resilient PrEP delivery: half the AHS-associated PrEP gap closed, AHS unchanged"},
    "halve_upstream_stigma": {"state_codes": ["halve_gss_from_2024", "halve_family_stigma_from_2024"], "relationship_codes": [],
                              "label": "Halve general social and family stigma from their 2024 levels"},
    "ahs_no_worsening": {"state_codes": ["hold_ahs_at_2024"], "relationship_codes": [],
                         "label": "Anticipated healthcare stigma held at or below its 2024 level"},
})
PAPER_SCENARIOS = ("halve_ahs", "ahs_to_benchmark", "resilient_prep_delivery", "halve_upstream_stigma")
SUPPLEMENT_SCENARIOS = ("ahs_no_worsening",)

# Shifted versions of the primary scenarios: the stigma measure's set point is lowered
# (``probability_relative_shift``) instead of the measure being pinned to 50% of baseline,
# so feedback onto the measure (e.g. PrEP or healthcare contact -> AHS) keeps operating.
for _code, _spec in list(INTERVENTION_CODEBOOK.items()):
    if _spec["effect"] == "probability_relative_to_baseline":
        INTERVENTION_CODEBOOK[f"{_code}_shift"] = {**_spec, "effect": "probability_relative_shift",
                                                   "label": _spec["label"] + " (shifted set point)"}
for _code in PRIMARY_SCENARIOS:
    _spec = SCENARIO_CODEBOOK[_code]
    SCENARIO_CODEBOOK[f"{_code}_shift"] = {
        "state_codes": [f"{c}_shift" for c in _spec["state_codes"]],
        "relationship_codes": list(_spec["relationship_codes"]),
        "label": _spec["label"] + " (shifted set point)",
    }
SHIFT_SCENARIOS = tuple(f"{c}_shift" for c in PRIMARY_SCENARIOS)

# Fixed-dose sensitivity (scripts/12): the same stigma measures reduced by a fixed number of
# percentage points below each draw's baseline, instead of by 50% of it, with the same ramp.
# The doses are derived from the primary scenario results, so scenarios are registered at run time.
FIXED_DOSE_STIGMA = {"ahs": "stigma_ahs", "gss": "stigma_gss", "family": "stigma_family"}


def register_fixed_dose_scenarios(doses: dict[str, dict[str, float]]) -> list[str]:
    """Add fixed-dose scenarios to the codebooks and return their codes.

    ``doses`` maps a dose label to the reduction (probability scale) for each stigma
    measure, e.g. {"matched": {"stigma_ahs": 0.08, "stigma_gss": 0.20, "stigma_family": 0.15}}.
    Each label gives abs_<label>_ahs (AHS only) and abs_<label>_combined (all stigma levels).
    """
    codes = []
    for label, amounts in doses.items():
        for short, var in FIXED_DOSE_STIGMA.items():
            INTERVENTION_CODEBOOK[f"reduce_{short}_abs_{label}"] = {
                "var": var,
                "delta": -float(amounts[var]),
                "effect": "probability_absolute_to_baseline",
                "label": f"Reduce {var} by {100 * amounts[var]:.1f} percentage points",
            }
        SCENARIO_CODEBOOK[f"abs_{label}_ahs"] = {
            "state_codes": [f"reduce_ahs_abs_{label}"],
            "relationship_codes": [],
            "label": f"Reduce anticipated healthcare stigma by a fixed dose ({label})",
        }
        SCENARIO_CODEBOOK[f"abs_{label}_combined"] = {
            "state_codes": [f"reduce_{s}_abs_{label}" for s in FIXED_DOSE_STIGMA],
            "relationship_codes": [],
            "label": f"All stigma levels reduced by a fixed dose ({label})",
        }
        codes += [f"abs_{label}_ahs", f"abs_{label}_combined"]
    return codes
