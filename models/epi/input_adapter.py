"""Epidemic-model inputs from a projected dynamic-model trajectory and the CDC stocks."""
from __future__ import annotations

import numpy as np

from models.epi.params import EpiInputs
from models.shared.alignment import align_to_years


def epi_inputs_from_dynamic(
    dynamic_traj: np.ndarray,
    unit,
    hivtest_idx: int,
    prep_idx: int,
    risk_idx: int,
    dynamic_years: np.ndarray,
    model_years: np.ndarray,
    n_elig_var: str,
    prevalence_var: str = "Estimated HIV prevalence (MSM)",
    viral_suppression_var: str = "HIV viral suppression",
) -> EpiInputs:
    """Epidemic-model inputs from a projected dynamic-model trajectory and the CDC stocks.

    Testing is the projected past-year HIV testing probability (hivtest12), as in the calibration. CDC stocks
    (PrEP-eligible and not virally suppressed) are interpolated within the CDC
    years and held at the last observed year beyond them.
    """
    hivtest = dynamic_traj[hivtest_idx, :]
    prep_on = dynamic_traj[prep_idx, :]
    risk_behavior = dynamic_traj[risk_idx, :]

    hivtest = align_to_years(dynamic_years, hivtest, model_years)
    prep_on = align_to_years(dynamic_years, prep_on, model_years)
    risk_behavior = align_to_years(dynamic_years, risk_behavior, model_years)
    hiv_testing = np.clip(hivtest, 1e-6, 1 - 1e-6)

    n_elig = np.asarray(unit.get_cdc(n_elig_var), dtype=float)
    if len(n_elig) != len(model_years):
        n_elig = np.interp(model_years, unit.cdc_years, n_elig)

    prevalence = np.asarray(unit.get_cdc(prevalence_var), dtype=float)
    viral_suppression = np.asarray(unit.get_cdc(viral_suppression_var), dtype=float)
    no_vs = np.maximum(0.0, prevalence - viral_suppression)
    if len(no_vs) != len(model_years):
        no_vs = np.interp(model_years, unit.cdc_years, no_vs)

    return EpiInputs(years=model_years, hiv_testing=hiv_testing, prep_on=prep_on, N_elig=n_elig, risk_behavior=risk_behavior,
                     no_vs=no_vs)
