"""Hierarchical fit of one input draw: the nation, then each Census division followed by its states."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.optimization import OptimConfig
from config.shrinkage import ShrinkageConfig
from .jacobian import JacobianEstimator
from .transforms import FeatureTransformer


def shrinkage_reference(unit, config: ShrinkageConfig, national_J, division_J: dict):
    """Reference J and shrinkage weight for a unit.

    The nation is not shrunk; each division is shrunk toward the national J with
    weight wJ; each state toward its division's J with weight
    wJ_state * state_n_ref / n, where n is the state's respondent count (states
    sheet), so larger samples are shrunk less.
    """
    if unit.kind == "division" and national_J is not None:
        return national_J, config.wJ
    if unit.kind == "state":
        n = unit.sample_size or config.state_n_ref
        return division_J.get(unit.census_div), config.wJ_state * config.state_n_ref / max(n, 1.0)
    return None, None


@dataclass
class FitSettings:
    """Everything a hierarchical fit needs besides the data: optimizer and shrinkage settings, the
    pathway signs and priors (J[target, source]; persistence on the diagonal) and the drift priors."""
    opt: OptimConfig
    shrink: ShrinkageConfig
    sign: np.ndarray
    prior_mean: np.ndarray
    prior_tau: np.ndarray
    drift_enabled: np.ndarray
    drift_mean: np.ndarray
    drift_tau: np.ndarray


@dataclass(frozen=True)
class UnitFit:
    """One unit's fit in one draw."""
    J: np.ndarray
    anchor: np.ndarray
    drift: np.ndarray
    diagnostics: dict
    hierarchy_weight: float | None


def fit_hierarchy(units, unit_Y: list[np.ndarray], slopes: np.ndarray, settings: FitSettings,
                  SigmaY: np.ndarray, M: np.ndarray, source: int, target: int) -> list[UnitFit]:
    """Fit every unit of one draw in order, each centred at its own mean logit (state anchor), shrunk
    toward its parent's fitted J and held to its evidence target: the total response of ``target`` to
    ``source`` should match the unit's slope (models/dynamic/estimation/calibration.py).

    ``units`` must list each parent before its children (nation, then each division and its states);
    ``unit_Y`` holds each unit's observed proportions (measure x period) and ``slopes`` its evidence target.
    """
    opt = settings.opt
    fits, us_J, division_J = [], None, {}
    for unit, Y, slope in zip(units, unit_Y, slopes):
        anchor = FeatureTransformer.state_anchor(Y)
        Xbar, SigmaX = FeatureTransformer(reference_logits=anchor).transform(Y, SigmaY)
        J_ref, wJ = shrinkage_reference(unit, settings.shrink, us_J, division_J)
        result = JacobianEstimator(opt).fit(
            Xbar=Xbar, SigmaX=SigmaX, M=M, sign_mask=settings.sign, u=np.zeros(Y.shape[0]),
            J_ref=J_ref, wJ=wJ,
            structural_prior_mean=settings.prior_mean, structural_prior_tau=settings.prior_tau,
            drift_enabled=settings.drift_enabled, drift_prior_mean=settings.drift_mean,
            drift_prior_tau=settings.drift_tau,
            evidence_target={"source": source, "target": target, "horizon": opt.evidence_horizon,
                          "value": float(slope), "tau": opt.evidence_tau},
        )
        fits.append(UnitFit(result.J, anchor, result.drift, result.diagnostics, wJ))
        if unit.kind == "nation":
            us_J = result.J
        elif unit.kind == "division":
            division_J[unit.census_div] = result.J
    return fits
