# models/shared/alignment.py
"""Year alignment utilities."""

import numpy as np
from scipy.interpolate import interp1d

from models.shared.transforms import hazard_proxy


def align_to_years(
    years_src: np.ndarray,
    values_src: np.ndarray,
    years_tgt: np.ndarray,
) -> np.ndarray:
    """
    Interpolate/extrapolate values from source years to target years.
    
    Parameters
    ----------
    years_src : array (T_src,)
    values_src : array (T_src,) or (m, T_src)
    years_tgt : array (T_tgt,)
        
    Returns
    -------
    array (T_tgt,) or (m, T_tgt)
    """
    years_src = np.asarray(years_src, dtype=float)
    values_src = np.asarray(values_src, dtype=float)
    years_tgt = np.asarray(years_tgt, dtype=float)
    
    squeeze_output = False
    if values_src.ndim == 1:
        values_src = values_src[np.newaxis, :]
        squeeze_output = True
    
    idx = np.argsort(years_src)
    xs = years_src[idx]
    ys = values_src[:, idx]
    
    f = interp1d(xs, ys, axis=1, kind='linear', fill_value='extrapolate')
    out = f(years_tgt)
    
    if squeeze_output:
        out = out.squeeze(axis=0)
    
    return out


def align_to_years_hold_last(
    years_src: np.ndarray,
    values_src: np.ndarray,
    years_tgt: np.ndarray,
) -> np.ndarray:
    """
    Interpolate to target years but hold boundary values outside observed range.

    This is useful when extrapolation is undesirable for forecast years.
    """
    years_src = np.asarray(years_src, dtype=float).ravel()
    values_src = np.asarray(values_src, dtype=float).ravel()
    years_tgt = np.asarray(years_tgt, dtype=float).ravel()

    valid = ~np.isnan(values_src)
    if valid.sum() == 0:
        return np.full_like(years_tgt, np.nan, dtype=float)
    if valid.sum() == 1:
        return np.full_like(years_tgt, values_src[valid][0], dtype=float)

    xs = years_src[valid]
    ys = values_src[valid]
    idx = np.argsort(xs)
    xs = xs[idx]
    ys = ys[idx]

    f = interp1d(
        xs,
        ys,
        kind='linear',
        bounds_error=False,
        fill_value=(float(ys[0]), float(ys[-1])),
    )
    return np.asarray(f(years_tgt), dtype=float)

# models/shared/alignment.py

def extend_years(years: np.ndarray, target_len: int) -> np.ndarray:
    """Extend year array to target length."""
    ys = np.asarray(years, dtype=float).ravel()
    if ys.size >= target_len:
        return ys[:target_len]
    if ys.size == 0:
        return np.arange(target_len, dtype=float)
    step = float(ys[-1] - ys[-2]) if ys.size >= 2 else 1.0
    if step == 0:
        step = 1.0
    extra = ys[-1] + step * np.arange(1, target_len - ys.size + 1, dtype=float)
    return np.concatenate([ys, extra])


def extend_to_end_year(
    years: np.ndarray,
    target_end_year: int | None = None,
    step: int | None = None,
) -> np.ndarray:
    """Extend a year sequence to a target end year using fixed step size."""
    ys = np.asarray(years, dtype=int).ravel()
    if ys.size == 0 or target_end_year is None:
        return ys
    if ys[-1] >= int(target_end_year):
        return ys

    if step is None:
        step = int(ys[-1] - ys[-2]) if ys.size >= 2 else 1
    if step <= 0:
        step = 1

    extra = np.arange(ys[-1] + step, int(target_end_year) + 1, step, dtype=int)
    return np.concatenate([ys, extra])


def build_model_years(
    cdc_native_years: np.ndarray,
    target_end_year: int | None = None,
) -> np.ndarray:
    """Build canonical model years anchored to CDC native years (annual)."""
    return extend_to_end_year(cdc_native_years, target_end_year=target_end_year, step=1)


def build_cdc_inputs_from_sem(
    sem_traj: np.ndarray,
    unit,
    hivtest_idx: int,
    prep_idx: int,
    risk_idx: int,
    sem_years: np.ndarray,
    model_years: np.ndarray,
    n_elig_var: str,
    prevalence_var: str,
    viral_suppression_var: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned CDC input series from SEM trajectory and CDC raw data."""
    hivtest = sem_traj[hivtest_idx, :]
    prep_on = sem_traj[prep_idx, :]
    risk_behavior = sem_traj[risk_idx, :]

    hivtest = align_to_years(sem_years, hivtest, model_years)
    prep_on = align_to_years(sem_years, prep_on, model_years)
    risk_behavior = align_to_years(sem_years, risk_behavior, model_years)
    tau = hazard_proxy(hivtest)

    n_elig = np.asarray(unit.get_cdc(n_elig_var), dtype=float)
    if len(n_elig) != len(model_years):
        n_elig = align_to_years(unit.cdc_years, n_elig, model_years)

    prevalence = np.asarray(unit.get_cdc(prevalence_var), dtype=float)
    viral_suppression = np.asarray(unit.get_cdc(viral_suppression_var), dtype=float)
    no_vs = prevalence - viral_suppression
    if len(no_vs) != len(model_years):
        no_vs = align_to_years_hold_last(unit.cdc_years, no_vs, model_years)

    return tau, prep_on, n_elig, risk_behavior, no_vs
