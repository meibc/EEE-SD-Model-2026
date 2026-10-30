"""Year-grid alignment utilities shared by model fitting and projection."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def align_to_years(years_src: np.ndarray, values_src: np.ndarray, years_tgt: np.ndarray) -> np.ndarray:
    """Interpolate or extrapolate values onto target years along their final (period) axis.

    ``values_src`` may be period or measure x period; the returned array preserves any leading measure
    axis and replaces the final axis with the target periods.
    """
    years_src = np.asarray(years_src, dtype=float)
    values_src = np.asarray(values_src, dtype=float)
    years_tgt = np.asarray(years_tgt, dtype=float)

    squeeze_output = values_src.ndim == 1
    if squeeze_output:
        values_src = values_src[np.newaxis, :]

    order = np.argsort(years_src)
    interpolate = interp1d(
        years_src[order], values_src[:, order], axis=1, kind="linear", fill_value="extrapolate"
    )
    aligned = interpolate(years_tgt)
    return aligned.squeeze(axis=0) if squeeze_output else aligned


def extend_to_end_year(
    years: np.ndarray,
    target_end_year: int | None = None,
    step: int | None = None,
) -> np.ndarray:
    """Extend a year sequence to a target end year using a fixed step size."""
    years = np.asarray(years, dtype=int).ravel()
    if years.size == 0 or target_end_year is None or years[-1] >= int(target_end_year):
        return years

    if step is None:
        step = int(years[-1] - years[-2]) if years.size >= 2 else 1
    step = max(step, 1)
    extra = np.arange(years[-1] + step, int(target_end_year) + 1, step, dtype=int)
    return np.concatenate([years, extra])


def build_model_years(cdc_native_years: np.ndarray, target_end_year: int | None = None) -> np.ndarray:
    """Build annual model years anchored to the CDC source years."""
    return extend_to_end_year(cdc_native_years, target_end_year=target_end_year, step=1)
