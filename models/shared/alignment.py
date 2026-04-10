# models/shared/alignment.py
"""Year alignment utilities."""

import numpy as np
from scipy.interpolate import interp1d


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