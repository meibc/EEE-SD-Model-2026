"""Evidence target for the total effect of anticipated healthcare stigma (AHS) on PrEP.

The individual-level, access-adjusted prevalence ratio r (scripts/2_evidence/01) is a total association:
it was not adjusted for disclosure, healthcare contact or testing, so it includes the routes
through them. At a unit's AHS share A and PrEP level P, PrEP among men without AHS is
P0 = P / (1 - A (1 - r)), and moving AHS from A to a benchmark B moves PrEP to P0 (1 - B (1 - r)).
The implied response of PrEP to AHS on the centred-logit scale is the secant

    slope = [logit P(B) - logit P] / [logit B - logit A]          (``evidence_slope``)

The fit (models/dynamic/estimation/loss.py) asks the model's own total response to match it: under a
sustained shift of AHS's set point (a constant input to the AHS equation), every route and
feedback acting, the change in PrEP per unit change in AHS after ``horizon`` steps
(``sustained_shift_response``). The drift cancels because the response is a difference between runs.
"""
from __future__ import annotations

import numpy as np

from core.transforms import PROB_EPS, logit


def evidence_slope(ahs: float, prep: float, apr: float, benchmark: float) -> float:
    """Centred-logit response of PrEP to AHS implied by the prevalence ratio."""
    if not (0 < ahs < 1 and 0 < prep < 1 and 0 < benchmark < 1 and apr > 0):
        raise ValueError(f"Invalid evidence inputs: AHS {ahs}, PrEP {prep}, benchmark {benchmark}, ratio {apr}")
    p0 = prep / (1 - ahs * (1 - apr))
    if not 0 < p0 < 1:
        raise ValueError(f"PrEP among men without AHS outside (0, 1): {p0} (AHS {ahs}, PrEP {prep}, ratio {apr})")
    if abs(logit(benchmark, PROB_EPS) - logit(ahs, PROB_EPS)) < 1e-3:
        # Derivative at A when the benchmark is (nearly) the current level.
        return float(-p0 * (1 - apr) * ahs * (1 - ahs) / (prep * (1 - prep)))
    prep_benchmark = p0 * (1 - benchmark * (1 - apr))
    return float((logit(prep_benchmark, PROB_EPS) - logit(prep, PROB_EPS))
                 / (logit(benchmark, PROB_EPS) - logit(ahs, PROB_EPS)))


def sustained_shift_response(J: np.ndarray, source: int, target: int, horizon: int) -> float:
    """Change in ``target`` per unit change in ``source`` after ``horizon`` steps of a sustained
    unit input to the source's equation (d[t+1] = J d[t] + e_source, d[0] = 0)."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    d = np.zeros(J.shape[0])
    e = np.zeros(J.shape[0])
    e[source] = 1.0
    for _ in range(horizon):
        d = J @ d + e
    if d[source] == 0:
        raise ValueError("The source's own response is zero; the ratio is undefined")
    return float(d[target] / d[source])


def sustained_shift_response_grad(J: np.ndarray, source: int, target: int, horizon: int) -> tuple[float, np.ndarray]:
    """``sustained_shift_response`` and its gradient with respect to J (m x m), analytically.

    d = sum_{k<H} J^k e; d d[q] / d J[i, j] = sum_{l + m <= H - 2} (J^l)[q, i] (J^m e)[j].
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    n = J.shape[0]
    e = np.zeros(n)
    e[source] = 1.0
    powers = [np.eye(n)]
    vectors = [e]
    for _ in range(horizon - 1):
        powers.append(powers[-1] @ J)
        vectors.append(J @ vectors[-1])
    d = np.sum(vectors, axis=0)
    cum_v = np.cumsum(vectors, axis=0)                 # cum_v[k] = sum_{m<=k} J^m e

    def grad_of(q: int) -> np.ndarray:
        g = np.zeros((n, n))
        for l in range(horizon - 1):
            g += np.outer(powers[l][q], cum_v[horizon - 2 - l])
        return g

    ratio = d[target] / d[source]
    grad = (grad_of(target) * d[source] - d[target] * grad_of(source)) / d[source] ** 2
    return float(ratio), grad
