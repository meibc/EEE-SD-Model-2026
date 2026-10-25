"""Structural coupling between two measures in a fitted transition matrix.

The coupling strength of ``target`` on ``source`` over H steps is the cumulative
centered-logit response of the target to a one-time unit impulse in the source,
with every other measure responding endogenously:

    S = sum_{h=1..H} e_target' J^h e_source.

It depends only on J (drift cancels because the system is linear and S is a
contrast). S is decomposed by the channel through which the effect first leaves
the source: directly to the target, or through another measure. Effects that
persist in the source, or return to it, are assigned to their first exit, so the
channel contributions sum exactly to S.
"""
from __future__ import annotations

import numpy as np

# Channels out of anticipated healthcare stigma (AHS) toward PrEP use, as reported.
AHS_PREP_CHANNELS = ("prep_used", "out_gid", "seehcp", "hivtest12")


def coupling_strength(J: np.ndarray, v_names: list[str], horizon: int, source: str = "stigma_ahs",
                      target: str = "prep_used", channels: tuple[str, ...] = AHS_PREP_CHANNELS) -> dict[str, np.ndarray]:
    """Total coupling and channel contributions for a stack of transition matrices.

    J has shape (draws, m, m) in J[target, source] orientation. Returns arrays of
    length ``draws`` keyed "total", each channel name, and "engagement" (the sum of
    the channels other than the direct one, ``target``). Raises if the channels do
    not sum to the total.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim == 2:
        J = J[None]
    p, a = v_names.index(target), v_names.index(source)
    powers = [np.repeat(np.eye(J.shape[-1])[None], J.shape[0], axis=0)]
    for _ in range(horizon):
        powers.append(powers[-1] @ J)
    powers = np.stack(powers)                                   # (H+1, draws, m, m)
    out = {"total": powers[1:, :, p, a].sum(axis=0)}

    stay = J[:, a, a][None, :] ** np.arange(horizon)[:, None]   # source persistence, (H, draws)
    cum_from = np.cumsum(powers[:, :, p, :], axis=0)            # cumulative target response to a unit on each node
    for node in channels:
        k = v_names.index(node)
        out[node] = sum(stay[j] * J[:, k, a] * cum_from[horizon - j - 1, :, k] for j in range(horizon))
    out["engagement"] = sum(out[n] for n in channels if n != target)
    gap = np.nanmax(np.abs(out["total"] - out[target] - out["engagement"]))
    if gap > 1e-8:
        raise AssertionError(f"Channel decomposition does not sum to the total (max gap {gap:.2e})")
    return out


def source_history_counterfactual(J: np.ndarray, drift: np.ndarray, X_obs: np.ndarray, v_names: list[str],
                                  source: str = "stigma_ahs", extra_steps: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fitted trajectories with the source on its observed path and held at its first value.

    J (draws, m, m) and drift (draws, m) are the fitted system; X_obs (draws, m, T) the observed
    centered logits. Both runs start from the first observed period and step
    x[t+1] = J x[t] + drift, with the source overwritten each step by its observed value
    (factual) or by its first observed value (held); every other measure evolves through the
    fitted pathways. ``extra_steps`` continues both runs past the last observed period with the
    source at its last (factual) or first (held) observed value, giving the effect the observed
    change has already set in motion. Returns (factual, held), each (draws, m, T + extra_steps).
    Their difference is the part of each trajectory attributable to the observed change in the
    source; the drift cancels.
    """
    J, drift, X_obs = (np.asarray(a, dtype=float) for a in (J, drift, X_obs))
    a = v_names.index(source)
    T = X_obs.shape[2]
    shape = (X_obs.shape[0], X_obs.shape[1], T + extra_steps)
    factual, held = np.empty(shape), np.empty(shape)
    factual[:, :, 0] = held[:, :, 0] = X_obs[:, :, 0]
    for t in range(1, T + extra_steps):
        factual[:, :, t] = np.einsum("dij,dj->di", J, factual[:, :, t - 1]) + drift
        held[:, :, t] = np.einsum("dij,dj->di", J, held[:, :, t - 1]) + drift
        factual[:, a, t] = X_obs[:, a, min(t, T - 1)]
        held[:, a, t] = X_obs[:, a, 0]
    return factual, held


def source_free_counterfactual(J: np.ndarray, drift: np.ndarray, X_obs: np.ndarray, v_names: list[str],
                               benchmark: np.ndarray, source: str = "stigma_ahs",
                               extra_steps: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fitted trajectories with the source as observed and with it held at a benchmark.

    Both runs start from the first observed period and step x[t+1] = J x[t] + drift. In the
    factual run the source follows its observed values and, after the last observed period,
    evolves through the fitted system; in the counterfactual it is held at ``benchmark``
    (draws,) from the first period on. Every other measure responds through the fitted
    pathways. Returns (factual, held), each (draws, m, T + extra_steps).
    """
    J, drift, X_obs = (np.asarray(a, dtype=float) for a in (J, drift, X_obs))
    a = v_names.index(source)
    T = X_obs.shape[2]
    shape = (X_obs.shape[0], X_obs.shape[1], T + extra_steps)
    factual, held = np.empty(shape), np.empty(shape)
    factual[:, :, 0] = held[:, :, 0] = X_obs[:, :, 0]
    held[:, a, 0] = benchmark
    for t in range(1, T + extra_steps):
        factual[:, :, t] = np.einsum("dij,dj->di", J, factual[:, :, t - 1]) + drift
        held[:, :, t] = np.einsum("dij,dj->di", J, held[:, :, t - 1]) + drift
        if t < T:
            factual[:, a, t] = X_obs[:, a, t]
        held[:, a, t] = benchmark
    return factual, held


def trajectory_decomposition(J: np.ndarray, drift: np.ndarray, x0: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """Exact decomposition of the projected change of every measure over ``steps`` steps.

    From x[0] = x0, x[t+1] = J x[t] + drift gives

        x[n] - x0 = (J^n - I) x0 + sum_{k<n} J^k drift,

    so each measure's change is the sum of
      carryover_own     its own departure from its anchor decaying through its own persistence,
                        (J_ii^n - 1) x0_i
      carryover_system  the other measures' departures reaching it, (J^n x0)_i - J_ii^n x0_i
      drift_own         its own drift accumulated through its own persistence, sum_k J_ii^k drift_i
      drift_by_source   [..., i, j] = sum_k (J^k)_ij drift_j: measure j's drift reaching measure i
                        through the whole system (for j = i, minus drift_own: its own drift
                        amplified or damped by feedback loops)
    J (draws, m, m), drift and x0 (draws, m). Every array has a leading draw axis; the parts sum to
    "total" exactly.
    """
    J, drift, x0 = (np.asarray(a, dtype=float) for a in (J, drift, x0))
    draws, m, _ = J.shape
    power = np.broadcast_to(np.eye(m), J.shape).copy()             # J^k, starting at k = 0
    cumulative = np.zeros_like(J)                                 # sum_{k<n} J^k
    for _ in range(steps):
        cumulative += power
        power = power @ J
    diag = np.einsum("dii->di", J)
    own_power = diag ** steps
    own_cumulative = np.where(np.isclose(diag, 1.0), steps, (1 - diag ** steps) / np.where(np.isclose(diag, 1.0), 1.0, 1 - diag))
    total = np.einsum("dij,dj->di", power, x0) - x0 + np.einsum("dij,dj->di", cumulative, drift)
    carryover_own = (own_power - 1) * x0
    carryover_system = np.einsum("dij,dj->di", power, x0) - own_power * x0
    drift_own = own_cumulative * drift
    drift_by_source = cumulative * drift[:, None, :]
    idx = np.arange(m)
    drift_by_source[:, idx, idx] -= drift_own
    parts = carryover_own + carryover_system + drift_own + drift_by_source.sum(axis=2)
    if not np.allclose(parts, total):
        raise RuntimeError("Trajectory decomposition does not sum to the projected change")
    return {"total": total, "carryover_own": carryover_own, "carryover_system": carryover_system,
            "drift_own": drift_own, "drift_by_source": drift_by_source}
