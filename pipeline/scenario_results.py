from __future__ import annotations

"""Paired baseline-scenario endpoint extraction and summaries.

``pipeline/projection.py`` produces the Monte Carlo samples represented here.
``pipeline/scenarios.py`` uses the helpers below to extract report-year values
and summarize paired baseline-scenario contrasts without knowing the internal
shape of each projection result.
"""

import numpy as np
from pipeline.projection import ProjectionOutput, ProjectionResult


EPI_ENDPOINTS = ("incidence", "diagnosed", "undiagnosed", "prep_on_count", "incidence_cumulative")


def _idx_for_year(years: np.ndarray, target_year: int) -> int:
    years = np.asarray(years, dtype=int).ravel()
    idx = np.where(years == int(target_year))[0]
    if idx.size == 0:
        raise ValueError(f"target_year={target_year} absent from years={years.tolist()}")
    return int(idx[0])


def _epi_endpoint(result: ProjectionResult, variable: str, endpoint_idx: int,
                  cumulative_start_idx: int) -> np.ndarray:
    """Return one epidemic endpoint per draw.

    ``incidence_cumulative`` is constructed by summing annual incidence from
    the configured cumulative start year through the endpoint year.
    """
    if variable == "incidence_cumulative":
        return result.epi_stack("incidence")[:, cumulative_start_idx:endpoint_idx + 1].sum(axis=1)
    return result.epi_stack(variable)[:, endpoint_idx]


def paired_endpoints(baseline: ProjectionOutput, scenario: ProjectionOutput,
                     dynamic_years: np.ndarray, target_year: int,
                     cumulative_start_year: int) -> dict[str, tuple[dict, dict]]:
    """Extract paired dynamic and epidemic values at one report year.

    Baseline and scenario runs use the same seed and sample count, so draw
    ``k`` in each output contains the same dynamic and epidemic parameter draws.
    """
    epi_idx = _idx_for_year(baseline.years, target_year)
    cumulative_idx = _idx_for_year(baseline.years, cumulative_start_year)
    dynamic_idx = _idx_for_year(dynamic_years, target_year)
    endpoints = {}
    for unit_id in sorted(set(baseline.results) & set(scenario.results)):
        baseline_result = baseline.results[unit_id]
        scenario_result = scenario.results[unit_id]
        baseline_dynamic = baseline_result.dynamic_stack()
        scenario_dynamic = scenario_result.dynamic_stack()
        dynamic = {
            variable: (baseline_dynamic[:, k, dynamic_idx], scenario_dynamic[:, k, dynamic_idx])
            for k, variable in enumerate(baseline.v_names)
        }
        epidemic = {
            variable: (
                _epi_endpoint(baseline_result, variable, epi_idx, cumulative_idx),
                _epi_endpoint(scenario_result, variable, epi_idx, cumulative_idx),
            )
            for variable in EPI_ENDPOINTS
        }
        endpoints[unit_id] = (dynamic, epidemic)
    return endpoints


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "p025": float(np.nanquantile(values, 0.025)),
        "p50": float(np.nanquantile(values, 0.5)),
        "p975": float(np.nanquantile(values, 0.975)),
    }


def _percent_change(scenario: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    scenario = np.asarray(scenario, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    return np.where(np.abs(baseline) > 1e-12, 100.0 * (scenario - baseline) / baseline, np.nan)


def summary_rows(endpoints: dict, labels: dict) -> tuple[list[dict], list[dict]]:
    """Summarize paired endpoints by their median and 95% interval."""
    dynamic_rows, epidemic_rows = [], []
    for unit_id, (dynamic, epidemic) in endpoints.items():
        for rows, values in ((dynamic_rows, dynamic), (epidemic_rows, epidemic)):
            for variable, (baseline, scenario) in values.items():
                rows.append({
                    "unit_id": unit_id,
                    **labels,
                    "variable": variable,
                    **{f"baseline_{key}": value for key, value in _quantiles(baseline).items()},
                    **{f"scenario_{key}": value for key, value in _quantiles(scenario).items()},
                    **{f"delta_{key}": value for key, value in _quantiles(scenario - baseline).items()},
                    **{
                        f"pct_change_{key}": value
                        for key, value in _quantiles(_percent_change(scenario, baseline)).items()
                    },
                })
    return dynamic_rows, epidemic_rows


def draw_rows(endpoints: dict, labels: dict) -> tuple[list[dict], list[dict]]:
    """Return the paired endpoint values as one row per draw."""
    dynamic_rows, epidemic_rows = [], []
    for unit_id, (dynamic, epidemic) in endpoints.items():
        for rows, values in ((dynamic_rows, dynamic), (epidemic_rows, epidemic)):
            percent_changes = {
                variable: _percent_change(scenario, baseline)
                for variable, (baseline, scenario) in values.items()
            }
            n_draws = len(next(iter(values.values()))[0])
            for draw in range(n_draws):
                for variable, (baseline, scenario) in values.items():
                    rows.append({
                        "unit_id": unit_id,
                        **labels,
                        "draw": int(draw),
                        "variable": variable,
                        "baseline": float(baseline[draw]),
                        "scenario_value": float(scenario[draw]),
                        "delta": float(scenario[draw]) - float(baseline[draw]),
                        "pct_change": float(percent_changes[variable][draw]),
                    })
    return dynamic_rows, epidemic_rows


