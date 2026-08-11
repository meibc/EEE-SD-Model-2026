from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.results import JointOutput, RunOutput, UncertaintyOutput


EPI_VARS = ("incidence", "diagnosed", "prep_on_count")


def _pct_change(baseline: float, scenario: float) -> float:
    if np.isnan(baseline) or np.isnan(scenario):
        return np.nan
    if abs(baseline) < 1e-12:
        return np.nan
    return 100.0 * (scenario - baseline) / baseline


def _idx_for_year(years: np.ndarray, target_year: int) -> int:
    years_i = np.asarray(years, dtype=int).ravel()
    idx = np.where(years_i == int(target_year))[0]
    if idx.size == 0:
        raise ValueError(f"target_year={target_year} not found in years={years_i.tolist()}")
    return int(idx[0])


def build_deterministic_year10_tables(
    sem_output: RunOutput,
    baseline: JointOutput,
    interventions: dict[str, JointOutput],
    target_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sem_idx = _idx_for_year(baseline.sem_years, target_year)
    cdc_idx = _idx_for_year(baseline.cdc_years, target_year)
    sem_names = list(sem_output.inputs.v_names)

    sem_rows: list[dict] = []
    epi_rows: list[dict] = []

    for scenario_name, scenario_out in interventions.items():
        common_units = sorted(set(baseline.results) & set(scenario_out.results))
        for unit_id in common_units:
            base_res = baseline.results[unit_id]
            scen_res = scenario_out.results[unit_id]

            for i, var in enumerate(sem_names):
                b = float(base_res.sem_trajectory[i, sem_idx])
                s = float(scen_res.sem_trajectory[i, sem_idx])
                sem_rows.append(
                    {
                        "unit_id": unit_id,
                        "scenario": scenario_name,
                        "year": int(target_year),
                        "variable": var,
                        "baseline_value": b,
                        "scenario_value": s,
                        "pct_change": _pct_change(b, s),
                    }
                )

            for var in EPI_VARS:
                b = float(getattr(base_res.cdc_output, var)[cdc_idx])
                s = float(getattr(scen_res.cdc_output, var)[cdc_idx])
                epi_rows.append(
                    {
                        "unit_id": unit_id,
                        "scenario": scenario_name,
                        "year": int(target_year),
                        "variable": var,
                        "baseline_value": b,
                        "scenario_value": s,
                        "pct_change": _pct_change(b, s),
                    }
                )

    return pd.DataFrame(sem_rows), pd.DataFrame(epi_rows)


def build_uncertainty_year10_tables(
    sem_output: RunOutput,
    baseline: UncertaintyOutput,
    interventions: dict[str, UncertaintyOutput],
    target_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = _idx_for_year(baseline.years, target_year)
    sem_names = list(sem_output.inputs.v_names)

    sem_rows: list[dict] = []
    epi_rows: list[dict] = []

    for scenario_name, scenario_out in interventions.items():
        common_units = sorted(set(baseline.results) & set(scenario_out.results))
        for unit_id in common_units:
            base_res = baseline.results[unit_id]
            scen_res = scenario_out.results[unit_id]

            base_sem = np.median(
                np.asarray([s.sem_trajectory for s in base_res.samples], dtype=float),
                axis=0,
            )  # (m, T)
            scen_sem = np.median(
                np.asarray([s.sem_trajectory for s in scen_res.samples], dtype=float),
                axis=0,
            )  # (m, T)

            for i, var in enumerate(sem_names):
                b = float(base_sem[i, idx])
                s = float(scen_sem[i, idx])
                sem_rows.append(
                    {
                        "unit_id": unit_id,
                        "scenario": scenario_name,
                        "year": int(target_year),
                        "variable": var,
                        "baseline_value_median": b,
                        "scenario_value_median": s,
                        "pct_change_median": _pct_change(b, s),
                    }
                )

            for var in EPI_VARS:
                b = float(base_res.get_quantiles(var, q=(0.5,))[0.5][idx])
                s = float(scen_res.get_quantiles(var, q=(0.5,))[0.5][idx])
                epi_rows.append(
                    {
                        "unit_id": unit_id,
                        "scenario": scenario_name,
                        "year": int(target_year),
                        "variable": var,
                        "baseline_value_median": b,
                        "scenario_value_median": s,
                        "pct_change_median": _pct_change(b, s),
                    }
                )

    return pd.DataFrame(sem_rows), pd.DataFrame(epi_rows)


def save_year10_tables(
    sem_df: pd.DataFrame,
    epi_df: pd.DataFrame,
    sem_path: Path,
    epi_path: Path,
) -> None:
    sem_path.parent.mkdir(parents=True, exist_ok=True)
    epi_path.parent.mkdir(parents=True, exist_ok=True)
    sem_df.to_csv(sem_path, index=False)
    epi_df.to_csv(epi_path, index=False)

