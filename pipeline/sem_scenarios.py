from __future__ import annotations

"""Scenario projections from the research bundle (input/research_bundle.py).

Runs the baseline and each scenario in the codebook (config/interventions.py)
through pipeline/joint_simulation.py and summarizes paired baseline-scenario
contrasts at the report years, for the eight modeled measures and the
epidemiological outcomes (incidence, diagnoses, undiagnosed, PrEP users, and
cumulative incidence from ``cumulative_start_year``).
"""

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from config.interventions import SCENARIO_CODEBOOK
from input.standardized_runtime import StandardizedBundle, StandardizedCDCParamsLoader, StandardizedSEMParamsLoader
from models.shared.alignment import build_model_years
from pipeline.joint_simulation import run_uncertainty
from pipeline.results import UncertaintyOutput


@dataclass(frozen=True)
class SEMScenarioRunConfig:
    standardized_input_path: Path = Path("results/03_fits/primary/standardized_research.npz")
    output_dir: Path = Path("results/04_scenarios/primary")
    scenario_codes: list[str] = field(default_factory=lambda: list(SCENARIO_CODEBOOK.keys()))
    n_samples: int = 500
    seed: int = 123
    target_end_year: int = 2036
    intervention_duration_steps: int = 3
    projection_start: str = "observed"
    report_years: tuple[int, ...] = ()
    cumulative_start_year: int = 2025
    show_progress: bool = False
    save_outputs_pickle: bool = False
    save_draw_level_endpoints: bool = False
    contextual_prep_pr: float | None = None   # contextual upper-bound sensitivity (pipeline/joint_simulation.py)
    constant_pr_prep: bool = False            # static evidence-translation sensitivity (pipeline/joint_simulation.py)
    zero_projection_edges: tuple[tuple[str, str], ...] = ()   # (source, target) pathways off in projections


EPI_VARS = ("incidence", "diagnosed", "undiagnosed", "prep_on_count", "incidence_cumulative")
SEM_VARS_OF_INTEREST = (
    "stigma_ahs",
    "stigma_gss",
    "stigma_family",
    "out_gid",
    "seehcp",
    "risk_behavior",
    "prep_used",
    "hivtest12",
)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "p025": float(np.nanquantile(arr, 0.025)),
        "p50": float(np.nanquantile(arr, 0.5)),
        "p975": float(np.nanquantile(arr, 0.975)),
    }


def _pct_change(scenario: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    scenario = np.asarray(scenario, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    return np.where(np.abs(baseline) > 1e-12, 100.0 * (scenario - baseline) / baseline, np.nan)


def _sample_sem_stack(output: UncertaintyOutput, unit_id: str) -> np.ndarray:
    return np.asarray([s.sem_trajectory for s in output.results[unit_id].samples], dtype=float)


def _sample_epi_stack(output: UncertaintyOutput, unit_id: str, var: str) -> np.ndarray:
    return np.asarray([getattr(s.cdc_output, var) for s in output.results[unit_id].samples], dtype=float)


def _epi_endpoint(output: UncertaintyOutput, unit_id: str, var: str, cdc_idx: int, cumulative_start_idx: int) -> np.ndarray:
    """Per-draw value at the endpoint; ``incidence_cumulative`` sums annual
    incidence from the cumulative start year through the endpoint."""
    if var == "incidence_cumulative":
        stack = _sample_epi_stack(output, unit_id, "incidence")
        return stack[:, cumulative_start_idx:cdc_idx + 1].sum(axis=1)
    return _sample_epi_stack(output, unit_id, var)[:, cdc_idx]


def _idx_for_year(years: np.ndarray, target_year: int) -> int:
    years = np.asarray(years, dtype=int).ravel()
    idx = np.where(years == int(target_year))[0]
    if idx.size == 0:
        raise ValueError(f"target_year={target_year} absent from years={years.tolist()}")
    return int(idx[0])


def _summarize_scenario(
    baseline: UncertaintyOutput,
    scenario: UncertaintyOutput,
    scenario_code: str,
    scenario_label: str,
    sem_years: np.ndarray,
    target_year: int,
    cumulative_start_year: int = 2025,
) -> tuple[list[dict], list[dict]]:
    cdc_idx = _idx_for_year(baseline.years, target_year)
    cum_idx = _idx_for_year(baseline.years, cumulative_start_year)
    sem_idx = _idx_for_year(sem_years, target_year)
    sem_names = list(baseline.v_names)
    common_units = sorted(set(baseline.results) & set(scenario.results))

    sem_rows: list[dict] = []
    epi_rows: list[dict] = []

    for unit_id in common_units:
        base_sem = _sample_sem_stack(baseline, unit_id)
        scen_sem = _sample_sem_stack(scenario, unit_id)
        n = min(base_sem.shape[0], scen_sem.shape[0])
        base_sem = base_sem[:n]
        scen_sem = scen_sem[:n]

        for var in SEM_VARS_OF_INTEREST:
            if var not in sem_names:
                continue
            k = sem_names.index(var)
            b = base_sem[:, k, sem_idx]
            s = scen_sem[:, k, sem_idx]
            delta = s - b
            pct = _pct_change(s, b)
            row = {
                "unit_id": unit_id,
                "scenario": scenario_code,
                "scenario_label": scenario_label,
                "year": int(target_year),
                "variable": var,
                **{f"baseline_{kk}": vv for kk, vv in _quantiles(b).items()},
                **{f"scenario_{kk}": vv for kk, vv in _quantiles(s).items()},
                **{f"delta_{kk}": vv for kk, vv in _quantiles(delta).items()},
                **{f"pct_change_{kk}": vv for kk, vv in _quantiles(pct).items()},
            }
            sem_rows.append(row)

        for var in EPI_VARS:
            b = _epi_endpoint(baseline, unit_id, var, cdc_idx, cum_idx)[:n]
            s = _epi_endpoint(scenario, unit_id, var, cdc_idx, cum_idx)[:n]
            delta = s - b
            pct = _pct_change(s, b)
            row = {
                "unit_id": unit_id,
                "scenario": scenario_code,
                "scenario_label": scenario_label,
                "year": int(target_year),
                "variable": var,
                **{f"baseline_{kk}": vv for kk, vv in _quantiles(b).items()},
                **{f"scenario_{kk}": vv for kk, vv in _quantiles(s).items()},
                **{f"delta_{kk}": vv for kk, vv in _quantiles(delta).items()},
                **{f"pct_change_{kk}": vv for kk, vv in _quantiles(pct).items()},
            }
            epi_rows.append(row)

    return sem_rows, epi_rows


def _draw_level_endpoint_rows(
    baseline: UncertaintyOutput,
    scenario: UncertaintyOutput,
    scenario_code: str,
    scenario_label: str,
    sem_years: np.ndarray,
    target_year: int,
    cumulative_start_year: int = 2025,
) -> tuple[list[dict], list[dict]]:
    """Return paired draw-level endpoint rows for baseline/scenario contrasts."""
    cdc_idx = _idx_for_year(baseline.years, target_year)
    cum_idx = _idx_for_year(baseline.years, cumulative_start_year)
    sem_idx = _idx_for_year(sem_years, target_year)
    sem_names = list(baseline.v_names)
    common_units = sorted(set(baseline.results) & set(scenario.results))

    sem_rows: list[dict] = []
    epi_rows: list[dict] = []

    for unit_id in common_units:
        base_sem = _sample_sem_stack(baseline, unit_id)
        scen_sem = _sample_sem_stack(scenario, unit_id)
        n = min(base_sem.shape[0], scen_sem.shape[0])
        base_sem = base_sem[:n]
        scen_sem = scen_sem[:n]

        epi_base = {var: _epi_endpoint(baseline, unit_id, var, cdc_idx, cum_idx) for var in EPI_VARS}
        epi_scen = {var: _epi_endpoint(scenario, unit_id, var, cdc_idx, cum_idx) for var in EPI_VARS}
        for draw in range(n):
            for var in SEM_VARS_OF_INTEREST:
                if var not in sem_names:
                    continue
                k = sem_names.index(var)
                b = float(base_sem[draw, k, sem_idx])
                s = float(scen_sem[draw, k, sem_idx])
                sem_rows.append({
                    "unit_id": unit_id,
                    "scenario": scenario_code,
                    "scenario_label": scenario_label,
                    "year": int(target_year),
                    "draw": int(draw),
                    "variable": var,
                    "baseline": b,
                    "scenario_value": s,
                    "delta": s - b,
                    "pct_change": float(_pct_change(np.asarray([s]), np.asarray([b]))[0]),
                })

            for var in EPI_VARS:
                b = float(epi_base[var][draw])
                s = float(epi_scen[var][draw])
                epi_rows.append({
                    "unit_id": unit_id,
                    "scenario": scenario_code,
                    "scenario_label": scenario_label,
                    "year": int(target_year),
                    "draw": int(draw),
                    "variable": var,
                    "baseline": b,
                    "scenario_value": s,
                    "delta": s - b,
                    "pct_change": float(_pct_change(np.asarray([s]), np.asarray([b]))[0]),
                })

    return sem_rows, epi_rows


def run_sem_scenarios(config: SEMScenarioRunConfig) -> dict[str, Path]:
    t0 = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = StandardizedBundle(config.standardized_input_path)
    sem_loader = StandardizedSEMParamsLoader(bundle)
    cdc_loader = StandardizedCDCParamsLoader(bundle)
    units = bundle.build_units()
    unit_ids = sorted(set(units.keys()) & set(cdc_loader.geo_names) & set(sem_loader.geo_names))
    model_years = build_model_years(cdc_loader.years, target_end_year=config.target_end_year)
    sem_years = np.asarray(sem_loader.ts, dtype=int)
    if sem_years[-1] < config.target_end_year:
        # SEM uses 2-year steps by construction.
        extra = np.arange(sem_years[-1] + 2, config.target_end_year + 1, 2, dtype=int)
        sem_years = np.concatenate([sem_years, extra])

    invalid = [s for s in config.scenario_codes if s not in SCENARIO_CODEBOOK]
    if invalid:
        raise ValueError(f"Unknown scenario codes: {invalid}")

    baseline = run_uncertainty(
        sem_loader,
        cdc_loader,
        units,
        unit_ids=unit_ids,
        n_samples=config.n_samples,
        seed=config.seed,
        show_progress=config.show_progress,
        model_years=model_years,
        state_intervention_codes=[],
        relationship_intervention_codes=[],
        intervention_duration_steps=config.intervention_duration_steps,
        projection_start=config.projection_start,
        zero_projection_edges=config.zero_projection_edges,
    )

    if config.save_outputs_pickle:
        with open(config.output_dir / "baseline_uncertainty.pkl", "wb") as f:
            pickle.dump(baseline, f)

    sem_rows: list[dict] = []
    epi_rows: list[dict] = []
    sem_draw_rows: list[dict] = []
    epi_draw_rows: list[dict] = []
    scenario_audit: list[dict] = []

    for scenario_code in config.scenario_codes:
        spec = SCENARIO_CODEBOOK[scenario_code]
        print(f"running scenario: {scenario_code}")
        scenario_output = run_uncertainty(
            sem_loader,
            cdc_loader,
            units,
            unit_ids=unit_ids,
            n_samples=config.n_samples,
            seed=config.seed,
            show_progress=config.show_progress,
            model_years=model_years,
            state_intervention_codes=spec.get("state_codes", []),
            relationship_intervention_codes=spec.get("relationship_codes", []),
            intervention_duration_steps=config.intervention_duration_steps,
            projection_start=config.projection_start,
            contextual_prep_pr=config.contextual_prep_pr,
            constant_pr_prep=config.constant_pr_prep,
            prep_only_fraction=spec.get("prep_only_fraction"),
            zero_projection_edges=config.zero_projection_edges,
        )
        for report_year in config.report_years or (config.target_end_year,):
            s_rows, e_rows = _summarize_scenario(
                baseline=baseline,
                scenario=scenario_output,
                scenario_code=scenario_code,
                scenario_label=spec.get("label", scenario_code),
                sem_years=sem_years,
                target_year=report_year,
                cumulative_start_year=config.cumulative_start_year,
            )
            sem_rows.extend(s_rows)
            epi_rows.extend(e_rows)
            if config.save_draw_level_endpoints:
                sd_rows, ed_rows = _draw_level_endpoint_rows(
                    baseline=baseline,
                    scenario=scenario_output,
                    scenario_code=scenario_code,
                    scenario_label=spec.get("label", scenario_code),
                    sem_years=sem_years,
                    target_year=report_year,
                    cumulative_start_year=config.cumulative_start_year,
                )
                sem_draw_rows.extend(sd_rows)
                epi_draw_rows.extend(ed_rows)
        scenario_audit.append({
            "scenario": scenario_code,
            "label": spec.get("label", scenario_code),
            "state_codes": ",".join(spec.get("state_codes", [])),
            "relationship_codes": ",".join(spec.get("relationship_codes", [])),
            "n_units": len(scenario_output.results),
            "n_samples": config.n_samples,
        })

        if config.save_outputs_pickle:
            with open(config.output_dir / f"{scenario_code}_uncertainty.pkl", "wb") as f:
                pickle.dump(scenario_output, f)

    endpoint_sem = pd.DataFrame(sem_rows)
    endpoint_epi = pd.DataFrame(epi_rows)
    draw_endpoint_sem = pd.DataFrame(sem_draw_rows)
    draw_endpoint_epi = pd.DataFrame(epi_draw_rows)
    audit = pd.DataFrame(scenario_audit)
    summary = pd.DataFrame({
        "metric": [
            "n_scenarios",
            "n_units",
            "n_samples",
            "target_end_year",
            "intervention_duration_steps",
            "elapsed_seconds",
        ],
        "value": [
            len(config.scenario_codes),
            len(unit_ids),
            config.n_samples,
            config.target_end_year,
            config.intervention_duration_steps,
            time.time() - t0,
        ],
    })

    paths = {
        "endpoint_sem": config.output_dir / "scenario_endpoint_sem.csv",
        "endpoint_epi": config.output_dir / "scenario_endpoint_epi.csv",
        "draw_endpoint_sem": config.output_dir / "scenario_endpoint_sem_draws.csv",
        "draw_endpoint_epi": config.output_dir / "scenario_endpoint_epi_draws.csv",
        "audit": config.output_dir / "scenario_audit.csv",
        "summary": config.output_dir / "scenario_run_summary.csv",
    }
    endpoint_sem.to_csv(paths["endpoint_sem"], index=False)
    endpoint_epi.to_csv(paths["endpoint_epi"], index=False)
    if config.save_draw_level_endpoints:
        draw_endpoint_sem.to_csv(paths["draw_endpoint_sem"], index=False)
        draw_endpoint_epi.to_csv(paths["draw_endpoint_epi"], index=False)
    else:
        paths.pop("draw_endpoint_sem")
        paths.pop("draw_endpoint_epi")
    audit.to_csv(paths["audit"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths
