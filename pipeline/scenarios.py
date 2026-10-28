from __future__ import annotations

"""Scenario projections from the research bundle (artifacts/research_bundle.py).

Runs the baseline and each scenario in the codebook (config/interventions.py)
through pipeline/projection.py and summarizes paired baseline-scenario
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
from config.projection import (CUMULATIVE_START_YEAR, INTERVENTION_DURATION_STEPS, N_SAMPLES, SEED,
                               TARGET_END_YEAR)
from artifacts.standardized_runtime import StandardizedBundle, StandardizedEpiLoader, StandardizedDynamicLoader
from models.shared.alignment import build_model_years, extend_to_end_year
from pipeline.projection import run_projection
from pipeline.projection_results import ProjectionOutput, ProjectionResult


@dataclass(frozen=True)
class ScenarioRunConfig:
    standardized_input_path: Path = Path("results/03_fits/primary/standardized_research.npz")
    output_dir: Path = Path("results/04_scenarios/paper_static")
    scenario_codes: list[str] = field(default_factory=lambda: list(SCENARIO_CODEBOOK.keys()))
    n_samples: int = N_SAMPLES
    seed: int = SEED
    target_end_year: int = TARGET_END_YEAR
    intervention_duration_steps: int = INTERVENTION_DURATION_STEPS
    report_years: tuple[int, ...] = ()
    cumulative_start_year: int = CUMULATIVE_START_YEAR
    show_progress: bool = False
    save_outputs_pickle: bool = False
    save_draw_level_endpoints: bool = False
    contextual_prep_pr: float | None = None   # contextual upper-bound sensitivity (pipeline/projection.py)
    constant_pr_prep: bool = False            # primary PrEP translation from the prevalence ratio (pipeline/projection.py)
    zero_projection_edges: tuple[tuple[str, str], ...] = ()   # (source, target) pathways off in projections


EPI_VARS = ("incidence", "diagnosed", "undiagnosed", "prep_on_count", "incidence_cumulative")


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


def _epi_endpoint(result: ProjectionResult, var: str, epi_idx: int, cumulative_start_idx: int) -> np.ndarray:
    """Per-draw value at the endpoint; ``incidence_cumulative`` sums annual
    incidence from the cumulative start year through the endpoint."""
    if var == "incidence_cumulative":
        return result.epi_stack("incidence")[:, cumulative_start_idx:epi_idx + 1].sum(axis=1)
    return result.epi_stack(var)[:, epi_idx]


def _idx_for_year(years: np.ndarray, target_year: int) -> int:
    years = np.asarray(years, dtype=int).ravel()
    idx = np.where(years == int(target_year))[0]
    if idx.size == 0:
        raise ValueError(f"target_year={target_year} absent from years={years.tolist()}")
    return int(idx[0])


def _paired_endpoints(baseline: ProjectionOutput, scenario: ProjectionOutput, dynamic_years: np.ndarray,
                      target_year: int, cumulative_start_year: int) -> dict[str, tuple[dict, dict]]:
    """Paired draws at ``target_year`` by state: ({measure: (baseline, scenario)}, {epi output: (baseline,
    scenario)}). Baseline and scenario use the same seed and sample count, so draw k is the same pair of
    dynamic-model and epidemic draws in both (pipeline/projection.py)."""
    epi_idx = _idx_for_year(baseline.years, target_year)
    cum_idx = _idx_for_year(baseline.years, cumulative_start_year)
    dynamic_idx = _idx_for_year(dynamic_years, target_year)
    out = {}
    for unit_id in sorted(set(baseline.results) & set(scenario.results)):
        b_res, s_res = baseline.results[unit_id], scenario.results[unit_id]
        base_dynamic, scen_dynamic = b_res.dynamic_stack(), s_res.dynamic_stack()
        dynamic = {var: (base_dynamic[:, k, dynamic_idx], scen_dynamic[:, k, dynamic_idx]) for k, var in enumerate(baseline.v_names)}
        epi = {var: (_epi_endpoint(b_res, var, epi_idx, cum_idx), _epi_endpoint(s_res, var, epi_idx, cum_idx))
               for var in EPI_VARS}
        out[unit_id] = (dynamic, epi)
    return out


def _summary_rows(endpoints: dict, labels: dict) -> tuple[list[dict], list[dict]]:
    """Medians and 95% intervals of baseline, scenario and their paired difference and percent change."""
    dynamic_rows, epi_rows = [], []
    for unit_id, (dynamic, epi) in endpoints.items():
        for rows, values in ((dynamic_rows, dynamic), (epi_rows, epi)):
            for var, (b, s) in values.items():
                rows.append({
                    "unit_id": unit_id, **labels, "variable": var,
                    **{f"baseline_{kk}": vv for kk, vv in _quantiles(b).items()},
                    **{f"scenario_{kk}": vv for kk, vv in _quantiles(s).items()},
                    **{f"delta_{kk}": vv for kk, vv in _quantiles(s - b).items()},
                    **{f"pct_change_{kk}": vv for kk, vv in _quantiles(_pct_change(s, b)).items()},
                })
    return dynamic_rows, epi_rows


def _draw_rows(endpoints: dict, labels: dict) -> tuple[list[dict], list[dict]]:
    """The same paired values, one row per draw."""
    dynamic_rows, epi_rows = [], []
    for unit_id, (dynamic, epi) in endpoints.items():
        for rows, values in ((dynamic_rows, dynamic), (epi_rows, epi)):
            pct = {var: _pct_change(s, b) for var, (b, s) in values.items()}
            for draw in range(len(next(iter(values.values()))[0])):
                for var, (b, s) in values.items():
                    rows.append({
                        "unit_id": unit_id, **labels, "draw": int(draw), "variable": var,
                        "baseline": float(b[draw]), "scenario_value": float(s[draw]),
                        "delta": float(s[draw]) - float(b[draw]), "pct_change": float(pct[var][draw]),
                    })
    return dynamic_rows, epi_rows


def run_scenarios(config: ScenarioRunConfig) -> dict[str, Path]:
    t0 = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = StandardizedBundle(config.standardized_input_path)
    dynamic_loader = StandardizedDynamicLoader(bundle)
    epi_loader = StandardizedEpiLoader(bundle)
    units = bundle.build_units()
    unit_ids = sorted(set(units.keys()) & set(epi_loader.geo_names) & set(dynamic_loader.geo_names))
    model_years = build_model_years(epi_loader.years, target_end_year=config.target_end_year)
    dynamic_years = extend_to_end_year(dynamic_loader.ts, target_end_year=config.target_end_year, step=2)

    invalid = [s for s in config.scenario_codes if s not in SCENARIO_CODEBOOK]
    if invalid:
        raise ValueError(f"Unknown scenario codes: {invalid}")

    baseline = run_projection(
        dynamic_loader,
        epi_loader,
        units,
        unit_ids=unit_ids,
        n_samples=config.n_samples,
        seed=config.seed,
        show_progress=config.show_progress,
        model_years=model_years,
        intervention_codes=[],
        intervention_duration_steps=config.intervention_duration_steps,
        zero_projection_edges=config.zero_projection_edges,
    )

    if config.save_outputs_pickle:
        with open(config.output_dir / "baseline_projection.pkl", "wb") as f:
            pickle.dump(baseline, f)

    dynamic_rows: list[dict] = []
    epi_rows: list[dict] = []
    dynamic_draw_rows: list[dict] = []
    epi_draw_rows: list[dict] = []
    scenario_audit: list[dict] = []

    for scenario_code in config.scenario_codes:
        spec = SCENARIO_CODEBOOK[scenario_code]
        print(f"running scenario: {scenario_code}")
        scenario_output = run_projection(
            dynamic_loader,
            epi_loader,
            units,
            unit_ids=unit_ids,
            n_samples=config.n_samples,
            seed=config.seed,
            show_progress=config.show_progress,
            model_years=model_years,
            intervention_codes=spec.get("intervention_codes", []),
            intervention_duration_steps=config.intervention_duration_steps,
            contextual_prep_pr=config.contextual_prep_pr,
            constant_pr_prep=config.constant_pr_prep,
            prep_only_fraction=spec.get("prep_only_fraction"),
            zero_projection_edges=config.zero_projection_edges,
        )
        for report_year in config.report_years or (config.target_end_year,):
            endpoints = _paired_endpoints(baseline, scenario_output, dynamic_years, report_year,
                                          config.cumulative_start_year)
            labels = {"scenario": scenario_code, "scenario_label": spec.get("label", scenario_code),
                      "year": int(report_year)}
            s_rows, e_rows = _summary_rows(endpoints, labels)
            dynamic_rows.extend(s_rows)
            epi_rows.extend(e_rows)
            if config.save_draw_level_endpoints:
                sd_rows, ed_rows = _draw_rows(endpoints, labels)
                dynamic_draw_rows.extend(sd_rows)
                epi_draw_rows.extend(ed_rows)
        scenario_audit.append({
            "scenario": scenario_code,
            "label": spec.get("label", scenario_code),
            "intervention_codes": ",".join(spec.get("intervention_codes", [])),
            "prep_only_fraction": spec.get("prep_only_fraction"),
            "n_units": len(scenario_output.results),
            "n_samples": config.n_samples,
        })

        if config.save_outputs_pickle:
            with open(config.output_dir / f"{scenario_code}_projection.pkl", "wb") as f:
                pickle.dump(scenario_output, f)

    endpoint_dynamic = pd.DataFrame(dynamic_rows)
    endpoint_epi = pd.DataFrame(epi_rows)
    draw_endpoint_dynamic = pd.DataFrame(dynamic_draw_rows)
    draw_endpoint_epi = pd.DataFrame(epi_draw_rows)
    audit = pd.DataFrame(scenario_audit)
    summary = pd.DataFrame({
        "metric": [
            "n_scenarios",
            "n_units",
            "n_samples",
            "target_end_year",
            "intervention_duration_steps",
            "seed",
            "report_years",
            "cumulative_start_year",
            "constant_pr_prep",
            "contextual_prep_pr",
            "zero_projection_edges",
            "elapsed_seconds",
        ],
        "value": [
            len(config.scenario_codes),
            len(unit_ids),
            config.n_samples,
            config.target_end_year,
            config.intervention_duration_steps,
            config.seed,
            ",".join(map(str, config.report_years or (config.target_end_year,))),
            config.cumulative_start_year,
            config.constant_pr_prep,
            config.contextual_prep_pr,
            ",".join(f"{a}:{b}" for a, b in config.zero_projection_edges),
            time.time() - t0,
        ],
    })

    paths = {
        "endpoint_dynamic": config.output_dir / "scenario_endpoint_dynamic.csv",
        "endpoint_epi": config.output_dir / "scenario_endpoint_epi.csv",
        "draw_endpoint_dynamic": config.output_dir / "scenario_endpoint_dynamic_draws.csv.gz",
        "draw_endpoint_epi": config.output_dir / "scenario_endpoint_epi_draws.csv.gz",
        "audit": config.output_dir / "scenario_audit.csv",
        "summary": config.output_dir / "scenario_run_summary.csv",
    }
    endpoint_dynamic.to_csv(paths["endpoint_dynamic"], index=False)
    endpoint_epi.to_csv(paths["endpoint_epi"], index=False)
    if config.save_draw_level_endpoints:
        draw_endpoint_dynamic.to_csv(paths["draw_endpoint_dynamic"], index=False)
        draw_endpoint_epi.to_csv(paths["draw_endpoint_epi"], index=False)
    else:
        paths.pop("draw_endpoint_dynamic")
        paths.pop("draw_endpoint_epi")
    audit.to_csv(paths["audit"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths
