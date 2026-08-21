from __future__ import annotations

"""Run scenario uncertainty from standardized v8 SEM/CDC draws.

This stage consumes the dashboard-ready standardized v8 artifact, applies the
finalized scenario codebook, and propagates outcomes through the CDC/hierarchical
model. Stage 02 SEM refitting is optional once standardized v8 has been exported.
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
    standardized_input_path: Path = Path("output/standardized_input_v8.npz")
    output_dir: Path = Path("output/sem_scenarios_centered_drift_v5")
    scenario_codes: list[str] = field(default_factory=lambda: list(SCENARIO_CODEBOOK.keys()))
    n_samples: int = 500
    seed: int = 123
    target_end_year: int = 2036
    intervention_duration_steps: int = 3
    show_progress: bool = False
    save_outputs_pickle: bool = False


EPI_VARS = ("incidence", "diagnosed", "undiagnosed", "prep_on_count")
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
) -> tuple[list[dict], list[dict]]:
    cdc_idx = _idx_for_year(baseline.years, target_year)
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
            b = _sample_epi_stack(baseline, unit_id, var)[:n, cdc_idx]
            s = _sample_epi_stack(scenario, unit_id, var)[:n, cdc_idx]
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


def _scenario_aggregate_rows(endpoint_epi: pd.DataFrame, endpoint_sem: pd.DataFrame) -> pd.DataFrame:
    """USA-focused compact table for quick review."""
    rows = []
    for df, domain in [(endpoint_epi, "epi"), (endpoint_sem, "sem")]:
        usa = df[df["unit_id"] == "USA"].copy()
        for _, row in usa.iterrows():
            rows.append({
                "domain": domain,
                "scenario": row["scenario"],
                "scenario_label": row["scenario_label"],
                "year": int(row["year"]),
                "variable": row["variable"],
                "baseline_p50": row["baseline_p50"],
                "scenario_p50": row["scenario_p50"],
                "delta_p50": row["delta_p50"],
                "pct_change_p50": row["pct_change_p50"],
                "delta_p025": row["delta_p025"],
                "delta_p975": row["delta_p975"],
            })
    return pd.DataFrame(rows)


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
    )

    if config.save_outputs_pickle:
        with open(config.output_dir / "baseline_uncertainty.pkl", "wb") as f:
            pickle.dump(baseline, f)

    sem_rows: list[dict] = []
    epi_rows: list[dict] = []
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
        )
        s_rows, e_rows = _summarize_scenario(
            baseline=baseline,
            scenario=scenario_output,
            scenario_code=scenario_code,
            scenario_label=spec.get("label", scenario_code),
            sem_years=sem_years,
            target_year=config.target_end_year,
        )
        sem_rows.extend(s_rows)
        epi_rows.extend(e_rows)
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
    aggregate = _scenario_aggregate_rows(endpoint_epi, endpoint_sem)
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
        "usa_summary": config.output_dir / "scenario_usa_summary.csv",
        "audit": config.output_dir / "scenario_audit.csv",
        "summary": config.output_dir / "scenario_run_summary.csv",
    }
    endpoint_sem.to_csv(paths["endpoint_sem"], index=False)
    endpoint_epi.to_csv(paths["endpoint_epi"], index=False)
    aggregate.to_csv(paths["usa_summary"], index=False)
    audit.to_csv(paths["audit"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths
