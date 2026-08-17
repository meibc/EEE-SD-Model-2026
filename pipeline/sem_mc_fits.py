from __future__ import annotations

"""Fit SEM J/c draws from cached SEM input trajectories.

This is Stage 2 of the SEM uncertainty workflow:

1. `pipeline.sem_mc_inputs` creates probability-space input draws from the
   Factor Analysis workbook.
2. This module consumes those cached draws and refits the centered-logit,
   structural-prior, drift-enabled SEM for each draw.

The output artifact is intentionally separate from the input-draw artifact and
is compatible with `input.standardized_runtime.StandardizedSEMParamsLoader` after Stage 03 export.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from config.optimization import OptimConfig
from config.sem import SEMConfig
from config.shrinkage import ShrinkageConfig
from models.sbm.runner import SBRunner


@dataclass(frozen=True)
class SEMDrawFitConfig:
    input_path: Path = Path("output/sem_input_draws_v5.npz")
    output_path: Path = Path("output/sem_fits_centered_drift_v5.npz")
    data_path: Path = Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx")
    n_draws: int | None = None
    start_draw: int = 0
    save_rolling: bool = False


def _as_str_list(arr) -> list[str]:
    return [str(x) for x in list(arr)]


def _load_input_draws(input_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing SEM input draws: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    required = {"Xbars_samples", "unit_order", "v_names", "ts"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"SEM input draw artifact missing fields: {sorted(missing)}")
    return {k: data[k] for k in data.files}


def _prepared_unit_index(prepared) -> dict[str, int]:
    return {str(unit.id): idx for idx, unit in enumerate(prepared.units)}


def _override_prepared_values(prepared, draw_values: np.ndarray, input_unit_order: list[str]) -> None:
    """Replace prepared unit AMIS values with one probability-space MC draw."""
    unit_idx = _prepared_unit_index(prepared)
    missing = [u for u in input_unit_order if u not in unit_idx]
    if missing:
        raise ValueError(f"Input draw units are absent from prepared data: {missing[:10]}")

    for input_geo_idx, unit_id in enumerate(input_unit_order):
        prepared.units[unit_idx[unit_id]].amis_values = np.asarray(draw_values[:, :, input_geo_idx], dtype=float)


def fit_sem_draws(config: SEMDrawFitConfig) -> Path:
    input_data = _load_input_draws(config.input_path)
    Xbars_samples = np.asarray(input_data["Xbars_samples"], dtype=float)
    input_unit_order = _as_str_list(input_data["unit_order"])
    input_v_names = _as_str_list(input_data["v_names"])
    input_ts = np.asarray(input_data["ts"], dtype=int)

    if Xbars_samples.ndim != 4:
        raise ValueError("Xbars_samples must have shape (draw, variable, year, unit)")

    total_draws, m, T, G = Xbars_samples.shape
    if len(input_v_names) != m or len(input_ts) != T or len(input_unit_order) != G:
        raise ValueError("Input artifact metadata does not match Xbars_samples shape")

    stop_draw = total_draws if config.n_draws is None else min(total_draws, config.start_draw + int(config.n_draws))
    draw_indices = list(range(int(config.start_draw), stop_draw))
    if not draw_indices:
        raise ValueError("No SEM draws selected for fitting")

    sem_config = SEMConfig(data_path=config.data_path)
    runner = SBRunner(sem_config, OptimConfig(), ShrinkageConfig())
    prepared_template = runner.data_prep.prepare_inputs()
    fit_unit_order = [str(unit.id) for unit in prepared_template.units]
    reference_probs = runner.reference_probs

    if input_v_names != list(prepared_template.v_names):
        raise ValueError(f"Variable order mismatch: input={input_v_names}, prepared={prepared_template.v_names}")
    if list(input_ts) != list(np.asarray(prepared_template.ts, dtype=int)):
        raise ValueError(f"Year order mismatch: input={input_ts.tolist()}, prepared={prepared_template.ts.tolist()}")
    if input_unit_order != fit_unit_order:
        raise ValueError("Unit order mismatch between input artifact and prepared data")

    n_fit = len(draw_indices)
    Jmeans_stack = np.full((n_fit, G, m, m), np.nan, dtype=np.float32)
    drift_stack = np.full((n_fit, G, m), np.nan, dtype=np.float32)
    Jrolling_stack = np.full((n_fit, G, m, m, T), np.nan, dtype=np.float32) if config.save_rolling else None
    success_stack = np.zeros((n_fit, G), dtype=bool)
    final_loss_stack = np.full((n_fit, G), np.nan, dtype=np.float64)

    audit_rows: list[dict] = []
    t0 = time.time()

    for out_idx, draw_idx in enumerate(draw_indices):
        prepared = runner.data_prep.prepare_inputs()
        _override_prepared_values(prepared, Xbars_samples[draw_idx], input_unit_order)

        fit_results = runner._fit_all(prepared)
        for geo_idx, unit_id in enumerate(fit_unit_order):
            fit = fit_results.results[unit_id]
            Jrolling = np.asarray(fit.J, dtype=float)
            Jmeans_stack[out_idx, geo_idx] = Jrolling[:, :, -1]
            if Jrolling_stack is not None:
                Jrolling_stack[out_idx, geo_idx] = Jrolling
            drift_stack[out_idx, geo_idx] = np.zeros(m, dtype=float) if fit.drift is None else np.asarray(fit.drift, dtype=float)

            diagnostics = fit.loss_diagnostics or []
            last_diag = diagnostics[-1]["diagnostics"] if diagnostics else {}
            success = bool(last_diag.get("success", False))
            success_stack[out_idx, geo_idx] = success
            final_loss = last_diag.get("final_loss", np.nan)
            final_loss_stack[out_idx, geo_idx] = float(final_loss) if np.isfinite(final_loss) else np.nan
            audit_rows.append({
                "output_draw_index": out_idx,
                "source_draw_index": draw_idx,
                "unit_id": unit_id,
                "success": success,
                "final_loss": final_loss_stack[out_idx, geo_idx],
                "drift_norm": float(np.linalg.norm(drift_stack[out_idx, geo_idx])),
                "J_abs_mean": float(np.nanmean(np.abs(Jmeans_stack[out_idx, geo_idx]))),
            })

        elapsed = time.time() - t0
        print(f"fit draw {draw_idx + 1}/{total_draws} -> saved index {out_idx + 1}/{n_fit} ({elapsed:.1f}s elapsed)")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "Jmeans_stack": Jmeans_stack,
        "drift_stack": drift_stack,
        "unit_order": np.array(fit_unit_order),
        "v_names": np.array(input_v_names),
        "ts": input_ts,
        "reference_probs": np.asarray(reference_probs, dtype=float),
        "source_input_path": str(config.input_path),
        "source_draw_indices": np.asarray(draw_indices, dtype=int),
        "data_path": str(config.data_path),
        "artifact_type": np.array("sem_centered_drift_fits"),
        "success_stack": success_stack,
        "final_loss_stack": final_loss_stack,
    }
    if Jrolling_stack is not None:
        save_kwargs["Jrolling_stack"] = Jrolling_stack
    np.savez_compressed(config.output_path, **save_kwargs)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(config.output_path.with_suffix(".audit.csv"), index=False)

    summary = pd.DataFrame({
        "metric": [
            "n_source_draws",
            "n_fit_draws",
            "n_variables",
            "n_years",
            "n_units",
            "success_rate",
            "mean_final_loss",
            "elapsed_seconds",
        ],
        "value": [
            total_draws,
            n_fit,
            m,
            T,
            G,
            float(success_stack.mean()),
            float(np.nanmean(final_loss_stack)),
            float(time.time() - t0),
        ],
    })
    summary.to_csv(config.output_path.with_suffix(".summary.csv"), index=False)
    return config.output_path
