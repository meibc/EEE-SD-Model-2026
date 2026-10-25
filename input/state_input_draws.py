"""Model-input draws from the respondent-bootstrap estimates (scripts/03).

``build_state_input_draws`` arranges the completed bootstrap estimates (state,
division and national) as a draws x measures x periods x units array in the unit,
measure and period order used by the model (data.data_prep), checks completeness
and range, clips to [eps, 1 - eps], and writes the ``Xbars_samples`` NPZ read by
pipeline/state_fit.py. Statistical estimation and division fallback are completed
upstream in scripts/03_state_bootstrap.R.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.sem import SEMConfig
from data.data_prep import DataPrep

BOOTSTRAP_COLUMNS = ["draw", "unit", "year", "variable", "model_draw"]


def model_layout(workbook_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Return the unit, measure, and period order used by the model."""
    data_prep = DataPrep(SEMConfig(data_path=workbook_path))
    prepared = data_prep.prepare_inputs()
    return (
        [str(u.id) for u in prepared.units],
        [str(v) for v in prepared.v_names],
        np.asarray(prepared.ts, dtype=int),
    )


def load_bootstrap(path: Path, expected_draws: int) -> tuple[pd.DataFrame, list[int]]:
    data = pd.read_csv(path, usecols=BOOTSTRAP_COLUMNS,
                       dtype={"draw": "int32", "unit": "string", "year": "int16", "variable": "string",
                              "model_draw": "float64"})
    if data.isna().any().any():
        raise ValueError(f"Bootstrap draws contain missing values:\n{data.isna().sum()}")
    draw_ids = sorted(data["draw"].astype(int).unique().tolist())
    if len(draw_ids) != expected_draws:
        raise ValueError(f"Expected {expected_draws} draws, found {len(draw_ids)}")
    return data, draw_ids


def dense_array(data: pd.DataFrame, draw_ids: list[int], units: list[str], v_names: list[str],
                years: np.ndarray) -> np.ndarray:
    """draws x measures x periods x units array; NaN where the bootstrap has no estimate."""
    maps = {"draw": {d: i for i, d in enumerate(draw_ids)}, "unit": {u: i for i, u in enumerate(units)},
            "variable": {v: i for i, v in enumerate(v_names)}, "year": {int(y): i for i, y in enumerate(years)}}
    idx = {}
    for col, mapping in maps.items():
        idx[col] = data[col].astype(int if col in ("draw", "year") else str).map(mapping)
        if idx[col].isna().any():
            raise ValueError(f"Bootstrap {col} values not in the model layout: "
                             f"{sorted(data.loc[idx[col].isna(), col].astype(str).unique())[:10]}")
    keys = pd.DataFrame(idx)
    if keys.duplicated().any():
        raise ValueError("Duplicate bootstrap cells")
    values = np.full((len(draw_ids), len(v_names), len(years), len(units)), np.nan, dtype=np.float32)
    values[keys["draw"].to_numpy(int), keys["variable"].to_numpy(int), keys["year"].to_numpy(int),
           keys["unit"].to_numpy(int)] = data["model_draw"].to_numpy(dtype=np.float32)
    return values


def validate_complete(values: np.ndarray) -> None:
    """Require a complete probability-scale array from the upstream R workflow."""
    if not np.isfinite(values).all():
        missing = np.argwhere(~np.isfinite(values))[:10]
        raise ValueError(
            "Model inputs are incomplete after R preparation; "
            f"first missing array positions: {missing.tolist()}"
        )
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Model inputs fall outside [0, 1]")


def build_state_input_draws(bootstrap_path: Path, workbook_path: Path, output_path: Path,
                            eps: float = 1e-6, expected_draws: int = 500) -> dict:
    """Write the model-input NPZ and its audits; return the audit metrics."""
    units, v_names, years = model_layout(workbook_path)
    data, draw_ids = load_bootstrap(bootstrap_path, expected_draws)
    values = dense_array(data, draw_ids, units, v_names, years)

    validate_complete(values)
    values = np.clip(values, eps, 1 - eps).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, Xbars_samples=values, unit_order=np.asarray(units), v_names=np.asarray(v_names), ts=years,
        source_draw_ids=np.asarray(draw_ids, dtype=int), source_bootstrap_path=np.asarray(str(bootstrap_path)),
        data_path=np.asarray(str(workbook_path)),
    )
    metrics = {"n_draws": values.shape[0], "n_variables": values.shape[1], "n_years": values.shape[2],
               "n_units": values.shape[3],
               "n_boundary": int(((values <= eps) | (values >= 1 - eps)).sum())}
    pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())}).to_csv(
        output_path.with_suffix(".audit.csv"), index=False)
    return metrics
