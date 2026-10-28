"""The model-input file: every survey value the later steps use, read and arranged once.

``build_state_inputs`` (scripts/3_model/01) reads the R outputs, the respondent-bootstrap estimates
(scripts/1_data/04) and the model-input workbook (scripts/1_data/03), and writes one NPZ holding, in the model's unit,
measure and period order:

    observed_draws    draw x unit x measure x period   bootstrap estimates, clipped to [eps, 1 - eps]
                                                       (PROB_EPS: only keeps logits finite; the fit clips
                                                       to LOGIT_EPS)
    observed_point    unit x measure x period          workbook point estimates (NaN where none)
    unit_ids, unit_kinds, census_divisions, sample_sizes
    covariance        measure x measure                pooled covariance (workbook "cov" sheet)
    v_names, years, source_draw_ids, and the source paths and workbook SHA-256

The units are the nation, then each Census division followed by its states. The state fits
(pipeline/dynamic_fit.py), the epidemic fit (pipeline/epi_fit.py) and the bundle
(artifacts/research_bundle.py) read only this file, through ``StateInputs.load``. Statistical estimation and
division fallback are completed upstream in scripts/1_data/04_state_bootstrap.R.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config.dynamic import DynamicConfig
from core.math.transforms import PROB_EPS
from data.workbook import ModelInputWorkbook
from data.unit import Unit

BOOTSTRAP_COLUMNS = ["draw", "unit", "year", "variable", "model_draw"]


@dataclass
class StateInputs:
    """The model-input file (see the module docstring); ``units`` rebuilds the ``Unit`` objects."""
    observed_draws: np.ndarray
    observed_point: np.ndarray
    unit_ids: list[str]
    unit_kinds: list[str]
    census_divisions: np.ndarray        # 0 for the nation
    sample_sizes: np.ndarray            # NaN except for states
    covariance: np.ndarray
    v_names: list[str]
    years: np.ndarray
    source_draw_ids: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "StateInputs":
        with np.load(path, allow_pickle=True) as d:
            return cls(
                observed_draws=np.asarray(d["observed_draws"]), observed_point=np.asarray(d["observed_point"]),
                unit_ids=d["unit_ids"].astype(str).tolist(), unit_kinds=d["unit_kinds"].astype(str).tolist(),
                census_divisions=np.asarray(d["census_divisions"], dtype=int),
                sample_sizes=np.asarray(d["sample_sizes"], dtype=float), covariance=np.asarray(d["covariance"]),
                v_names=d["v_names"].astype(str).tolist(), years=np.asarray(d["years"], dtype=int),
                source_draw_ids=np.asarray(d["source_draw_ids"], dtype=int),
            )

    def save(self, path: Path, **provenance) -> None:
        """Write the file; ``provenance`` (e.g. source paths) is stored alongside as strings."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, observed_draws=self.observed_draws, observed_point=self.observed_point,
            unit_ids=np.asarray(self.unit_ids), unit_kinds=np.asarray(self.unit_kinds),
            census_divisions=self.census_divisions, sample_sizes=self.sample_sizes, covariance=self.covariance,
            v_names=np.asarray(self.v_names), years=self.years, source_draw_ids=self.source_draw_ids,
            **{k: np.asarray(str(v)) for k, v in provenance.items()},
        )

    @property
    def units(self) -> list[Unit]:
        return [Unit(id=uid, kind=kind, amis_years=self.years, amis_values=self.observed_point[k],
                     sample_size=None if np.isnan(self.sample_sizes[k]) else float(self.sample_sizes[k]),
                     census_div=None if kind == "nation" else int(self.census_divisions[k]))
                for k, (uid, kind) in enumerate(zip(self.unit_ids, self.unit_kinds))]

    def unit_index(self, unit_ids) -> list[int]:
        """Positions of the given units in the file's order."""
        missing = [u for u in unit_ids if u not in self.unit_ids]
        if missing:
            raise ValueError(f"Units not in the model inputs: {missing}")
        return [self.unit_ids.index(u) for u in unit_ids]


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
    """draw x unit x measure x period array; NaN where the bootstrap has no estimate."""
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
    values = np.full((len(draw_ids), len(units), len(v_names), len(years)), np.nan, dtype=np.float32)
    values[keys["draw"].to_numpy(int), keys["unit"].to_numpy(int), keys["variable"].to_numpy(int),
           keys["year"].to_numpy(int)] = data["model_draw"].to_numpy(dtype=np.float32)
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


def build_state_inputs(bootstrap_path: Path, workbook_path: Path, output_path: Path,
                            eps: float = PROB_EPS, expected_draws: int = 500) -> dict:
    """Write the model-input NPZ and its audit; return the audit metrics."""
    workbook = ModelInputWorkbook(workbook_path, DynamicConfig().v_names)
    units = workbook.units()
    unit_ids, v_names, years = [str(u.id) for u in units], workbook.v_names, np.asarray(units[0].amis_years, dtype=int)
    data, draw_ids = load_bootstrap(bootstrap_path, expected_draws)
    values = dense_array(data, draw_ids, unit_ids, v_names, years)

    validate_complete(values)
    values = np.clip(values, eps, 1 - eps).astype(np.float32)

    inputs = StateInputs(
        observed_draws=values, observed_point=np.array([u.amis_values for u in units], dtype=float),
        unit_ids=unit_ids, unit_kinds=[u.kind for u in units],
        census_divisions=np.array([0 if u.census_div is None else u.census_div for u in units], dtype=int),
        sample_sizes=np.array([np.nan if u.sample_size is None else u.sample_size for u in units], dtype=float),
        covariance=workbook.covariance(), v_names=v_names, years=years,
        source_draw_ids=np.asarray(draw_ids, dtype=int),
    )
    inputs.save(output_path, source_bootstrap_path=bootstrap_path, workbook_path=workbook_path,
                workbook_sha256=hashlib.sha256(Path(workbook_path).read_bytes()).hexdigest())
    metrics = {"n_draws": values.shape[0], "n_units": values.shape[1], "n_variables": values.shape[2],
               "n_years": values.shape[3],
               "n_clipped_or_boundary": int(((values <= eps) | (values >= 1 - eps)).sum())}
    pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())}).to_csv(
        output_path.with_suffix(".audit.csv"), index=False)
    return metrics
