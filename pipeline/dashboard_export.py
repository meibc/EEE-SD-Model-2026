from __future__ import annotations

"""Export dashboard-compatible v8 standardized SEM parameter artifact.

This exporter intentionally does not include scenario results. It updates the
dashboard input/parameter file with the newly fitted centered-logit SEM fields
and preserves the existing dashboard CDC/hierarchical fields.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.sem import SEMConfig
from data.data_prep import DataPrep
from models.sbm.prediction.predictor import Predictor
from models.shared.alignment import extend_years
from core.math.transforms import Transforms


@dataclass(frozen=True)
class DashboardExportConfig:
    base_standardized_path: Path = Path("/Users/meibinchen/Documents/GitHub/EEE-Model-Dashboard-Integration/standardized_input_v7.npz")
    sem_fits_path: Path = Path("output/sem_fits_centered_drift_v5.npz")
    data_path: Path = Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx")
    output_path: Path = Path("output/standardized_input_v8.npz")
    schema_report_path: Path = Path("output/standardized_input_v8_schema.csv")


def _as_str_list(arr) -> list[str]:
    return [str(x) for x in list(arr)]


def _load_factor_obs(data_path: Path, geo_ids: list[str], v_names: list[str], years: np.ndarray) -> np.ndarray:
    prepared = DataPrep(SEMConfig(data_path=data_path)).prepare_inputs()
    unit_map = {str(unit.id): unit for unit in prepared.units}
    out = np.full((len(geo_ids), len(v_names), len(years)), np.nan, dtype=float)
    for g, geo in enumerate(geo_ids):
        if geo not in unit_map:
            continue
        unit = unit_map[geo]
        if list(unit.amis_names) != list(v_names):
            raise ValueError(f"Variable order mismatch for {geo}: {unit.amis_names} vs {v_names}")
        year_idx = {int(y): i for i, y in enumerate(unit.amis_years)}
        for t, year in enumerate(years):
            if int(year) in year_idx:
                out[g, :, t] = unit.amis_values[:, year_idx[int(year)]]
    if not np.isfinite(out).all():
        bad = np.argwhere(~np.isfinite(out))[:10]
        raise ValueError(f"Missing SEM observed values for dashboard export: {bad}")
    return np.clip(out, 1e-6, 1 - 1e-6)


def _build_sem_pred(
    sem_obs: np.ndarray,
    J_point: np.ndarray,
    drift_point: np.ndarray,
    reference_probs: np.ndarray,
    sem_pred_years: np.ndarray,
) -> np.ndarray:
    """Build point SEM predictions for the dashboard artifact.

    Prediction starts at the first observed SEM value and uses the point fitted
    final J/drift. For the current v7 dashboard schema, `sem_pred_years` is the
    same 4-year anchor sequence as `sem_obs_years`.
    """
    n_geo, m, _ = sem_obs.shape
    pred = np.full((n_geo, m, len(sem_pred_years)), np.nan, dtype=float)
    transforms = Transforms()
    predictor = Predictor(reference_probs=reference_probs)
    ref_logits = predictor.reference_logits
    if ref_logits is None:
        ref_logits = np.zeros(m, dtype=float)

    for g in range(n_geo):
        x0 = transforms.logit(sem_obs[g, :, 0]) - ref_logits
        y, _ = predictor.predict_trajectory(J_point[g], x0, drift_point[g], len(sem_pred_years))
        pred[g] = y
    return np.clip(pred, 1e-6, 1 - 1e-6)


def _schema_report(keys_before: list[str], arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    before = set(keys_before)
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.dtype.kind in {"f", "i", "u"} and arr.size:
            finite = bool(np.isfinite(arr).all())
            min_value = float(np.nanmin(arr))
            max_value = float(np.nanmax(arr))
        else:
            finite = None
            min_value = None
            max_value = None
        rows.append({
            "key": key,
            "status": "existing_updated" if key in before else "new_added",
            "shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "finite": finite,
            "min": min_value,
            "max": max_value,
        })
    return pd.DataFrame(rows).sort_values(["status", "key"]).reset_index(drop=True)


def export_dashboard_standardized_v8(config: DashboardExportConfig) -> Path:
    if not config.base_standardized_path.exists():
        raise FileNotFoundError(f"Missing base dashboard standardized file: {config.base_standardized_path}")
    if not config.sem_fits_path.exists():
        raise FileNotFoundError(f"Missing fitted SEM artifact: {config.sem_fits_path}")

    base = np.load(config.base_standardized_path, allow_pickle=True)
    fits = np.load(config.sem_fits_path, allow_pickle=True)

    arrays = {key: base[key] for key in base.files}
    geo_ids = _as_str_list(arrays["geo_ids"])
    sem_v_names = _as_str_list(arrays["sem_v_names"])
    sem_obs_years = np.asarray(arrays["sem_obs_years"], dtype=int)
    sem_pred_years = np.asarray(arrays["sem_pred_years"], dtype=int)

    fit_units = _as_str_list(fits["unit_order"])
    fit_vars = _as_str_list(fits["v_names"])
    if fit_vars != sem_v_names:
        raise ValueError(f"SEM variable order mismatch: fitted={fit_vars}, dashboard={sem_v_names}")

    missing = [geo for geo in geo_ids if geo not in fit_units]
    if missing:
        raise ValueError(f"Dashboard geographies missing from fitted SEM artifact: {missing}")
    fit_idx = np.array([fit_units.index(geo) for geo in geo_ids], dtype=int)

    J_samples = np.asarray(fits["Jmeans_stack"], dtype=float)[:, fit_idx, :, :]
    drift_samples = np.asarray(fits["drift_stack"], dtype=float)[:, fit_idx, :]
    reference_probs = np.asarray(fits["reference_probs"], dtype=float)

    J_point = np.nanmedian(J_samples, axis=0)
    drift_point = np.nanmedian(drift_samples, axis=0)
    sem_obs = _load_factor_obs(config.data_path, geo_ids, sem_v_names, sem_obs_years)
    sem_pred = _build_sem_pred(sem_obs, J_point, drift_point, reference_probs, sem_pred_years)

    arrays["sem_obs"] = sem_obs
    arrays["sem_pred"] = sem_pred
    arrays["sem_fit_J_last"] = J_point
    arrays["sem_J_samples"] = J_samples
    arrays["sem_fit_drift"] = drift_point
    arrays["sem_drift_samples"] = drift_samples
    arrays["sem_reference_probs"] = reference_probs

    # Defensive checks for the dashboard v7 state-only artifact.
    if "USA" in geo_ids:
        raise ValueError("Dashboard v8 export should be state-level only; USA is present in geo_ids")
    if len(geo_ids) != 49:
        raise ValueError(f"Expected 49 dashboard geographies, got {len(geo_ids)}")
    if "cdc_kappa_prep" in arrays and not np.allclose(np.asarray(arrays["cdc_kappa_prep"], dtype=float), 1.0):
        raise ValueError("Dashboard v8 export expected cdc_kappa_prep == 1.0 for all geographies")
    for key in ["sem_obs", "sem_pred", "sem_fit_J_last", "sem_J_samples", "sem_fit_drift", "sem_drift_samples"]:
        if not np.isfinite(np.asarray(arrays[key], dtype=float)).all():
            raise ValueError(f"Non-finite values in exported key: {key}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.output_path, **arrays)

    report = _schema_report(base.files, arrays)
    config.schema_report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.schema_report_path, index=False)
    return config.output_path
