from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine.results import JointOutput, RunOutput, UncertaintyResult
from models.shared.alignment import extend_years


def export_unified_table(
    output_path: Path,
    sem_output: RunOutput | None = None,
    joint_output: JointOutput | None = None,
    uncertainty: dict[str, UncertaintyResult] | None = None,
    quantiles: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """
    Export a unified long-format table combining SEM and CDC outputs.

    Columns:
      unit_id, year, source, variable, stat, value
    """
    rows: list[dict] = []

    if sem_output is not None:
        units = {u.id: u for u in sem_output.inputs.units}
        sem_hist_years = np.asarray(sem_output.inputs.ts, dtype=float)
        sem_var_names = list(sem_output.inputs.v_names)

        for unit_id, unit in units.items():
            for i, var in enumerate(sem_var_names):
                vals = unit.amis_values[i]
                for year, value in zip(sem_hist_years, vals):
                    rows.append(
                        {
                            "unit_id": unit_id,
                            "year": int(year),
                            "source": "sem_raw",
                            "variable": var,
                            "stat": "point",
                            "value": float(value),
                        }
                    )

        if sem_output.predictions is not None:
            pred = sem_output.predictions
            sem_pred_var_names = list(pred.v_names)
            for unit_id, pred_res in pred.results.items():
                vals = np.asarray(pred_res.Ypred_trajectory, dtype=float)  # (m, T_pred)
                years = extend_years(pred.ts, vals.shape[1])
                for i, var in enumerate(sem_pred_var_names):
                    for year, value in zip(years, vals[i]):
                        rows.append(
                            {
                                "unit_id": unit_id,
                                "year": int(year),
                                "source": "sem_pred",
                                "variable": var,
                                "stat": "point",
                                "value": float(value),
                            }
                        )

    if joint_output is not None:
        cdc_vars = ["prep_on_count", "incidence", "diagnosed", "undiagnosed"]
        for unit_id, result in joint_output.results.items():
            years = np.asarray(result.cdc_output.years, dtype=float)
            for var in cdc_vars:
                vals = np.asarray(getattr(result.cdc_output, var), dtype=float)
                for year, value in zip(years, vals):
                    rows.append(
                        {
                            "unit_id": unit_id,
                            "year": int(year),
                            "source": "cdc_pred",
                            "variable": var,
                            "stat": "point",
                            "value": float(value),
                        }
                    )

        if sem_output is not None:
            unit_map = {u.id: u for u in sem_output.inputs.units}
            for unit_id, result in joint_output.results.items():
                unit = unit_map.get(unit_id)
                if unit is None or unit.cdc_values is None or unit.cdc_names is None:
                    continue
                years = np.asarray(unit.cdc_years, dtype=float)
                for i, var in enumerate(unit.cdc_names):
                    vals = np.asarray(unit.cdc_values[i], dtype=float)
                    for year, value in zip(years, vals):
                        if np.isnan(value):
                            continue
                        rows.append(
                            {
                                "unit_id": unit_id,
                                "year": int(year),
                                "source": "cdc_raw",
                                "variable": var,
                                "stat": "point",
                                "value": float(value),
                            }
                        )

    if uncertainty is not None:
        cdc_vars = ["prep_on_count", "incidence", "diagnosed", "undiagnosed"]
        for unit_id, result in uncertainty.items():
            years = np.asarray(result.years, dtype=float)
            for var in cdc_vars:
                qvals = result.get_quantiles(var, q=list(quantiles))
                for q in quantiles:
                    label = f"q{q:.3f}".rstrip("0").rstrip(".")
                    for year, value in zip(years, qvals[q]):
                        rows.append(
                            {
                                "unit_id": unit_id,
                                "year": int(year),
                                "source": "cdc_pred_unc",
                                "variable": var,
                                "stat": label,
                                "value": float(value),
                            }
                        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["unit_id", "year", "source", "variable", "stat"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
