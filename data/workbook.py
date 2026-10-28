"""The model-input workbook (scripts/1_data/03), read once by scripts/3_model/01, which stores what the model needs in the
model-input file (data/state_inputs.py); later steps read that file, not the workbook."""
from pathlib import Path

import numpy as np
import pandas as pd
from data.unit import Unit
from data.unit_builder import UnitDataBuilder


class ModelInputWorkbook:
    def __init__(self, workbook_path: Path, v_names: list[str]):
        self.path = workbook_path
        self.v_names = list(v_names)

    def units(self) -> list[Unit]:
        """The nation, then each Census division followed by its states (AMIS trends and states sheets)."""
        tbl_all = pd.read_excel(self.path, sheet_name="trends")
        tbl_states = pd.read_excel(self.path, sheet_name="states")
        ts_amis = sorted(tbl_all["Year"].unique())
        unit_builder = UnitDataBuilder(tbl_all, tbl_states)
        units = [unit_builder.build_nation(ts_amis, self.v_names)]
        for div_id in sorted(tbl_states["Census_div"].unique()):
            units.append(unit_builder.build_division(div_id, ts_amis, self.v_names))
            states_in_div = tbl_states[tbl_states["Census_div"] == div_id]["state_calc"].tolist()
            for state_code in states_in_div:
                units.append(unit_builder.build_state(state_code, ts_amis, self.v_names))
        return units

    def covariance(self) -> np.ndarray:
        """Pooled weighted covariance of the measures (workbook "cov" sheet)."""
        cov = pd.read_excel(self.path, sheet_name="cov")
        if list(cov.columns) != self.v_names:
            raise ValueError(f"Covariance sheet variables {list(cov.columns)} do not match model order {self.v_names}")
        return cov.to_numpy(dtype=float)
