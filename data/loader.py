import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_amis(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the AMIS trends and states sheets of the model input workbook (scripts/01)."""
        tbl_all = pd.read_excel(self.data_path, sheet_name="trends")
        tbl_states = pd.read_excel(self.data_path, sheet_name="states")
        return tbl_all, tbl_states

    def load_covariance(self, v_names: list[str]) -> np.ndarray:
        """Pooled weighted covariance of the measures (workbook "cov" sheet, scripts/01)."""
        cov = pd.read_excel(self.data_path, sheet_name="cov")
        if list(cov.columns) != list(v_names):
            raise ValueError(f"Covariance sheet variables {list(cov.columns)} do not match model order {v_names}")
        return cov.to_numpy(dtype=float)
