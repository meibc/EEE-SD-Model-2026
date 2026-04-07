
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_amis(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load AMIS data, filling missing 2016 stigma data with 2018 values."""
        tbl_all = pd.read_excel(self.data_path, sheet_name="trends")
        tbl_states = pd.read_excel(self.data_path, sheet_name="states")

        # Fill missing 2016 stigma data with 2018 values
        stigma_vars = ["stigma_ahs", "stigma_gss", "stigma_family"]
        for var in stigma_vars:
            for geo in tbl_all["Geography"].unique():
                mask_2018 = (
                    (tbl_all["Indicator"] == var) &
                    (tbl_all["Geography"] == geo) &
                    (tbl_all["Year"] == 2018)
                )
                mask_2016 = (
                    (tbl_all["Indicator"] == var) &
                    (tbl_all["Geography"] == geo) &
                    (tbl_all["Year"] == 2016)
                )
                val_2018 = tbl_all.loc[mask_2018, "Value"].values
                if len(val_2018) > 0 and mask_2016.any():
                    tbl_all.loc[mask_2016, "Value"] = val_2018[0]

        tbl_all = tbl_all[tbl_all["Year"] != 2016].reset_index(drop=True)
        return tbl_all, tbl_states
    
    def load_cdc(self):
        """load CDC data from Excel file"""
        df = pd.read_excel(self.data_path, sheet_name="CDC")
        return df

    def get_sign_mask(self):
        """Return sign mask matrix."""

        sign_matrix = np.array([
                [0, 1, 1, 1, -1, 0, 0, -1],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 1, 0, 1],
                [-1, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 1, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 1, 0]
            ])
        return sign_matrix
    
    def get_sigma_matrix(self):
        """Return covariance matrix for AMIS variables"""

        cov = np.array([
        [0.21702, 0.03379, 0.03571, -0.02594, -0.00294, 0.00662, -0.00879, -0.01150],
        [0.03379, 0.22943, 0.08292, 0.02499, 0.00117, 0.01351, 0.01681, 0.01132],
        [0.03571, 0.08292, 0.25001, 0.01790, -0.00041, 0.00838, 0.00780, 0.00711],
        [-0.02594, 0.02499, 0.01790, 0.17090, 0.02214, 0.01162, 0.05703, 0.05417],
        [-0.00294, 0.00117, -0.00041, 0.02214, 0.10255, 0.00456, 0.02420, 0.03122],
        [0.00662, 0.01351, 0.00838, 0.01162, 0.00456, 0.24592, 0.04485, 0.00765],
        [-0.00879, 0.01681, 0.00780, 0.05703, 0.02420, 0.04485, 0.20988, 0.06354],
        [-0.01150, 0.01132, 0.00711, 0.05417, 0.03122, 0.00765, 0.06354, 0.18736]
        ])
        return cov