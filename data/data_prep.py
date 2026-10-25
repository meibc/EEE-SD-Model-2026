import numpy as np
from data.unit_builder import UnitDataBuilder
from data.loader import DataLoader
from config.sem import SEMConfig
from pipeline.results import PreparedData


class DataPrep:
    def __init__(self, baseconfig: SEMConfig):
        self.base = baseconfig
        self.loader = DataLoader(self.base.data_path)

    def prepare_inputs(self) -> PreparedData:
        """Units (nation, divisions, states), structural mask and covariance for estimation."""
        tbl_all, tbl_states = self.loader.load_amis()

        sign_mask, _, _ = self.base.structural_prior_arrays()
        SigmaY = self.loader.load_covariance(self.base.v_names)
        m = sign_mask.shape[0]
        M = np.eye(m, dtype=int) + self.base.allowed_mask().astype(int)

        ts_amis = sorted(tbl_all["Year"].unique())
        v_names = self.base.v_names
        unit_builder = UnitDataBuilder(tbl_all, tbl_states)

        units = [unit_builder.build_nation(ts_amis, v_names)]
        for div_id in sorted(tbl_states["Census_div"].unique()):
            units.append(unit_builder.build_division(div_id, ts_amis, v_names))
            states_in_div = tbl_states[tbl_states["Census_div"] == div_id]["state_calc"].tolist()
            for state_code in states_in_div:
                units.append(unit_builder.build_state(state_code, ts_amis, v_names))

        return PreparedData(
            units=units,
            sign_mask=sign_mask,
            SigmaY=SigmaY,
            M=M,
            ts=np.array(ts_amis),
            v_names=v_names,
        )
