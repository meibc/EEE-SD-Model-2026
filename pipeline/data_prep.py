import numpy as np
from data.unit_builder import UnitDataBuilder
from data.loader import DataLoader
from config.base import BaseConfig
from pipeline.results import PreparedData


class DataPrep:
    def __init__(self, baseconfig: BaseConfig):
        self.base = baseconfig
        self.loader = DataLoader(self.base.data_path)

    def prepare_inputs(self) -> PreparedData:
        """Prepare inputs for estimation."""
        tbl_all, tbl_states = self.loader.load_amis()
        sign_mask = self.loader.get_sign_mask()
        SigmaY = self.loader.get_sigma_matrix()
        m = sign_mask.shape[0]
        M = np.eye(m) + (sign_mask != 0).astype(int)
        
        unit_builder = UnitDataBuilder(tbl_all, tbl_states)
        ts = sorted(tbl_all["Year"].unique())
        v_names = self.base.v_names

        # Build units in hierarchy order
        units = []
        
        # 1. Nation first
        units.append(unit_builder.build_nation(ts, v_names))
        
        # 2. Divisions and their states (grouped)
        for div_id in sorted(tbl_states["Census_div"].unique()):
            # Division
            units.append(unit_builder.build_division(div_id, ts, v_names))
            
            # States in this division
            states_in_div = tbl_states[tbl_states["Census_div"] == div_id]["state_calc"].tolist()
            for state_code in states_in_div:
                units.append(unit_builder.build_state(state_code, ts, v_names))

        return PreparedData(
            units=units,
            sign_mask=sign_mask,
            SigmaY=SigmaY,
            M=M,
            ts=np.array(ts),
            v_names=v_names,
        )