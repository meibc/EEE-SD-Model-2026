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
        """Prepare inputs for estimation."""
        tbl_all, tbl_states = self.loader.load_amis()
        tbl_cdc = self.loader.load_cdc()

        sign_mask_from_prior, _, _ = self.base.structural_prior_arrays()
        sign_mask = sign_mask_from_prior if sign_mask_from_prior is not None else self.loader.get_sign_mask()
        SigmaY = self.loader.get_sigma_matrix()
        m = sign_mask.shape[0]
        M = np.eye(m, dtype=int) + (sign_mask != 0).astype(int)

        ts_amis = sorted(tbl_all["Year"].unique())
        ts_cdc = (
            sorted(tbl_cdc["Year"].dropna().unique())
            if tbl_cdc is not None and not tbl_cdc.empty
            else None
        )

        v_names = self.base.v_names
        unit_builder = UnitDataBuilder(tbl_all, tbl_states, tbl_cdc=tbl_cdc)

        units = []
        units.append(unit_builder.build_nation(ts_amis, v_names, ts_cdc))
        for div_id in sorted(tbl_states["Census_div"].unique()):
            units.append(unit_builder.build_division(div_id, ts_amis, v_names, ts_cdc))
            states_in_div = tbl_states[tbl_states["Census_div"] == div_id]["state_calc"].tolist()
            for state_code in states_in_div:
                units.append(unit_builder.build_state(state_code, ts_amis, v_names, ts_cdc))

        return PreparedData(
            units=units,
            sign_mask=sign_mask,
            SigmaY=SigmaY,
            M=M,
            ts=np.array(ts_amis),
            v_names=v_names,
            ts_cdc=np.array(ts_cdc) if ts_cdc is not None else None,
            cdc_names=unit_builder.cdc_names if ts_cdc is not None else None,
        )
