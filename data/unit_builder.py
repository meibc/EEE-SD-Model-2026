from __future__ import annotations

import pandas as pd
import numpy as np
from config.data import DEFAULT_CDC_NAMES, STATE_TO_DIVISION
from .unit import Unit

class UnitDataBuilder:
    """Builds Unit objects for nation, divisions, and states from raw data tables."""

    def __init__(
        self,
        tbl_all: pd.DataFrame,
        tbl_states: pd.DataFrame,
        tbl_cdc: pd.DataFrame | None = None,
        cdc_names: list[str] = DEFAULT_CDC_NAMES,
    ):
        self.tbl_all = tbl_all
        self.tbl_states = tbl_states
        self.tbl_cdc = tbl_cdc
        self.cdc_names = cdc_names


    def build_nation(self, ts_amis, v_names, ts_cdc=None) -> Unit:
        amis_values = self._extract_amis("USA", ts_amis, v_names)
        cdc_values, cdc_years, cdc_names = self._get_cdc("USA", ts_cdc)
        
        return Unit(
            id="USA",
            kind="nation",
            amis_years=np.asarray(ts_amis),
            amis_values=amis_values,
            amis_names=v_names,
            cdc_years=cdc_years,
            cdc_values=cdc_values,
            cdc_names=cdc_names,
        )

    def build_state(self, state_code: str, ts_amis, v_names, ts_cdc=None) -> Unit:
        amis_values = self._extract_amis(state_code, ts_amis, v_names)
        cdc_values, cdc_years, cdc_names = self._get_cdc(state_code, ts_cdc)
        meta = self._get_state_meta(state_code)
        
        return Unit(
            id=state_code,
            kind="state",
            amis_years=np.asarray(ts_amis),
            amis_values=amis_values,
            amis_names=v_names,
            cdc_years=cdc_years,
            cdc_values=cdc_values,
            cdc_names=cdc_names,
            sample_size=meta.get("sample_size"),
            census_div=meta.get("census_div"),
        )

    def build_division(self, div_id: int, ts_amis, v_names, ts_cdc=None) -> Unit:
        amis_values = self._build_division_weighted_amis(div_id, ts_amis, v_names)
        cdc_values, cdc_years, cdc_names = self._get_cdc_division(div_id, ts_cdc)
        
        return Unit(
            id=f"div_{div_id}",
            kind="division",
            amis_years=np.asarray(ts_amis),
            amis_values=amis_values,
            amis_names=v_names,
            cdc_years=cdc_years,
            cdc_values=cdc_values,
            cdc_names=cdc_names,
            census_div=div_id,
        )

    def _extract_amis(self, geo: str, ts, v_names) -> np.ndarray:
        """Extract (m, T) array for a geography."""
        filtered = self.tbl_all[
            (self.tbl_all["Geography"] == geo) &
            (self.tbl_all["Year"].isin(ts)) &
            (self.tbl_all["Indicator"].isin(v_names))
        ]
        
        if filtered.empty:
            raise ValueError(f"No data for geography '{geo}'")
        
        pivoted = filtered.pivot(index="Indicator", columns="Year", values="Value")
        pivoted = pivoted.reindex(index=v_names, columns=ts)  # Ensure correct order

        # fill missing vals with 0. (overwritten later)
        
        return pivoted.fillna(0.0).values

    def _get_state_meta(self, state_code: str) -> dict:
        """Get sample size and census division for a state."""
        row = self.tbl_states[self.tbl_states["state_calc"] == state_code]
        
        if row.empty:
            return {}
        
        return {
            "sample_size": float(row["n"].values[0]),
            "census_div": int(row["Census_div"].values[0]),
        }

    def _build_division_weighted_amis(self, div_id: int, ts, v_names) -> np.ndarray:
        """Population-weighted average across states in division."""
        states_df = self.tbl_states[self.tbl_states["Census_div"] == div_id]
        if states_df.empty:
            raise ValueError(f"No states for division {div_id}")

        states = states_df["state_calc"].values
        weights = states_df.set_index("state_calc")["n"]

        m, T = len(v_names), len(ts)
        Ybar = np.zeros((m, T))

        for j, var in enumerate(v_names):
            for t, year in enumerate(ts):
                mask = (
                    (self.tbl_all["Indicator"] == var) &
                    (self.tbl_all["Geography"].isin(states)) &
                    (self.tbl_all["Year"] == year)
                )
                rows = self.tbl_all.loc[mask, ["Geography", "Value"]]

                if not rows.empty:
                    w = rows["Geography"].map(weights).values
                    w = w / w.sum()
                    Ybar[j, t] = (rows["Value"].values * w).sum()

        return Ybar

    # -------------------------------------------------------------------------
    # CDC Extraction
    # -------------------------------------------------------------------------

    def _get_cdc(self, geo: str, ts_cdc) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
        """Extract CDC data for a geography. Returns (values, years, names)."""
        if self.tbl_cdc is None or ts_cdc is None:
            return None, None, None
        
        cdc_values = self._extract_cdc(geo, ts_cdc)
        if cdc_values is None:
            return None, None, None
        
        return cdc_values, np.asarray(ts_cdc), self.cdc_names

    def _get_cdc_division(self, div_id: int, ts_cdc) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
        """Extract CDC data for a division. Returns (values, years, names)."""
        if self.tbl_cdc is None or ts_cdc is None:
            return None, None, None
        
        cdc_values = self._build_division_summed_cdc(div_id, ts_cdc)
        if cdc_values is None:
            return None, None, None
        
        return cdc_values, np.asarray(ts_cdc), self.cdc_names

    def _extract_cdc(self, geo: str, ts) -> np.ndarray | None:
        """Extract (m, T) array for a geography from CDC data."""
        filtered = self.tbl_cdc[
            (self.tbl_cdc["Geography"] == geo) &
            (self.tbl_cdc["Year"].isin(ts))
        ]
        
        if filtered.empty:
            return None
        
        pivoted = filtered.pivot(index="Indicator", columns="Year", values="Value")
        pivoted = pivoted.reindex(index=self.cdc_names, columns=ts)
        
        return pivoted.values  # NaN where missing

    def _build_division_summed_cdc(self, div_id: int, ts) -> np.ndarray | None:
        """Sum CDC counts across states in division."""
        states = [s for s, d in STATE_TO_DIVISION.items() if d == div_id]
        
        if not states:
            return None
        
        state_arrays = []
        for state in states:
            arr = self._extract_cdc(state, ts)
            if arr is not None:
                state_arrays.append(arr)
        
        if not state_arrays:
            return None
        
        stacked = np.stack(state_arrays, axis=0)  # (n_states, m, T)
        return np.nansum(stacked, axis=0)          # (m, T)

  
