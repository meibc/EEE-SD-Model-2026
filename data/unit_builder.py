import pandas as pd
import numpy as np
from .unit import Unit


class UnitDataBuilder:
    """Builds Unit objects for nation, divisions, and states from raw data tables."""

    def __init__(self, tbl_all: pd.DataFrame, tbl_states: pd.DataFrame):
        self.tbl_all = tbl_all
        self.tbl_states = tbl_states

    def build_nation(self, ts, v_names) -> Unit:
        values = self._extract("USA", ts, v_names)
        return Unit(id="USA", kind="nation", values=values)

    def build_state(self, state_code: str, ts, v_names) -> Unit:
        values = self._extract(state_code, ts, v_names)
        meta = self._get_state_meta(state_code)
        return Unit(
            id=state_code,
            kind="state",
            values=values,
            sample_size=meta.get("sample_size"),
            census_div=meta.get("census_div"),
        )

    def build_division(self, div_id: int, ts, v_names) -> Unit:
        values = self._build_division_weighted(div_id, ts, v_names)
        return Unit(id=f"div_{div_id}", kind="division", values=values, census_div=div_id)

    def _extract(self, geo: str, ts, v_names) -> np.ndarray:
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

    def _build_division_weighted(self, div_id: int, ts, v_names) -> np.ndarray:
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
