"""The model-input workbook (scripts/1_data/03), read once by scripts/3_model/01, which stores what the model needs in the
model-input file (data/state_inputs.py); later steps read that file, not the workbook."""
from pathlib import Path

import numpy as np
import pandas as pd
from data.unit import Unit


class UnitDataBuilder:
    """Build nation, division and state units from the workbook's AMIS tables."""

    def __init__(self, tbl_all: pd.DataFrame, tbl_states: pd.DataFrame):
        self.tbl_all = tbl_all
        self.tbl_states = tbl_states

    def build_nation(self, years, measure_names) -> Unit:
        return Unit(
            id="USA",
            kind="nation",
            amis_years=np.asarray(years),
            amis_values=self._extract_amis("USA", years, measure_names),
        )

    def build_state(self, state_code: str, years, measure_names) -> Unit:
        meta = self._get_state_meta(state_code)
        return Unit(
            id=state_code,
            kind="state",
            amis_years=np.asarray(years),
            amis_values=self._extract_amis(state_code, years, measure_names),
            sample_size=meta.get("sample_size"),
            census_div=meta.get("census_div"),
        )

    def build_division(self, division_id: int, years, measure_names) -> Unit:
        return Unit(
            id=f"div_{division_id}",
            kind="division",
            amis_years=np.asarray(years),
            amis_values=self._build_division_weighted_amis(division_id, years, measure_names),
            census_div=division_id,
        )

    def _extract_amis(self, geography: str, years, measure_names) -> np.ndarray:
        """Return a measure x period array for one geography."""
        filtered = self.tbl_all[
            (self.tbl_all["Geography"] == geography)
            & self.tbl_all["Year"].isin(years)
            & self.tbl_all["Indicator"].isin(measure_names)
        ]
        if filtered.empty:
            raise ValueError(f"No data for geography '{geography}'")

        pivoted = filtered.pivot(index="Indicator", columns="Year", values="Value")
        # Reindexing is essential: every Unit must follow the model's measure and period order.
        return pivoted.reindex(index=measure_names, columns=years).to_numpy()

    def _get_state_meta(self, state_code: str) -> dict:
        """Return sample size and Census division for a state."""
        row = self.tbl_states[self.tbl_states["state_calc"] == state_code]
        if row.empty:
            return {}
        return {
            "sample_size": float(row["n"].iloc[0]),
            "census_div": int(row["Census_div"].iloc[0]),
        }

    def _build_division_weighted_amis(self, division_id: int, years, measure_names) -> np.ndarray:
        """Return respondent-count-weighted state averages for one Census division."""
        states_df = self.tbl_states[self.tbl_states["Census_div"] == division_id]
        if states_df.empty:
            raise ValueError(f"No states for division {division_id}")

        states = states_df["state_calc"].to_numpy()
        weights = states_df.set_index("state_calc")["n"]
        values = np.full((len(measure_names), len(years)), np.nan)
        for measure_idx, measure in enumerate(measure_names):
            for period_idx, year in enumerate(years):
                mask = (
                    (self.tbl_all["Indicator"] == measure)
                    & self.tbl_all["Geography"].isin(states)
                    & (self.tbl_all["Year"] == year)
                )
                rows = self.tbl_all.loc[mask, ["Geography", "Value"]]
                if not rows.empty:
                    cell_weights = rows["Geography"].map(weights).to_numpy()
                    cell_weights = cell_weights / cell_weights.sum()
                    values[measure_idx, period_idx] = np.sum(rows["Value"].to_numpy() * cell_weights)
        return values


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
