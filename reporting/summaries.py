"""Summaries shared by the results scripts: scenario effects pooled across states and by state,
observed changes from the input draws, and interval helpers. Intervals are 2.5th-97.5th percentiles
across draws; p25-p75 is the interquartile range."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from data.state_inputs import StateInputs
from config.projection import TARGET_END_YEAR
from reporting import sources

END_YEAR = TARGET_END_YEAR
OBSERVED_YEARS = (2018, 2024)


def interval(values) -> dict[str, float]:
    """Median, interquartile range (p25, p75) and 95% interval (low, high)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:     # e.g. PrEP gain per AHS point when the scenario removes no AHS in any draw
        return dict.fromkeys(("median", "p25", "p75", "low", "high"), float("nan"))
    q = np.quantile(v, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {"median": float(q[2]), "p25": float(q[1]), "p75": float(q[3]), "low": float(q[0]), "high": float(q[4])}


def primary_states() -> list[str]:
    return pd.read_csv(sources.PRIMARY_STATES)["state"].astype(str).tolist()


@lru_cache(maxsize=None)
def epi_draws(scenario_dir: Path) -> pd.DataFrame:
    """Draw-level epidemiological endpoints from a scenario run (scripts/3_model/05 --save-draw-level-endpoints)."""
    return pd.read_csv(Path(scenario_dir) / "scenario_endpoint_epi_draws.csv.gz")


def pooled_effect(scenario_dir: Path, scenario: str, year: int = END_YEAR) -> dict[str, float]:
    """Effect summed over the primary states, per draw, then summarized: percentage change in incidence in
    ``year``, and infections averted from 2025 through ``year`` (number and percentage of baseline)."""
    d = epi_draws(scenario_dir)
    d = d[d["scenario"].eq(scenario) & d["year"].eq(year) & d["unit_id"].isin(primary_states())]
    totals = d.groupby(["variable", "draw"])[["baseline", "scenario_value"]].sum()
    inc, cum = totals.loc["incidence"], totals.loc["incidence_cumulative"]
    out = {}
    for name, values in {"incidence_pct": 100 * (inc["scenario_value"] / inc["baseline"] - 1),
                         "averted": cum["baseline"] - cum["scenario_value"],
                         "averted_pct": 100 * (1 - cum["scenario_value"] / cum["baseline"])}.items():
        out.update({f"{name}_{k}": v for k, v in interval(values).items()})
    out["baseline_cumulative_median"] = float(np.median(cum["baseline"]))
    return out


def state_effects(scenario_dir: Path, scenario: str, variable: str = "incidence", year: int = END_YEAR) -> pd.DataFrame:
    """Percentage change in ``variable`` in ``year`` by state: median and 95% interval across draws."""
    d = epi_draws(scenario_dir)
    d = d[d["scenario"].eq(scenario) & d["year"].eq(year) & d["variable"].eq(variable) & d["unit_id"].isin(primary_states())]
    rows = {state: interval(g["pct_change"]) for state, g in d.groupby("unit_id")}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("state")


@lru_cache(maxsize=None)
def input_draws() -> StateInputs:
    """The model-input file (observed draws: draw x unit x measure x period)."""
    return StateInputs.load(sources.INPUT_DRAWS)


def observed_change(measure: str, first: int = OBSERVED_YEARS[0], last: int = OBSERVED_YEARS[1]) -> pd.DataFrame:
    """Observed level in ``first`` and ``last`` and their difference (percentage points) by primary state, with
    95% intervals across the respondent-bootstrap draws. Direction is 'increase' or 'decrease' when the 95%
    interval of the difference excludes zero, otherwise 'no clear change'."""
    inputs = input_draws()
    years = list(inputs.years)
    rows = {}
    for state in primary_states():
        series = 100 * inputs.observed_draws[:, inputs.unit_ids.index(state), inputs.v_names.index(measure)]
        a, b = series[:, years.index(first)], series[:, years.index(last)]
        diff = interval(b - a)
        direction = "increase" if diff["low"] > 0 else "decrease" if diff["high"] < 0 else "no clear change"
        rows[state] = {f"level_{first}": float(np.median(a)), f"level_{last}": float(np.median(b)),
                       "change": diff["median"], "change_low": diff["low"], "change_high": diff["high"], "direction": direction}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("state")


def observed_paths(measures: tuple[str, ...]) -> pd.DataFrame:
    """Median observed level (percent) by primary state, measure and period."""
    inputs = input_draws()
    x, v_names, units = inputs.observed_draws, inputs.v_names, inputs.unit_ids
    rows = [{"state": s, "measure": m, "year": int(y), "level": float(np.median(100 * x[:, units.index(s), v_names.index(m), i]))}
            for s in primary_states() for m in measures for i, y in enumerate(inputs.years)]
    return pd.DataFrame(rows)


def projected_ahs() -> pd.DataFrame:
    """AHS by primary state: observed in 2023-24 and the 2036 baseline projection with the fitted
    PrEP -> AHS relationship continued and not continued (median, interquartile range, 95% interval
    across draws, and the share of draws above 30% and 50%)."""
    out = observed_change("stigma_ahs")[["level_2024"]].rename(columns={"level_2024": "observed_2024"})
    for name, path in {"continued": sources.CONTINUED, "not_continued": sources.NOT_CONTINUED}.items():
        s = pd.read_csv(path / "scenario_endpoint_dynamic_draws.csv.gz")
        s = s[s["year"].eq(END_YEAR) & s["variable"].eq("stigma_ahs") & s["scenario"].eq("halve_ahs")]
        g = s.groupby("unit_id")["baseline"]
        for label, q in {"": 0.5, "_p25": 0.25, "_p75": 0.75, "_low": 0.025, "_high": 0.975}.items():
            out[f"{name}_2036{label}"] = 100 * g.quantile(q)
        out[f"{name}_share_above_30"] = g.apply(lambda x: float((x > 0.30).mean()))
        out[f"{name}_share_above_50"] = g.apply(lambda x: float((x > 0.50).mean()))
    return out.loc[primary_states()]
