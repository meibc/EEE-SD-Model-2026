from __future__ import annotations

"""Generate cached SEM Monte Carlo input draws from a Factor Analysis workbook.

This is the productionized version of the original `sem_mc_ci_demo_v3.ipynb`
input-draw stage. It intentionally only generates probability-space input
trajectories (`Xbars_samples` is the historical name); it does not fit J
matrices. Fitting is a separate stage because draw generation is slow and only
needs to be rerun when the source workbook or summary statistics change.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.sem import SEMConfig
from data.data_prep import DataPrep


DEFAULT_YEAR_GROUP_MAP = {
    2017: "2017-2018", 2018: "2017-2018",
    2019: "2019-2020", 2020: "2019-2020",
    2021: "2021-2022", 2022: "2021-2022",
    2023: "2023-2024", 2024: "2023-2024",
}

DEFAULT_T_INDEX = {
    "2017-2018": 0,
    "2019-2020": 1,
    "2021-2022": 2,
    "2023-2024": 3,
}


@dataclass(frozen=True)
class SEMInputDrawConfig:
    data_path: Path
    summary_path: Path
    output_path: Path = Path("output/sem_input_draws_v5.npz")
    n_draws: int = 500
    seed: int = 123
    value_source: str = "factor_analysis"
    eps: float = 1e-6


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def normalize_proportions(values, eps: float) -> np.ndarray:
    proportions = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.nanpercentile(proportions, 95) > 1.5:
        proportions = proportions / 100.0
    return np.clip(proportions, eps, 1 - eps)


def load_factor_analysis_values(data_path: Path) -> pd.DataFrame:
    trends = pd.read_excel(data_path, sheet_name="trends")
    require_columns(trends, {"Geography", "Year", "Indicator", "Value"}, "trends sheet")

    values = trends[["Geography", "Year", "Indicator", "Value"]].copy()
    values = values.rename(columns={"Geography": "state_calc", "Indicator": "variable", "Value": "fa_value"})
    values["year_group"] = pd.to_numeric(values["Year"], errors="coerce").map(DEFAULT_YEAR_GROUP_MAP)
    values["fa_value"] = pd.to_numeric(values["fa_value"], errors="coerce")
    values = values.dropna(subset=["year_group"])
    return values.groupby(["state_calc", "year_group", "variable"], as_index=False)["fa_value"].mean()


def build_beta_input_table(summary_path: Path, data_path: Path, value_source: str, eps: float) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    require_columns(summary, {"state_calc", "year_group", "variable", "n_eff"}, "summary CSV")

    if value_source == "weighted_prop":
        require_columns(summary, {"weighted_prop"}, "summary CSV")
        df = summary.copy()
        df["p_use"] = df["weighted_prop"]
    elif value_source == "factor_analysis":
        fa_values = load_factor_analysis_values(data_path)
        df = summary.merge(fa_values, on=["state_calc", "year_group", "variable"], how="left")
        df["p_use"] = df["fa_value"]
    else:
        raise ValueError(f"Unknown value_source: {value_source}")

    df["p_use"] = normalize_proportions(df["p_use"], eps)
    df["n_eff"] = pd.to_numeric(df["n_eff"], errors="coerce")
    df["alpha"] = df["p_use"] * df["n_eff"] + 1.0
    df["beta"] = (1.0 - df["p_use"]) * df["n_eff"] + 1.0
    return df


def fill_divisions_and_missing_states(
    X: np.ndarray,
    unit_order: list[str],
    v_names: list[str],
    tbl_states: pd.DataFrame,
    eps: float,
    fallback_unit: str = "USA",
) -> np.ndarray:
    """Fill division values and missing state cells after stochastic draws.

    Division rows are weighted averages of available state rows within each
    division. Missing state cells are then filled with the division value. This
    mirrors the original notebook behavior while adapting division IDs to the
    2026 repo's `div_#` naming convention.
    """
    X = np.asarray(X, dtype=float).copy()
    unit_idx = {str(u): i for i, u in enumerate(unit_order)}
    fallback_idx = unit_idx.get(fallback_unit)

    require_columns(tbl_states, {"state_calc", "Census_div", "n"}, "states sheet")
    states = tbl_states[["state_calc", "Census_div", "n"]].copy()
    states["state_calc"] = states["state_calc"].astype(str)
    states["Census_div"] = pd.to_numeric(states["Census_div"], errors="coerce").astype("Int64")
    states["n"] = pd.to_numeric(states["n"], errors="coerce")

    for div_id, div_states in states.dropna(subset=["Census_div"]).groupby("Census_div"):
        div_key = f"div_{int(div_id)}"
        if div_key not in unit_idx:
            continue
        state_ids = [s for s in div_states["state_calc"].tolist() if s in unit_idx]
        if not state_ids:
            continue

        state_indices = np.array([unit_idx[s] for s in state_ids], dtype=int)
        weights = div_states.set_index("state_calc").reindex(state_ids)["n"].to_numpy(dtype=float)
        valid_weights = np.isfinite(weights) & (weights > 0)
        weights = np.where(valid_weights, weights, 0.0) if valid_weights.any() else np.ones_like(weights)

        div_index = unit_idx[div_key]
        for var_idx in range(len(v_names)):
            for time_idx in range(X.shape[1]):
                vals = X[var_idx, time_idx, state_indices]
                observed = np.isfinite(vals)
                if observed.any():
                    w = weights[observed]
                    if np.sum(w) <= 0:
                        w = np.ones(np.sum(observed), dtype=float)
                    div_value = float(np.average(vals[observed], weights=w))
                elif fallback_idx is not None and np.isfinite(X[var_idx, time_idx, fallback_idx]):
                    div_value = float(X[var_idx, time_idx, fallback_idx])
                else:
                    div_value = 0.5

                div_value = float(np.clip(div_value, eps, 1 - eps))
                X[var_idx, time_idx, div_index] = div_value
                missing_state_indices = state_indices[~observed]
                if missing_state_indices.size:
                    X[var_idx, time_idx, missing_state_indices] = div_value

    if fallback_idx is not None:
        missing = ~np.isfinite(X)
        if missing.any():
            for var_idx in range(X.shape[0]):
                for time_idx in range(X.shape[1]):
                    fallback = X[var_idx, time_idx, fallback_idx]
                    if np.isfinite(fallback):
                        X[var_idx, time_idx, missing[var_idx, time_idx, :]] = fallback

    X = np.clip(X, eps, 1 - eps)
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))[:10]
        raise ValueError(f"Non-finite SEM input draws after fill: {bad}")
    return X


def generate_sem_input_draws(config: SEMInputDrawConfig) -> Path:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Missing Factor Analysis workbook: {config.data_path}")
    if not config.summary_path.exists():
        raise FileNotFoundError(f"Missing AMIS summary CSV: {config.summary_path}")

    sem_config = SEMConfig(data_path=config.data_path)
    data_prep = DataPrep(sem_config)
    prepared = data_prep.prepare_inputs()
    tbl_all, tbl_states = data_prep.loader.load_amis()

    unit_order = [str(u.id) for u in prepared.units]
    v_names = list(prepared.v_names)
    ts = np.asarray(prepared.ts, dtype=int)

    g_map = {geo: idx for idx, geo in enumerate(unit_order)}
    t_map = {k: v for k, v in DEFAULT_T_INDEX.items() if v < len(ts)}
    v_map = {name: idx for idx, name in enumerate(v_names)}

    beta_df = build_beta_input_table(config.summary_path, config.data_path, config.value_source, config.eps)
    beta_df["g_idx"] = beta_df["state_calc"].astype(str).map(g_map)
    beta_df["t_idx"] = beta_df["year_group"].map(t_map)
    beta_df["v_idx"] = beta_df["variable"].map(v_map)
    beta_df = beta_df.dropna(subset=["g_idx", "t_idx", "v_idx", "alpha", "beta"])
    beta_df = beta_df.astype({"g_idx": int, "t_idx": int, "v_idx": int})

    m, T, G = len(v_names), len(ts), len(unit_order)
    rng = np.random.default_rng(config.seed)
    Xbars_samples = np.full((config.n_draws, m, T, G), np.nan, dtype=np.float32)

    for sample_idx in range(config.n_draws):
        X = np.full((m, T, G), np.nan, dtype=float)
        for row in beta_df.itertuples(index=False):
            draw = rng.beta(row.alpha, row.beta)
            X[row.v_idx, row.t_idx, row.g_idx] = np.clip(draw, config.eps, 1 - config.eps)
        X = fill_divisions_and_missing_states(X, unit_order, v_names, tbl_states, config.eps)
        Xbars_samples[sample_idx] = X

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_path,
        Xbars_samples=Xbars_samples,
        unit_order=np.array(unit_order),
        v_names=np.array(v_names),
        ts=ts,
        seed=np.array(config.seed),
        S=np.array(config.n_draws),
        value_source=np.array(config.value_source),
        summary_path=str(config.summary_path),
        data_path=str(config.data_path),
        artifact_type=np.array("sem_input_draws"),
    )

    audit = pd.DataFrame({
        "metric": [
            "n_draws", "n_variables", "n_years", "n_units", "rows_used",
            "min_value", "max_value", "finite",
        ],
        "value": [
            config.n_draws, m, T, G, len(beta_df),
            float(np.nanmin(Xbars_samples)), float(np.nanmax(Xbars_samples)),
            bool(np.isfinite(Xbars_samples).all()),
        ],
    })
    audit.to_csv(config.output_path.with_suffix(".audit.csv"), index=False)
    return config.output_path
