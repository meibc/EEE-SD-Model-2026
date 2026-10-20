#!/usr/bin/env python3
from __future__ import annotations

"""Convert AMIS respondent-bootstrap estimates into SEM input-draw format.

The R preparation notebook writes one probability-scale value per
draw/unit/year/variable. This adapter:

1. uses the production model repository to recover the exact unit, variable,
   and year order expected by the SEM fitter;
2. inserts the joint respondent-bootstrap values into that dense array;
3. reconstructs division and national draws as sample-size-weighted aggregates
   of the state draws, matching the production Factor Analysis v5 geography
   construction;
4. fills survey cells missing for low-information states from the matching
   Census-division draw, mirroring the production Beta-input workflow; and
5. writes the same ``Xbars_samples`` NPZ interface consumed by
   ``scripts/02_fit_sem_draws.py``.

It does not fit the SEM and does not modify an existing standardized input.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL_REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx")
DEFAULT_MSM_POP_PATH = Path(
    "/Users/meibinchen/Documents/GitHub/MSM-Population-Estimates/"
    "outputs/final/msm_state_summary.csv"
)
DEFAULT_BOOTSTRAP_PATH = Path(
    "output/amis_joint_bootstrap_v1_n500/amis_joint_model_draws_v1.csv.gz"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/amis_joint_bootstrap_v1_n500/sem_input_draws_joint_bootstrap_n500.npz"
)

REQUIRED_COLUMNS = {
    "draw",
    "unit",
    "geography_type",
    "year",
    "variable",
    "model_draw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SEM-compatible NPZ inputs from joint AMIS bootstrap draws"
    )
    parser.add_argument("--bootstrap-path", type=Path, default=DEFAULT_BOOTSTRAP_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-repo", type=Path, default=DEFAULT_MODEL_REPO)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--msm-pop-path", type=Path, default=DEFAULT_MSM_POP_PATH)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--expected-draws",
        type=int,
        default=500,
        help="Fail unless this many complete source replicates are present",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_production_layout(model_repo: Path, data_path: Path, msm_pop_path: Path):
    """Load layout through the production data-preparation code."""
    require_file(model_repo / "config" / "sem.py", "production SEM configuration")
    require_file(data_path, "factor-analysis workbook")
    require_file(msm_pop_path, "state MSM-population estimates")

    repo_text = str(model_repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    from config.sem import SEMConfig  # pylint: disable=import-outside-toplevel
    from data.data_prep import DataPrep  # pylint: disable=import-outside-toplevel

    # SEMConfig intentionally keeps several repository-relative configuration
    # paths. Resolve those through the production repository without changing
    # the caller's working directory after this function returns.
    original_cwd = Path.cwd()
    try:
        os.chdir(model_repo)
        sem_config = SEMConfig(data_path=data_path)
        data_prep = DataPrep(sem_config)
        prepared = data_prep.prepare_inputs()
        _, states = data_prep.loader.load_amis()
    finally:
        os.chdir(original_cwd)

    unit_order = [str(unit.id) for unit in prepared.units]
    v_names = [str(v) for v in prepared.v_names]
    years = np.asarray(prepared.ts, dtype=int)

    state_meta = states[["state_calc", "Census_div", "n"]].copy()
    state_meta["state_calc"] = state_meta["state_calc"].astype(str)
    state_meta["Census_div"] = pd.to_numeric(
        state_meta["Census_div"], errors="raise"
    ).astype(int)
    state_meta["n"] = pd.to_numeric(state_meta["n"], errors="coerce")
    msm_pop = pd.read_csv(msm_pop_path)
    required_msm = {"state", "rounded"}
    if not required_msm.issubset(msm_pop.columns):
        raise ValueError(
            f"MSM population file missing fields: {sorted(required_msm.difference(msm_pop.columns))}"
        )
    msm_pop = msm_pop[["state", "rounded"]].rename(
        columns={"state": "state_calc", "rounded": "state_msm_pop"}
    )
    msm_pop["state_calc"] = msm_pop["state_calc"].astype(str)
    msm_pop["state_msm_pop"] = pd.to_numeric(
        msm_pop["state_msm_pop"], errors="coerce"
    )
    state_meta = state_meta.merge(msm_pop, on="state_calc", how="left", validate="one_to_one")
    return unit_order, v_names, years, state_meta


def load_bootstrap(path: Path, expected_draws: int) -> tuple[pd.DataFrame, list[int]]:
    require_file(path, "joint-bootstrap CSV")
    data = pd.read_csv(
        path,
        usecols=lambda name: name in REQUIRED_COLUMNS,
        dtype={
            "draw": "int32",
            "unit": "string",
            "geography_type": "string",
            "year": "int16",
            "variable": "string",
            "model_draw": "float64",
        },
    )
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Joint-bootstrap CSV missing fields: {sorted(missing)}")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        counts = data[list(REQUIRED_COLUMNS)].isna().sum()
        raise ValueError(f"Joint-bootstrap CSV contains missing required values:\n{counts}")

    draw_ids = sorted(data["draw"].astype(int).unique().tolist())
    if len(draw_ids) != expected_draws:
        raise ValueError(
            f"Expected {expected_draws} draws, found {len(draw_ids)} "
            f"({draw_ids[:3]}...{draw_ids[-3:]})"
        )
    return data, draw_ids


def build_dense_array(
    data: pd.DataFrame,
    draw_ids: list[int],
    unit_order: list[str],
    v_names: list[str],
    years: np.ndarray,
) -> np.ndarray:
    draw_map = {draw: idx for idx, draw in enumerate(draw_ids)}
    unit_map = {unit: idx for idx, unit in enumerate(unit_order)}
    var_map = {var: idx for idx, var in enumerate(v_names)}
    year_map = {int(year): idx for idx, year in enumerate(years)}

    unknown_units = sorted(set(data["unit"].astype(str)).difference(unit_map))
    unknown_vars = sorted(set(data["variable"].astype(str)).difference(var_map))
    unknown_years = sorted(set(data["year"].astype(int)).difference(year_map))
    if unknown_units or unknown_vars or unknown_years:
        raise ValueError(
            "Bootstrap labels do not match production layout: "
            f"unknown_units={unknown_units}, unknown_vars={unknown_vars}, "
            f"unknown_years={unknown_years}"
        )

    data = data.copy()
    data["draw_idx"] = data["draw"].map(draw_map)
    data["unit_idx"] = data["unit"].astype(str).map(unit_map)
    data["var_idx"] = data["variable"].astype(str).map(var_map)
    data["year_idx"] = data["year"].astype(int).map(year_map)

    key_columns = ["draw_idx", "unit_idx", "year_idx", "var_idx"]
    duplicates = data.duplicated(key_columns, keep=False)
    if duplicates.any():
        sample = data.loc[duplicates, ["draw", "unit", "year", "variable"]].head()
        raise ValueError(f"Duplicate bootstrap cells found:\n{sample}")

    shape = (len(draw_ids), len(v_names), len(years), len(unit_order))
    values = np.full(shape, np.nan, dtype=np.float32)
    values[
        data["draw_idx"].to_numpy(dtype=int),
        data["var_idx"].to_numpy(dtype=int),
        data["year_idx"].to_numpy(dtype=int),
        data["unit_idx"].to_numpy(dtype=int),
    ] = data["model_draw"].to_numpy(dtype=np.float32)
    return values


def fill_missing_states_from_divisions(
    values: np.ndarray,
    unit_order: list[str],
    state_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Fill only missing state cells; never overwrite observed bootstrap cells."""
    unit_map = {unit: idx for idx, unit in enumerate(unit_order)}
    audit_rows: list[dict] = []

    for row in state_meta.itertuples(index=False):
        state = str(row.state_calc)
        division = f"div_{int(row.Census_div)}"
        if state not in unit_map:
            raise ValueError(f"Production state absent from unit order: {state}")
        if division not in unit_map:
            raise ValueError(f"Division draw absent from unit order: {division}")

        state_idx = unit_map[state]
        division_idx = unit_map[division]
        missing = ~np.isfinite(values[:, :, :, state_idx])
        division_values = values[:, :, :, division_idx]
        unavailable = missing & ~np.isfinite(division_values)
        if unavailable.any():
            raise ValueError(
                f"Cannot fill {int(unavailable.sum())} cells for {state}; "
                f"matching {division} values are missing"
            )

        n_filled = int(missing.sum())
        if n_filled:
            state_values = values[:, :, :, state_idx]
            state_values[missing] = division_values[missing]
            values[:, :, :, state_idx] = state_values
        audit_rows.append(
            {
                "unit": state,
                "division": division,
                "n_values_total": int(np.prod(missing.shape)),
                "n_values_filled": n_filled,
                "share_values_filled": float(missing.mean()),
            }
        )
    return pd.DataFrame(audit_rows)


def rebuild_aggregates_from_state_draws(
    values: np.ndarray,
    unit_order: list[str],
    state_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild division and USA draws from originally observed state draws.

    Factor Analysis v5 constructs broader geographies as weighted aggregates
    of state estimates using the state-level ``n`` field. Reconstructing the
    bootstrap aggregates here keeps every level of a draw coherent with that
    production definition. This function must run before missing state cells
    are filled from their divisions, so imputed state values cannot feed back
    into the aggregate that supplied them.
    """
    unit_map = {unit: idx for idx, unit in enumerate(unit_order)}
    meta = state_meta.loc[state_meta["state_calc"].isin(unit_map)].copy()
    if meta["n"].isna().any() or (meta["n"] <= 0).any():
        bad = meta.loc[meta["n"].isna() | (meta["n"] <= 0), "state_calc"].tolist()
        raise ValueError(f"Invalid state aggregation weights for: {bad}")

    audit_rows: list[dict] = []

    def rebuild(target: str, members: pd.DataFrame, weight_column: str) -> None:
        if target not in unit_map:
            raise ValueError(f"Aggregate unit absent from production order: {target}")
        member_idx = [unit_map[str(state)] for state in members["state_calc"]]
        weights = members[weight_column].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"Invalid {weight_column} values while rebuilding {target}")
        state_values = values[:, :, :, member_idx].astype(np.float64, copy=False)
        observed = np.isfinite(state_values)
        weighted_values = np.where(observed, state_values * weights, 0.0)
        denominators = np.sum(observed * weights, axis=-1)
        if np.any(denominators <= 0):
            bad = np.argwhere(denominators <= 0)[:10]
            raise ValueError(
                f"No observed state values available to rebuild {target}: {bad.tolist()}"
            )
        rebuilt = np.sum(weighted_values, axis=-1) / denominators
        target_idx = unit_map[target]
        supplied = values[:, :, :, target_idx].astype(np.float64, copy=True)
        overlap = np.isfinite(supplied)
        abs_diff = np.abs(supplied[overlap] - rebuilt[overlap])
        values[:, :, :, target_idx] = rebuilt.astype(np.float32)
        audit_rows.append(
            {
                "unit": target,
                "n_member_states": int(len(member_idx)),
                "n_values_rebuilt": int(rebuilt.size),
                "median_abs_difference_from_supplied": (
                    float(np.median(abs_diff)) if abs_diff.size else np.nan
                ),
                "maximum_abs_difference_from_supplied": (
                    float(np.max(abs_diff)) if abs_diff.size else np.nan
                ),
            }
        )

    for division in sorted(meta["Census_div"].unique()):
        rebuild(
            f"div_{int(division)}",
            meta.loc[meta["Census_div"] == division],
            "n",
        )
    rebuild("USA", meta, "state_msm_pop")
    return pd.DataFrame(audit_rows)


def validate_complete(
    values: np.ndarray,
    unit_order: list[str],
    eps: float,
) -> dict[str, float | int | bool]:
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[:10]
        raise ValueError(f"Non-finite SEM inputs remain after filling: {bad.tolist()}")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("SEM input probabilities fall outside [0, 1]")

    clipped = np.clip(values, eps, 1 - eps).astype(np.float32, copy=False)
    values[...] = clipped
    unit_map = {unit: idx for idx, unit in enumerate(unit_order)}
    robust_states = [
        "MA", "NJ", "NY", "PA", "IL", "IN", "MI", "OH", "FL",
        "GA", "MD", "NC", "TN", "TX", "VA", "CA", "WA",
    ]
    robust_idx = [unit_map[s] for s in robust_states]
    robust = values[:, :, :, robust_idx]
    return {
        "n_draws": int(values.shape[0]),
        "n_variables": int(values.shape[1]),
        "n_years": int(values.shape[2]),
        "n_units": int(values.shape[3]),
        "finite": bool(np.isfinite(values).all()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "n_boundary_all": int(((values <= eps) | (values >= 1 - eps)).sum()),
        "n_boundary_robust_states": int(
            ((robust <= eps) | (robust >= 1 - eps)).sum()
        ),
    }


def main() -> None:
    args = parse_args()
    unit_order, v_names, years, state_meta = load_production_layout(
        args.model_repo, args.data_path, args.msm_pop_path
    )
    source, draw_ids = load_bootstrap(args.bootstrap_path, args.expected_draws)
    values = build_dense_array(source, draw_ids, unit_order, v_names, years)

    missing_before = int((~np.isfinite(values)).sum())
    aggregate_audit = rebuild_aggregates_from_state_draws(
        values, unit_order, state_meta
    )
    fill_audit = fill_missing_states_from_divisions(values, unit_order, state_meta)
    metrics = validate_complete(values, unit_order, args.eps)
    metrics["missing_before_division_fill"] = missing_before
    metrics["filled_from_division"] = int(fill_audit["n_values_filled"].sum())
    metrics["aggregate_values_rebuilt"] = int(
        aggregate_audit["n_values_rebuilt"].sum()
    )
    metrics["missing_after_division_fill"] = int((~np.isfinite(values)).sum())

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_path,
        Xbars_samples=values,
        unit_order=np.asarray(unit_order),
        v_names=np.asarray(v_names),
        ts=years,
        seed=np.asarray(20261002),
        S=np.asarray(len(draw_ids)),
        source_draw_ids=np.asarray(draw_ids, dtype=int),
        value_source=np.asarray(
            "joint_respondent_bootstrap_factor_centered_logit_delta_state_aggregated"
        ),
        source_bootstrap_path=np.asarray(str(args.bootstrap_path.resolve())),
        data_path=np.asarray(str(args.data_path.resolve())),
        missing_state_rule=np.asarray("fill_from_matching_division_draw"),
        aggregate_rule=np.asarray(
            "state_n_weighted_from_originally_observed_state_draws"
        ),
        raking_weight_rule=np.asarray("fixed_final_raking_weights"),
        artifact_type=np.asarray("sem_input_draws"),
    )

    pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    ).to_csv(args.output_path.with_suffix(".audit.csv"), index=False)
    fill_audit.to_csv(
        args.output_path.with_suffix(".division_fill_audit.csv"), index=False
    )
    aggregate_audit.to_csv(
        args.output_path.with_suffix(".aggregate_rebuild_audit.csv"), index=False
    )

    print(f"Wrote SEM-compatible joint-bootstrap inputs: {args.output_path}")
    print(f"Shape: {values.shape}")
    print(f"Rebuilt {metrics['aggregate_values_rebuilt']} division/USA values")
    print(f"Filled {metrics['filled_from_division']} missing state values from divisions")
    print(f"Boundary values among robust states: {metrics['n_boundary_robust_states']}")


if __name__ == "__main__":
    main()
