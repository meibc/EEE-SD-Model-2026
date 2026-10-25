#!/usr/bin/env python3
from __future__ import annotations

"""State precision screen for the primary comparative analysis.

Inclusion depends only on AMIS input precision and completeness: the cell-level
Kish effective sample sizes of the 8 measures x 4 two-year periods for each
state. It is computed before any model is fitted.

Screens
-------
primary      all 32 cells; median n_eff >= 60; 25th percentile n_eff >= 40
median55     sensitivity: the same rule with a median cutoff of 55
strict       sensitivity: all 32 cells; 25th percentile >= 50; minimum >= 25
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

PERIODS = (2018, 2020, 2022, 2024)
VARIABLES = (
    "stigma_ahs",
    "stigma_gss",
    "stigma_family",
    "out_gid",
    "seehcp",
    "risk_behavior",
    "prep_used",
    "hivtest12",
)


def build_precision_screen(cells: pd.DataFrame, median_cutoff: float, q25_cutoff: float) -> pd.DataFrame:
    use = cells[
        cells["geography_type"].eq("state")
        & cells["year"].isin(PERIODS)
        & cells["variable"].isin(VARIABLES)
    ]
    expected_cells = len(PERIODS) * len(VARIABLES)

    rows = []
    for state, group in use.groupby("unit", sort=True):
        values = pd.to_numeric(group["n_eff"], errors="coerce").dropna().to_numpy(dtype=float)
        # Cells with a direct state estimate (division-fallback cells from scripts/03 have no n_eff).
        n_cells = int(group.loc[group["n_eff"].notna(), ["year", "variable"]].drop_duplicates().shape[0])
        complete = n_cells == expected_cells and values.size == expected_cells
        row = {
            "state": str(state),
            "n_expected_cells": expected_cells,
            "n_complete_cells": n_cells,
            "complete_32_cells": bool(complete),
            "min_cell_neff": float(np.min(values)) if values.size else np.nan,
            "q25_cell_neff": float(np.quantile(values, 0.25)) if values.size else np.nan,
            "median_cell_neff": float(np.median(values)) if values.size else np.nan,
            "q75_cell_neff": float(np.quantile(values, 0.75)) if values.size else np.nan,
            "max_cell_neff": float(np.max(values)) if values.size else np.nan,
        }
        row["primary_precision_screen"] = bool(
            complete and row["median_cell_neff"] >= median_cutoff and row["q25_cell_neff"] >= q25_cutoff
        )
        row["median55_precision_screen"] = bool(
            complete and row["median_cell_neff"] >= 55.0 and row["q25_cell_neff"] >= q25_cutoff
        )
        row["strict_precision_screen"] = bool(
            complete and row["q25_cell_neff"] >= 50.0 and row["min_cell_neff"] >= 25.0
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["primary_precision_screen", "median_cell_neff", "state"],
        ascending=[False, False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cells-path",
        type=Path,
        default=RESULTS / "02_inputs/state_bootstrap/cell_precision.csv",
        help="State-period-measure cells with Kish n_eff (scripts/03).",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "02_inputs/state_screen")
    parser.add_argument("--median-neff-cutoff", type=float, default=60.0)
    parser.add_argument("--q25-neff-cutoff", type=float, default=40.0)
    args = parser.parse_args()

    screen = build_precision_screen(
        pd.read_csv(args.cells_path),
        median_cutoff=args.median_neff_cutoff,
        q25_cutoff=args.q25_neff_cutoff,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen.to_csv(args.output_dir / "state_effective_n_screen.csv", index=False)
    for name in ("primary", "median55", "strict"):
        screen.loc[screen[f"{name}_precision_screen"]].to_csv(args.output_dir / f"{name}_states.csv", index=False)
        states = screen.loc[screen[f"{name}_precision_screen"], "state"].tolist()
        print(f"{name} ({len(states)}): {', '.join(states)}")


if __name__ == "__main__":
    main()
