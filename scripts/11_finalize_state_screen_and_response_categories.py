#!/usr/bin/env python3
from __future__ import annotations

"""Finalize the state precision screen and descriptive response categories.

The state screen is based only on survey-input precision.  It is deliberately
separate from model fit and scenario response so that inclusion cannot be
conditioned on whether a state produces an interesting result.

The response categories are descriptive summaries of continuous scenario
effects, not latent classes or statistical clusters.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PERIODS = ("2017-2018", "2019-2020", "2021-2022", "2023-2024")
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


def build_precision_screen(
    summary_path: Path,
    median_neff_cutoff: float,
    q25_neff_cutoff: float,
) -> pd.DataFrame:
    data = pd.read_csv(summary_path)
    needed = {"state_calc", "year_group", "variable", "n_eff"}
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {summary_path}: {sorted(missing)}")

    use = data[
        data["year_group"].isin(PERIODS) & data["variable"].isin(VARIABLES)
    ].copy()
    # The summary also contains USA and Census-division aggregate rows.  The
    # primary comparison is state-only, identified by two-letter postal codes.
    use = use[use["state_calc"].astype(str).str.fullmatch(r"[A-Z]{2}")].copy()
    use["n_eff"] = pd.to_numeric(use["n_eff"], errors="coerce")
    expected_cells = len(PERIODS) * len(VARIABLES)

    rows = []
    for state, group in use.groupby("state_calc", sort=True):
        values = group["n_eff"].dropna().to_numpy(dtype=float)
        n_cells = int(group[["year_group", "variable"]].drop_duplicates().shape[0])
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
            complete
            and row["median_cell_neff"] >= median_neff_cutoff
            and row["q25_cell_neff"] >= q25_neff_cutoff
        )
        # A stricter sensitivity screen used to assess dependence on the primary cutoff.
        row["strict_precision_screen"] = bool(
            complete and row["q25_cell_neff"] >= 50.0 and row["min_cell_neff"] >= 25.0
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["primary_precision_screen", "median_cell_neff", "state"],
        ascending=[False, False, True],
    )


def summarize_response_categories(
    scenario_draws_path: Path,
    selected_states: list[str],
    response_cutoff_pct: float,
    ahs_share_cutoff: float,
) -> pd.DataFrame:
    data = pd.read_csv(scenario_draws_path)
    required = {"unit_id", "scenario", "variable", "pct_change"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {scenario_draws_path}: {sorted(missing)}")

    aliases = {
        "ahs": "s1_reduce_ahs",
        "combined": "s7_combined_stigma_package",
    }
    use = data[
        data["unit_id"].isin(selected_states)
        & data["variable"].eq("incidence")
        & data["scenario"].isin(aliases.values())
    ].copy()
    if use.empty:
        raise ValueError("No selected-state incidence draws found in scenario file")

    summary = (
        use.groupby(["unit_id", "scenario"], as_index=False)["pct_change"]
        .agg(
            p025=lambda x: np.nanquantile(x, 0.025),
            p25=lambda x: np.nanquantile(x, 0.25),
            p50=lambda x: np.nanquantile(x, 0.50),
            p75=lambda x: np.nanquantile(x, 0.75),
            p975=lambda x: np.nanquantile(x, 0.975),
        )
    )

    med = summary.pivot(index="unit_id", columns="scenario", values="p50")
    missing_scenarios = [code for code in aliases.values() if code not in med.columns]
    if missing_scenarios:
        raise ValueError(f"Missing scenario medians: {missing_scenarios}")

    out = pd.DataFrame(index=med.index)
    out["combined_incidence_change_pct"] = med[aliases["combined"]]
    out["ahs_incidence_change_pct"] = med[aliases["ahs"]]
    out["combined_reduction_magnitude_pct"] = -out["combined_incidence_change_pct"]
    out["ahs_reduction_magnitude_pct"] = -out["ahs_incidence_change_pct"]
    out["increment_beyond_ahs_pp"] = (
        out["combined_reduction_magnitude_pct"] - out["ahs_reduction_magnitude_pct"]
    )
    out["ahs_share_of_combined"] = np.where(
        out["combined_reduction_magnitude_pct"] > 0,
        out["ahs_reduction_magnitude_pct"] / out["combined_reduction_magnitude_pct"],
        np.nan,
    )

    def classify(row: pd.Series) -> str:
        if row["combined_reduction_magnitude_pct"] < response_cutoff_pct:
            return "Limited aggregate responsiveness"
        if row["ahs_share_of_combined"] >= ahs_share_cutoff:
            return "AHS-centered responsiveness"
        return "Broader stigma responsiveness"

    out["response_category"] = out.apply(classify, axis=1)
    out["response_cutoff_pct"] = float(response_cutoff_pct)
    out["ahs_share_cutoff"] = float(ahs_share_cutoff)
    out = out.reset_index(names="state")

    interval = summary.pivot(index="unit_id", columns="scenario", values=["p25", "p75", "p025", "p975"])
    for stat in ("p25", "p75", "p025", "p975"):
        for short, code in aliases.items():
            out[f"{short}_{stat}_pct_change"] = out["state"].map(interval[(stat, code)])

    return out.sort_values("combined_reduction_magnitude_pct", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-path", type=Path, default=Path("amis_summary_stats_v4.csv"))
    parser.add_argument(
        "--scenario-draws-path",
        type=Path,
        default=Path(
            "/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/"
            "output/sem_scenarios_seehcpdiag_ref0p5_riskweakdrift_500/"
            "scenario_endpoint_epi_draws.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/final_state_screen_and_response_categories"),
    )
    parser.add_argument("--median-neff-cutoff", type=float, default=60.0)
    parser.add_argument("--q25-neff-cutoff", type=float, default=40.0)
    parser.add_argument("--response-cutoff-pct", type=float, default=10.0)
    parser.add_argument("--ahs-share-cutoff", type=float, default=0.80)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen = build_precision_screen(
        args.summary_path,
        median_neff_cutoff=args.median_neff_cutoff,
        q25_neff_cutoff=args.q25_neff_cutoff,
    )
    selected = screen.loc[screen["primary_precision_screen"], "state"].tolist()
    categories = summarize_response_categories(
        args.scenario_draws_path,
        selected_states=selected,
        response_cutoff_pct=args.response_cutoff_pct,
        ahs_share_cutoff=args.ahs_share_cutoff,
    )

    screen.to_csv(args.output_dir / "state_effective_n_screen.csv", index=False)
    categories.to_csv(args.output_dir / "state_response_categories.csv", index=False)

    sensitivity_rows = []
    for response_cutoff in (7.5, 10.0, 12.5):
        for share_cutoff in (0.75, 0.80, 0.85):
            trial = categories.copy()
            trial["response_cutoff_pct"] = response_cutoff
            trial["ahs_share_cutoff"] = share_cutoff
            trial["response_category"] = np.where(
                trial["combined_reduction_magnitude_pct"] < response_cutoff,
                "Limited aggregate responsiveness",
                np.where(
                    trial["ahs_share_of_combined"] >= share_cutoff,
                    "AHS-centered responsiveness",
                    "Broader stigma responsiveness",
                ),
            )
            sensitivity_rows.append(
                trial[
                    [
                        "state",
                        "response_cutoff_pct",
                        "ahs_share_cutoff",
                        "response_category",
                    ]
                ]
            )
    sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    sensitivity.to_csv(
        args.output_dir / "response_category_threshold_sensitivity.csv", index=False
    )
    stability = (
        sensitivity.groupby("state")["response_category"]
        .agg(
            n_distinct_categories="nunique",
            modal_category=lambda x: x.value_counts().index[0],
            modal_fraction=lambda x: x.value_counts(normalize=True).iloc[0],
        )
        .reset_index()
    )
    stability.to_csv(args.output_dir / "response_category_threshold_stability.csv", index=False)
    screen.loc[screen["primary_precision_screen"]].to_csv(
        args.output_dir / "primary_states.csv", index=False
    )
    screen.loc[screen["strict_precision_screen"]].to_csv(
        args.output_dir / "strict_sensitivity_states.csv", index=False
    )

    notes = [
        "State inclusion is based only on AMIS input precision and completeness.",
        f"Primary: all 32 cells complete; median cell n_eff >= {args.median_neff_cutoff:g}; 25th percentile cell n_eff >= {args.q25_neff_cutoff:g}.",
        "Strict sensitivity: all 32 cells complete; 25th percentile cell n_eff >= 50; minimum cell n_eff >= 25.",
        f"Limited response: combined-package median incidence reduction < {args.response_cutoff_pct:g}%.",
        f"Among responsive states, AHS-centered means AHS-only accounts for >= {100*args.ahs_share_cutoff:g}% of the combined response; otherwise broader stigma responsive.",
        "Response categories are descriptive summaries of continuous effects, not statistical clusters.",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(notes) + "\n")

    print(f"Primary states ({len(selected)}): {', '.join(selected)}")
    for category, group in categories.groupby("response_category", sort=False):
        print(f"{category}: {', '.join(group['state'])}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
