#!/usr/bin/env python3
"""Compare the previous Beta/zero-drift and new bootstrap/state-drift population fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD_NATIONAL = Path(
    "/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/output/"
    "priority_population_sem_mc_annual_smoothed_n500/"
    "priority_population_mc_J_draws_n500_seed123_drift_zero_tau0p20.csv"
)
OLD_CA = ROOT / (
    "outputs/ca_hispanic_latino_sem_mc_corrected/"
    "priority_population_mc_J_draws_n500_seed913_drift_zero_tau0p20.csv"
)
NEW = ROOT / (
    "output/population_joint_bootstrap_n500/fits_state_drift/"
    "population_J_draws_n500_state_drift.csv"
)
NEW_ZERO = ROOT / (
    "output/population_joint_bootstrap_n500/fits_zero_drift/"
    "population_J_draws_n500_zero_drift_tau0p20.csv"
)
OUT = ROOT / "output/population_joint_bootstrap_n500/old_vs_new_pathway_comparison.csv"
OUT_DRIFT = ROOT / "output/population_joint_bootstrap_n500/state_drift_vs_zero_drift_pathway_comparison.csv"

PATHWAYS = [
    ("stigma_ahs", "out_gid", "AHS -> disclosure"),
    ("stigma_ahs", "prep_used", "AHS -> PrEP use"),
    ("out_gid", "prep_used", "Disclosure -> PrEP use"),
    ("out_gid", "hivtest12", "Disclosure -> HIV testing"),
    ("seehcp", "stigma_ahs", "Saw HCP -> AHS"),
    ("seehcp", "hivtest12", "Saw HCP -> HIV testing"),
]


def summarize(path: Path, version: str, populations: list[str] | None = None) -> pd.DataFrame:
    d = pd.read_csv(path)
    if populations is not None:
        d = d[d["population_id"].isin(populations)].copy()
    labels = d[["population_id", "population_label"]].drop_duplicates()
    rows = []
    for pop in labels.itertuples(index=False):
        dp = d[d["population_id"].eq(pop.population_id)]
        draw_ids = np.sort(dp["draw"].unique())
        for source, target, label in PATHWAYS:
            values = (
                pd.DataFrame({"draw": draw_ids})
                .merge(
                    dp[(dp["source"].eq(source)) & (dp["target"].eq(target))][["draw", "J"]],
                    on="draw", how="left",
                )["J"]
                .fillna(0.0)
                .to_numpy()
            )
            rows.append({
                "version": version,
                "population_id": pop.population_id,
                "population_label": pop.population_label,
                "pathway": label,
                "p25": np.quantile(values, 0.25),
                "p50": np.quantile(values, 0.50),
                "p75": np.quantile(values, 0.75),
                "p025": np.quantile(values, 0.025),
                "p975": np.quantile(values, 0.975),
            })
    return pd.DataFrame(rows)


def main():
    national_ids = ["black_nh", "hispanic_latino", "white_nh", "age_under_24", "age_25_34", "age_35_plus"]
    old = pd.concat([
        summarize(OLD_NATIONAL, "old_beta_zero_drift", national_ids),
        summarize(OLD_CA, "old_beta_zero_drift", ["ca_hispanic_latino"]),
    ], ignore_index=True)
    new = summarize(NEW, "new_joint_bootstrap_state_drift", national_ids + ["ca_hispanic_latino"])
    wide = old.merge(
        new,
        on=["population_id", "population_label", "pathway"],
        suffixes=("_old", "_new"),
    )
    wide["median_change"] = wide["p50_new"] - wide["p50_old"]
    wide["iqr_width_old"] = wide["p75_old"] - wide["p25_old"]
    wide["iqr_width_new"] = wide["p75_new"] - wide["p25_new"]
    wide["iqr_width_ratio_new_to_old"] = np.where(
        wide["iqr_width_old"] > 0,
        wide["iqr_width_new"] / wide["iqr_width_old"],
        np.nan,
    )
    wide.to_csv(OUT, index=False)
    print(OUT)
    print(wide.groupby("pathway").agg(
        median_absolute_shift=("median_change", lambda x: np.median(np.abs(x))),
        median_iqr_ratio=("iqr_width_ratio_new_to_old", "median"),
    ).to_string())

    zero = summarize(NEW_ZERO, "new_joint_bootstrap_zero_drift", national_ids + ["ca_hispanic_latino"])
    drift = new.merge(
        zero,
        on=["population_id", "population_label", "pathway"],
        suffixes=("_state_drift", "_zero_drift"),
    )
    drift["median_change_state_minus_zero"] = drift["p50_state_drift"] - drift["p50_zero_drift"]
    drift.to_csv(OUT_DRIFT, index=False)
    print(OUT_DRIFT)
    print(drift.groupby("pathway").agg(
        median_absolute_shift=("median_change_state_minus_zero", lambda x: np.median(np.abs(x))),
    ).to_string())


if __name__ == "__main__":
    main()
