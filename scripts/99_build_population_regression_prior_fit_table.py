#!/usr/bin/env python3
"""Build a publication audit table linking regressions, priors, and fitted pathways."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/Users/meibinchen/Documents/Codex/2026-08-13/d/outputs")
MODEL_ROOT = Path("/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/output")

REGRESSION_BOOK = (
    SOURCE_ROOT
    / "priority_sem_inputs_age_prep_primary_six_marginal"
    / "EEE Regression Analysis - Six Marginal Priority Populations core18 age-prep-primary.xlsx"
)
PRIOR_ROOT = SOURCE_ROOT / "priority_sem_inputs_age_prep_primary_six_marginal_annual_smoothed_logit3yr"
NATIONAL_DRAWS = (
    ROOT
    / "output/population_joint_bootstrap_n500/fits_state_drift"
    / "population_J_draws_n500_state_drift.csv"
)
CA_PRIORS = ROOT / "outputs/ca_hispanic_latino_sem_inputs_corrected/ca_hispanic_latino_structural_priors.csv"
CA_DRAWS = (
    ROOT
    / "output/population_joint_bootstrap_n500/fits_state_drift"
    / "population_J_draws_n500_state_drift.csv"
)
OUT_DIR = ROOT / "output/supplementary_tables"
OUT_XLSX = OUT_DIR / "table_S_population_regression_prior_dynamic_estimates.xlsx"
OUT_CSV = OUT_DIR / "table_S_population_regression_prior_dynamic_estimates.csv"


POPULATION_ORDER = [
    "black_nh",
    "hispanic_latino",
    "white_nh",
    "age_under_24",
    "age_25_34",
    "age_35_plus",
]


def load_priors() -> pd.DataFrame:
    frames = []
    for pop in POPULATION_ORDER:
        path = PRIOR_ROOT / f"{pop}_sem_inputs" / f"{pop}_structural_priors.csv"
        d = pd.read_csv(path)
        d["population_id"] = pop
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def zero_inclusive_dynamic_summary(
    draws_path: Path,
    populations: pd.DataFrame,
    pathways: pd.DataFrame,
) -> pd.DataFrame:
    draws = pd.read_csv(draws_path)
    draw_keys = draws[["population_id", "draw"]].drop_duplicates()
    grid = (
        populations[["population_id", "population_label"]]
        .drop_duplicates()
        .merge(draw_keys, on="population_id", how="inner")
        .assign(_key=1)
        .merge(pathways[["source", "target"]].drop_duplicates().assign(_key=1), on="_key")
        .drop(columns="_key")
    )
    merged = grid.merge(
        draws[["population_id", "draw", "source", "target", "J"]],
        on=["population_id", "draw", "source", "target"],
        how="left",
    )
    merged["J"] = merged["J"].fillna(0.0)
    return (
        merged.groupby(["population_id", "population_label", "source", "target"], as_index=False)
        .agg(
            dynamic_p025=("J", lambda x: np.quantile(x, 0.025)),
            dynamic_p25=("J", lambda x: np.quantile(x, 0.25)),
            dynamic_median=("J", "median"),
            dynamic_p75=("J", lambda x: np.quantile(x, 0.75)),
            dynamic_p975=("J", lambda x: np.quantile(x, 0.975)),
            dynamic_zero_fraction=("J", lambda x: np.mean(np.isclose(x, 0.0))),
            dynamic_n_draws=("J", "size"),
        )
    )


def make_national_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    regression = pd.read_excel(REGRESSION_BOOK, sheet_name="population_key_terms")
    specs = pd.read_excel(REGRESSION_BOOK, sheet_name="model_specifications")
    priors = load_priors()
    labels = regression[["population_id", "population_label"]].drop_duplicates()
    pathways = priors[["source", "target"]].drop_duplicates()
    dynamic = zero_inclusive_dynamic_summary(NATIONAL_DRAWS, labels, pathways)

    prior_cols = [
        "population_id", "source", "target", "sign", "prior_mean", "prior_tau",
        "evidence_tier", "evidence_note",
    ]
    table = regression.rename(columns={"term": "source", "outcome": "target"}).merge(
        priors[prior_cols], on=["population_id", "source", "target"], how="left", validate="one_to_one"
    )
    table = table.merge(
        dynamic,
        on=["population_id", "population_label", "source", "target"],
        how="left",
        validate="one_to_one",
    )
    table["prior_direction"] = table["sign"].map({1: "positive", -1: "negative"})
    table["regression_prior_sign_agreement"] = np.where(
        table["direction"].eq(table["prior_direction"]), "agrees", "does not agree"
    )
    table["prior_role"] = np.where(
        table["evidence_tier"].eq("strong"),
        "direction agrees with prespecified pathway and regression evidence supports strong tier",
        "weak theory-informed tier; regression magnitude was not transferred to prior mean",
    )
    columns = [
        "relationship_id", "relationship", "population_id", "population_label",
        "source", "target", "estimate", "std.error", "conf_low", "conf_high",
        "odds_ratio", "conf_low_or", "conf_high_or", "p.value", "n_complete",
        "formula", "direction", "significant_0_05", "prior_direction",
        "regression_prior_sign_agreement", "evidence_tier", "prior_mean", "prior_tau",
        "prior_role", "evidence_note", "dynamic_p025", "dynamic_p25",
        "dynamic_median", "dynamic_p75", "dynamic_p975", "dynamic_zero_fraction",
        "dynamic_n_draws",
    ]
    return table[columns].sort_values(["relationship_id", "population_id"]), specs


def make_california_table(national: pd.DataFrame) -> pd.DataFrame:
    priors = pd.read_csv(CA_PRIORS)
    labels = pd.DataFrame({
        "population_id": ["ca_hispanic_latino"],
        "population_label": ["California Hispanic/Latino"],
    })
    dynamic = zero_inclusive_dynamic_summary(CA_DRAWS, labels, priors[["source", "target"]])
    inherited = national[national["population_id"].eq("hispanic_latino")].copy()
    keep = [
        "relationship_id", "relationship", "source", "target", "estimate", "std.error",
        "conf_low", "conf_high", "odds_ratio", "conf_low_or", "conf_high_or", "p.value",
        "n_complete", "formula", "direction", "significant_0_05",
    ]
    table = priors.merge(inherited[keep], on=["source", "target"], how="left", validate="one_to_one")
    table["population_id"] = "ca_hispanic_latino"
    table["population_label"] = "California Hispanic/Latino"
    table["regression_source"] = "National Hispanic/Latino regression (inherited prior evidence)"
    table = table.merge(
        dynamic,
        on=["population_id", "population_label", "source", "target"],
        how="left",
        validate="one_to_one",
    )
    table["prior_direction"] = table["sign"].map({1: "positive", -1: "negative"})
    table["regression_prior_sign_agreement"] = np.where(
        table["direction"].eq(table["prior_direction"]), "agrees", "does not agree"
    )
    return table.sort_values("relationship_id")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    national, specs = make_national_table()
    california = make_california_table(national)
    national.to_csv(OUT_CSV, index=False)

    notes = pd.DataFrame({
        "item": [
            "Regression interpretation", "Prior construction", "Forced-weak pathways",
            "Dynamic intervals", "Zero handling", "California bridge",
        ],
        "description": [
            "Survey-weighted quasibinomial regressions supplied cross-sectional supporting evidence; they were not interpreted as causal or longitudinal pathway estimates.",
            "Regression magnitudes were not copied into prior means. Strong and weak tiers used fixed centers of +/-0.15 and +/-0.04 and scales of 0.30 and 0.70, respectively.",
            "Disclosure -> AHS, saw HCP -> AHS, AHS -> saw HCP, AHS -> PrEP use, and AHS -> HIV testing were retained in the weak tier regardless of regression significance.",
            "Fitted dynamic summaries use 500 shared respondent-bootstrap refits. National trajectories were smoothed within each replicate; all models used the primary state drift specification.",
            "Constrained estimates omitted from sparse draw files were restored as exact zeros before calculating quantiles.",
            "California used the national Hispanic/Latino prior evidence but California-specific two-year trajectories and covariance; its regression columns therefore describe the inherited national evidence.",
        ],
    })
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        national.to_excel(writer, sheet_name="national_groups", index=False)
        california.to_excel(writer, sheet_name="california_bridge", index=False)
        specs.to_excel(writer, sheet_name="regression_models", index=False)
        notes.to_excel(writer, sheet_name="notes", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    print(OUT_XLSX)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
