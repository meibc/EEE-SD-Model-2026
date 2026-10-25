#!/usr/bin/env python3
from __future__ import annotations

"""State-year CDC surveillance and PrEP stocks for MSM (formerly `EEE Stocks 7.csv`).

Replaces three notebooks in EEE/EEE Feb 2025: `EEE CDC Clean.ipynb` (AtlasPlus
download -> CDC Data Raw), `AIDSVU PrEP Cleaning/PrEP Cleaning.ipynb` (AIDSVu
state PrEP files -> PrEP Data) and `EEE Stocks Final 2026 - Stocks 7.ipynb`.

Inputs (inputs/cdc_raw/, public): AtlasPlus state HIV indicators, AIDSVu state
PrEP users 2017-2023, and state MSM population estimates
(inputs/msm_state_population.csv). AtlasPlus indicators come from one tidy file,
atlasplus_combined.csv (October 2026 exports; see inputs/cdc_raw/SOURCES.md):
diagnoses, prevalence, AIDS classifications, viral suppression, care and linkage
(2017-2024); estimated incidence, estimated prevalence and knowledge of status
(2017-2022, the latest CDC estimates); PrEP coverage for men and by age group.
Years after 2024 are excluded (AtlasPlus marks 2025-2026 as preliminary).

PrEP stocks (MSM):
  PrEP           AIDSVu male PrEP users x 0.98 (MSM share of male users)
  PrEP Eligible  CDC-implied men with PrEP indications (persons prescribed /
                 coverage), x 2 (denominator scale); missing state-years are
                 imputed (within-state mean of available years, or the
                 cross-state mean eligible/MSM-population ratio when a state
                 has none); 2023 is the 2017-2022 mean
  Off PrEP       PrEP Eligible - PrEP
Surveillance stocks: MSM diagnoses, AIDS classifications, prevalence, viral
suppression and care sum the two MSM transmission categories; incidence,
estimated prevalence, knowledge of status and undiagnosed infections are
allocated from all-population totals by MSM shares; viral suppression and care
are imputed as bounded rates of estimated MSM prevalence where missing.
After the last CDC estimate year, estimated MSM prevalence is diagnosed MSM
prevalence times the state's ratio of estimated to diagnosed MSM prevalence in
the last estimate year (a fixed knowledge-of-status share); estimated incidence
is not extended.

National PrEP targets by age group (for the national raking, scripts/02):
  target_age = target_men x coverage_age / coverage_all
  target_men     national PrEP / PrEP Eligible from the state stocks above
  coverage_age   CDC persons prescribed / persons with indications by age (both
                 sexes, age rows of atlasplus_combined.csv, ages 16-54, states with data)
  coverage_all   all ages combined, weighting each age group by its AIDSVu users
Ages 55+ have no CDC indication estimate and are assumed to have the coverage
of ages 45-54; the 35+ group combines 35-44, 45-54 and 55+ by AIDSVu users.
2023 (no CDC age data): indications are the 2017-2022 mean by age, with AIDSVu
2023 users, as for the state eligible counts.

Outputs: results/00_cdc/cdc_stocks.csv, results/00_cdc/national_prep_age_targets.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "inputs/cdc_raw"

MSM_SHARE = 0.98  # share of male PrEP users who are MSM
ELIGIBLE_SCALE = 2.0  # PrEP-eligible denominator scale
P_CARE_INITIATED_GIVEN_KNOWLEDGE = 0.942
LAST_SURVEILLANCE_YEAR = 2024  # later AtlasPlus years are preliminary (12-month reporting delay)
ELIGIBLE_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
PREP_YEARS = ELIGIBLE_YEARS + [2023]
MSM_TCS = ["Male-to-male sexual contact", "Male-to-male sexual contact and injection drug use"]
KEYS = ["St_Abbr", "Year"]
NON_STATES = ["DC", "PR", "District of Columbia"]

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "American Samoa": "AS", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "District of Columbia", "Florida": "FL", "Georgia": "GA", "Guam": "GU",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Marshall Islands": "MH", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Micronesia": "FM", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Northern Marianas": "MP", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Palau": "PW", "Pennsylvania": "PA", "Puerto Rico": "PR", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virgin Islands": "VI", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "USA": "USA",
}


# ---- Raw sources ----------------------------------------------------------------

def load_atlasplus(path: Path) -> pd.DataFrame:
    """State-level AtlasPlus rows (ages 13+) through the last surveillance year, from the combined
    file (scripts/00_combine_atlasplus.py): numeric cases (estimates keep the point value), zeros as
    missing. PrEP coverage is the men's series; the both-sexes total and the age rows are not used here."""
    df = pd.read_csv(path, dtype={"fips": str})
    df = df[df["age_group"].eq("13+") & df["year"].le(LAST_SURVEILLANCE_YEAR)]
    prep = df["indicator"].eq("PrEP coverage and number of persons prescribed")
    df = df[~prep | df["sex"].eq("Male")]
    out = pd.DataFrame({
        "Indicator": df["indicator"], "Year": df["year"], "Geography": df["geography"],
        "Transmission Category": df["transmission_category"], "Cases": df["cases"].replace(0, np.nan),
        "Lower Bound": df["cases_lower"], "Upper Bound": df["cases_upper"], "Percent": df["percent"],
    })
    out["St_Abbr"] = out["Geography"].str.rstrip("^*").map(STATE_ABBR)  # strip AtlasPlus footnote markers
    return out.reset_index(drop=True)


def load_aidsvu_male_users(raw_dir: Path) -> pd.DataFrame:
    """Wide table of AIDSVu male PrEP users by state, one column per year."""
    out = None
    for year in PREP_YEARS:
        df = pd.read_excel(raw_dir / f"AIDSVu_State_PrEP_{year}_20240522.xlsx", header=3)
        df = df[["State Abbreviation", "Male PrEP Users"]].rename(columns={"Male PrEP Users": str(year)})
        out = df if out is None else out.merge(df, on="State Abbreviation", how="outer")
    return out


# ---- Helpers -----------------------------------------------------------------------

def series(base: pd.DataFrame, indicator: str, tc) -> pd.DataFrame:
    tcs = [tc] if isinstance(tc, str) else list(tc)
    df = base.loc[base["Indicator"].eq(indicator) & base["Transmission Category"].isin(tcs)]
    return df.groupby(KEYS, as_index=False)["Cases"].sum(min_count=1)


def label(df: pd.DataFrame, population: str, indicator: str) -> pd.DataFrame:
    return df.assign(Population=population, Indicator=indicator)[["St_Abbr", "Year", "Cases", "Population", "Indicator"]]


def subtract(a, b, *, out_indicator, population):
    out = (a.rename(columns={"Cases": "a"}).merge(b.rename(columns={"Cases": "b"}), on=KEYS, how="left")
           .fillna({"b": 0}).assign(Cases=lambda d: d["a"] - d["b"]))
    return label(out, population, out_indicator)


def allocate_to_msm(*, total_all, msm_part, all_part, out_indicator):
    """All-population total x (MSM part / all part), clipped to [0, 1]; missing if the MSM part is."""
    out = (total_all.rename(columns={"Cases": "Total"})
           .merge(msm_part[KEYS + ["Cases"]].rename(columns={"Cases": "MSM"}), on=KEYS, how="left")
           .merge(all_part[KEYS + ["Cases"]].rename(columns={"Cases": "ALL"}), on=KEYS, how="left")
           .fillna({"ALL": 0})
           .assign(Cases=lambda d: d["Total"] * (d["MSM"] / d["ALL"].replace(0, np.nan)).clip(0, 1)))
    return label(out, "MSM", out_indicator)


# ---- PrEP stocks -------------------------------------------------------------------

def prep_eligible(cdc: pd.DataFrame, msm_pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scaled PrEP-eligible MSM by state (wide, 2017-2023) and the imputation method per state-year."""
    cov = cdc.loc[cdc["Indicator"].eq("PrEP coverage and number of persons prescribed"), ["Year", "St_Abbr", "Cases", "Percent"]].copy()
    for c in ["Year", "Cases", "Percent"]:
        cov[c] = pd.to_numeric(cov[c], errors="coerce")
    cov["eligible"] = np.where(cov["Percent"].gt(0), cov["Cases"] / (cov["Percent"] / 100), np.nan)
    wide = cov.pivot_table(index="St_Abbr", columns="Year", values="eligible", aggfunc="mean").reindex(columns=ELIGIBLE_YEARS)

    df = msm_pop.join(wide, on="St_Abbr")
    raw_missing = df[ELIGIBLE_YEARS].isna()
    n_missing = raw_missing.sum(axis=1)
    complete, all_missing = n_missing.eq(0), n_missing.eq(len(ELIGIBLE_YEARS))
    part_missing = n_missing.between(1, len(ELIGIBLE_YEARS) - 1)

    mean_ratio = (df.loc[complete, ELIGIBLE_YEARS].mean(axis=1) / df.loc[complete, "msm_pop"]).mean()
    df.loc[all_missing, ELIGIBLE_YEARS] = np.repeat((df.loc[all_missing, "msm_pop"] * mean_ratio).to_numpy()[:, None],
                                                    len(ELIGIBLE_YEARS), axis=1)
    part = df.loc[part_missing, ELIGIBLE_YEARS]
    df.loc[part_missing, ELIGIBLE_YEARS] = part.T.fillna(part.mean(axis=1)).T
    df[2023] = df[ELIGIBLE_YEARS].mean(axis=1)
    df[PREP_YEARS] = df[PREP_YEARS] * ELIGIBLE_SCALE

    method = pd.DataFrame({"St_Abbr": df["St_Abbr"]})
    for y in ELIGIBLE_YEARS:
        method[y] = np.select([raw_missing[y] & all_missing, raw_missing[y] & part_missing],
                              ["prep_eligible_cross_state_mean_proportion", "prep_eligible_within_state_available_year_mean"],
                              default="none")
    method[2023] = "prep_eligible_2017_2022_mean"
    method = method.melt(id_vars="St_Abbr", var_name="Year", value_name="prep_elig_imputation_method")
    method["Year"] = method["Year"].astype(int)
    return df[["St_Abbr"] + PREP_YEARS], method


def prep_stocks(cdc, msm_pop, aidsvu) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible, method = prep_eligible(cdc, msm_pop)
    years = [str(y) for y in PREP_YEARS]
    on = aidsvu.rename(columns={"State Abbreviation": "St_Abbr"})
    on[years] = on[years].apply(pd.to_numeric, errors="coerce") * MSM_SHARE
    on = on[~on["St_Abbr"].isin(NON_STATES)]
    eligible = eligible[~eligible["St_Abbr"].isin(NON_STATES)].rename(columns={y: str(y) for y in PREP_YEARS})
    off = eligible.merge(on, on="St_Abbr", how="inner", suffixes=("_elig", "_on"))
    for y in years:
        off[y] = off[f"{y}_elig"] - off[f"{y}_on"]

    def long(df, indicator):
        out = df[["St_Abbr"] + years].melt(id_vars="St_Abbr", var_name="Year", value_name="Cases")
        out["Year"] = out["Year"].astype(int)
        out["Cases"] = pd.to_numeric(out["Cases"], errors="coerce")
        return label(out, "MSM", indicator)

    return pd.concat([long(on, "PrEP"), long(off, "Off PrEP"), long(eligible, "PrEP Eligible")], ignore_index=True), method


# ---- Surveillance stocks --------------------------------------------------------

def surveillance_stocks(cdc: pd.DataFrame) -> list[pd.DataFrame]:
    base = cdc.groupby(["St_Abbr", "Year", "Indicator", "Transmission Category"], as_index=False)["Cases"].sum(min_count=1)
    missing = set(MSM_TCS) - set(base["Transmission Category"].dropna())
    if missing:
        raise ValueError(f"Missing MSM transmission categories: {sorted(missing)}")

    msm_diag = label(series(base, "HIV diagnoses", MSM_TCS), "MSM", "HIV diagnoses")
    msm_care = label(series(base, "Receipt of HIV medical care", MSM_TCS), "MSM", "Receipt of HIV medical care")
    all_care = label(series(base, "Receipt of HIV medical care", "All"), "All", "Receipt of HIV medical care")
    msm_vs = label(series(base, "HIV viral suppression", MSM_TCS), "MSM", "HIV viral suppression")
    all_prev = label(series(base, "HIV prevalence", "All"), "All", "HIV prevalence")
    msm_prev = label(series(base, "HIV prevalence", MSM_TCS), "MSM", "HIV prevalence")
    all_knowledge = series(base, "Knowledge of Status", "All")
    estimated_prev = series(base, "Estimated HIV prevalence (undiagnosed and diagnosed)", "All")
    msm_estimated_prev = allocate_to_msm(total_all=estimated_prev, msm_part=msm_prev, all_part=all_prev,
                                         out_indicator="Estimated HIV prevalence (MSM)")
    # After the last estimate year: diagnosed MSM prevalence x the last estimated/diagnosed ratio.
    last = (msm_estimated_prev.dropna(subset=["Cases"]).sort_values("Year").groupby("St_Abbr").tail(1)
            .merge(msm_prev[KEYS + ["Cases"]].rename(columns={"Cases": "diagnosed"}), on=KEYS))
    ratio = last.assign(ratio=last["Cases"] / last["diagnosed"])[["St_Abbr", "Year", "ratio"]]
    later = (msm_prev.merge(ratio.rename(columns={"Year": "last_year"}), on="St_Abbr")
             .query("Year > last_year").assign(Cases=lambda d: d["Cases"] * d["ratio"]))
    msm_estimated_prev = pd.concat([msm_estimated_prev, label(later, "MSM", "Estimated HIV prevalence (MSM)")],
                                   ignore_index=True)

    msm_knowledge = allocate_to_msm(total_all=all_knowledge, msm_part=msm_care, all_part=all_care,
                                    out_indicator="Knowledge of Status")
    msm_ltc = (msm_knowledge.merge(msm_care[KEYS + ["Cases"]].rename(columns={"Cases": "care"}), on=KEYS, how="left")
               .fillna({"care": 0})
               .assign(Cases=lambda d: d["Cases"] * P_CARE_INITIATED_GIVEN_KNOWLEDGE - d["care"]))
    undiagnosed = subtract(estimated_prev, all_knowledge, out_indicator="Undiagnosed", population="All")
    return [
        label(series(base, "AIDS classifications", MSM_TCS), "MSM", "AIDS classifications"),
        msm_diag,
        subtract(series(base, "HIV diagnoses", MSM_TCS), series(base, "AIDS classifications", MSM_TCS),
                 out_indicator="New HIV diagnoses", population="MSM"),
        subtract(series(base, "HIV diagnoses", "All"), series(base, "AIDS classifications", "All"),
                 out_indicator="New diagnoses (not AIDS)", population="All"),
        allocate_to_msm(total_all=series(base, "Estimated HIV incidence", "All"), msm_part=msm_diag,
                        all_part=series(base, "HIV diagnoses", "All"), out_indicator="Estimated HIV incidence (MSM)"),
        msm_vs,
        msm_care,
        subtract(msm_care[KEYS + ["Cases"]], msm_vs[KEYS + ["Cases"]],
                 out_indicator="Receipt of HIV medical care, no VS", population="MSM"),
        all_care,
        msm_knowledge,
        label(msm_ltc, "MSM", "Lost to HIV care"),
        all_prev,
        msm_prev,
        msm_estimated_prev,
        allocate_to_msm(total_all=undiagnosed[KEYS + ["Cases"]], msm_part=msm_prev, all_part=all_prev,
                        out_indicator="Undiagnosed (MSM)"),
    ]


def impute_rate(df: pd.DataFrame, indicator: str, tag: str) -> None:
    """Fill missing MSM counts as a bounded rate of estimated MSM prevalence (in place).

    Rates are interpolated within state, then filled with the year median, then the
    overall median.
    """
    mask = df["Indicator"].eq(indicator) & df["Population"].eq("MSM")
    prev_mask = df["Indicator"].eq("Estimated HIV prevalence (MSM)") & df["Population"].eq("MSM")
    rate = df.loc[mask, KEYS + ["Cases"]].rename(columns={"Cases": "count"})
    rate["row_index"] = df.index[mask]
    rate = rate.merge(df.loc[prev_mask, KEYS + ["Cases"]].rename(columns={"Cases": "prev"}), on=KEYS, how="left")
    rate["raw_rate"] = np.where(rate["count"].notna() & rate["prev"].gt(0), rate["count"] / rate["prev"], np.nan)
    rate["raw_rate"] = rate["raw_rate"].clip(0.0, 1.0)
    rate = rate.sort_values(KEYS)
    rate["observed_state"] = rate.groupby("St_Abbr")["raw_rate"].transform("count").gt(0)
    rate["imputed_rate"] = (rate.groupby("St_Abbr")["raw_rate"]
                            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
                            .fillna(rate.groupby("Year")["raw_rate"].transform("median"))
                            .fillna(rate["raw_rate"].median(skipna=True))
                            .clip(0.0, 1.0))
    fill = rate["count"].isna() & rate["prev"].gt(0) & rate["imputed_rate"].notna()
    if fill.any():
        idx = rate.loc[fill, "row_index"].astype(int).to_numpy()
        df.loc[idx, "Cases"] = (rate.loc[fill, "imputed_rate"] * rate.loc[fill, "prev"]).to_numpy()
        df.loc[idx, "imputation_method"] = np.where(rate.loc[fill, "observed_state"],
                                                    f"rate_impute_{tag}_within_state", f"rate_impute_{tag}_year_median")


def recompute_care_no_vs(df: pd.DataFrame) -> pd.DataFrame:
    """Care without viral suppression from the (possibly imputed) care and VS counts, floored at 0."""
    care = (df.loc[df["Indicator"].eq("Receipt of HIV medical care") & df["Population"].eq("MSM"),
                   KEYS + ["Cases", "imputation_method"]]
            .rename(columns={"Cases": "care", "imputation_method": "care_method"}))
    vs = (df.loc[df["Indicator"].eq("HIV viral suppression") & df["Population"].eq("MSM"),
                 KEYS + ["Cases", "imputation_method"]]
          .rename(columns={"Cases": "vs", "imputation_method": "vs_method"}))
    out = care.merge(vs, on=KEYS, how="left")
    out["raw_difference"] = out["care"] - out["vs"]
    out["Cases"] = out["raw_difference"].clip(lower=0.0)
    out["Population"] = "MSM"
    out["Indicator"] = "Receipt of HIV medical care, no VS"
    out["Cases_raw"] = out["raw_difference"]
    care_imp = out["care_method"].str.startswith("rate_impute_care_", na=False)
    vs_imp = out["vs_method"].str.startswith("rate_impute_vs_", na=False)
    out.loc[care_imp | vs_imp, "Cases_raw"] = np.nan
    out["imputation_method"] = np.where(
        out["Cases"].isna(), "missing",
        np.select([care_imp & vs_imp, care_imp, vs_imp, out["raw_difference"].lt(0)],
                  ["derived_care_no_vs_from_imputed_care_and_vs", "derived_care_no_vs_from_imputed_care",
                   "derived_care_no_vs_from_imputed_vs", "derived_care_no_vs_clipped_zero"], default="none"))
    replace = df["Indicator"].eq("Receipt of HIV medical care, no VS") & df["Population"].eq("MSM")
    cols = ["St_Abbr", "Year", "Cases", "Population", "Indicator", "Cases_raw", "imputation_method"]
    return pd.concat([df.loc[~replace], out[cols]], ignore_index=True)


def national_age_targets(stocks: pd.DataFrame) -> pd.DataFrame:
    """National PrEP targets for the <=24, 25-34 and 35+ raking groups, 2017-2023."""
    cdc = pd.read_csv(RAW / "atlasplus_combined.csv")
    cdc = cdc[cdc["indicator"].eq("PrEP coverage and number of persons prescribed") & cdc["sex"].eq("Both sexes")
              & cdc["age_group"].ne("13+")].rename(columns={"year": "Year", "cases": "Cases", "percent": "Percent"})
    cdc = cdc[cdc["Cases"].notna() & cdc["Percent"].gt(0) & ~cdc["geography"].isin(["District of Columbia", "Puerto Rico"])]
    cdc["indicated"] = cdc["Cases"] / (cdc["Percent"] / 100)
    cdc["age"] = cdc["age_group"].replace({"13-24": "<=24"})
    sums = cdc.groupby(["Year", "age"])[["Cases", "indicated"]].sum()
    coverage = (sums["Cases"] / sums["indicated"]).unstack("age")  # Year x {<=24, 25-34, 35-44, 45-54}
    coverage["55+"] = coverage["45-54"]

    age_cols = {"Age LE 24 PrEP Users": "<=24", "Age 25-34 PrEP Users": "25-34", "Age 35-44 PrEP Users": "35-44",
                "Age 45-54 PrEP Users": "45-54", "Age 55+ PrEP Users": "55+"}
    users = {}
    for year in PREP_YEARS:
        df = pd.read_excel(RAW / f"AIDSVu_State_PrEP_{year}_20240522.xlsx", header=3)
        df = df[~df["State Abbreviation"].isin(NON_STATES)]
        users[year] = df[list(age_cols)].apply(pd.to_numeric, errors="coerce").sum().rename(index=age_cols)
    users = pd.DataFrame(users).T  # Year x age

    indicated = users.loc[ELIGIBLE_YEARS] / coverage.loc[ELIGIBLE_YEARS, users.columns]
    indicated.loc[2023] = indicated.mean()
    groups = {"<=24": ["<=24"], "25-34": ["25-34"], "35+": ["35-44", "45-54", "55+"]}
    cov_group = pd.DataFrame({g: users[a].sum(axis=1) / indicated[a].sum(axis=1) for g, a in groups.items()})
    cov_all = users.sum(axis=1) / indicated.sum(axis=1)

    prep = stocks[stocks["Population"].eq("MSM") & stocks["Indicator"].isin(["PrEP", "PrEP Eligible"])]
    men = prep.pivot_table(index="Year", columns="Indicator", values="Cases", aggfunc="sum")
    target_men = men["PrEP"] / men["PrEP Eligible"]

    rows = []
    for year in PREP_YEARS:
        for g, ages in groups.items():
            target = target_men[year] * cov_group.loc[year, g] / cov_all[year]
            eligible = ELIGIBLE_SCALE * indicated.loc[year, ages].sum()
            rows.append({"Year": year, "age_group": g, "prep_prop": target, "prep_eligible": eligible,
                         "prep_users": target * eligible, "coverage_ratio": cov_group.loc[year, g] / cov_all[year],
                         "target_men": target_men[year]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "results/00_cdc/cdc_stocks.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cdc = load_atlasplus(RAW / "atlasplus_combined.csv")
    msm_pop = (pd.read_csv(ROOT / "inputs/msm_state_population.csv")
               .rename(columns={"state": "St_Abbr", "mean": "msm_pop"})[["St_Abbr", "msm_pop"]])
    prep, eligible_method = prep_stocks(cdc, msm_pop, load_aidsvu_male_users(RAW))

    df = pd.concat([prep] + surveillance_stocks(cdc), ignore_index=True)
    df["Year"] = df["Year"].astype(int)
    df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
    df["Cases_raw"] = df["Cases"]
    df["imputation_method"] = np.where(df["Cases"].isna(), "missing", "none")
    last_estimate = int(cdc.loc[cdc["Indicator"].eq("Estimated HIV prevalence (undiagnosed and diagnosed)")
                                & cdc["Cases"].notna(), "Year"].max())
    extended = df["Indicator"].eq("Estimated HIV prevalence (MSM)") & df["Year"].gt(last_estimate) & df["Cases"].notna()
    df.loc[extended, "Cases_raw"] = np.nan
    df.loc[extended, "imputation_method"] = f"diagnosed_x_{last_estimate}_estimated_to_diagnosed_ratio"
    impute_rate(df, "HIV viral suppression", "vs")
    impute_rate(df, "Receipt of HIV medical care", "care")
    df = recompute_care_no_vs(df)

    df = df.sort_values(["St_Abbr", "Population", "Indicator", "Year"])
    df["was_imputed"] = df["Cases_raw"].isna() & df["Cases"].notna()
    df = df.merge(eligible_method, on=KEYS, how="left")
    imputed = (df["Indicator"].isin(["PrEP Eligible", "Off PrEP"]) & df["prep_elig_imputation_method"].notna()
               & df["prep_elig_imputation_method"].ne("none"))
    df.loc[imputed, "was_imputed"] = True
    df.loc[imputed, "Cases_raw"] = np.nan
    elig = imputed & df["Indicator"].eq("PrEP Eligible")
    off = imputed & df["Indicator"].eq("Off PrEP")
    df.loc[elig, "imputation_method"] = df.loc[elig, "prep_elig_imputation_method"]
    df.loc[off, "imputation_method"] = "derived_off_prep_from_" + df.loc[off, "prep_elig_imputation_method"]
    df = df.drop(columns="prep_elig_imputation_method")
    df["still_missing_after_imputation"] = df["Cases"].isna()
    df.to_csv(args.output, index=False)
    age = national_age_targets(df)
    age.to_csv(args.output.parent / "national_prep_age_targets.csv", index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print(age.pivot(index="Year", columns="age_group", values="prep_prop").round(4).to_string())


if __name__ == "__main__":
    main()
