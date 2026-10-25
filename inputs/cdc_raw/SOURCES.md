# CDC and AIDSVu inputs

Public data used by `scripts/00_build_cdc_stocks.py` (state-year surveillance and PrEP stocks for MSM)
and, through its outputs, by the national weights and the epidemiological model.

## atlasplus_combined.csv

CDC NCHHSTP AtlasPlus (https://gis.cdc.gov/grasp/nchhstpatlas/tables.html), table exports combined
into one tidy file: one row per indicator x year x state x sex x age group x transmission category,
with numeric cases (estimates split into value and 95% bounds; `cases_unstable` marks estimates
AtlasPlus flags as statistically unstable), rate, percent, the status of non-numeric entries
(`value_status`: suppressed, not available, not applicable), and the export each row came from.

All exports: all states and DC, all races/ethnicities. Years are as available (2025-2026 are
preliminary and excluded by the builder).

| Exports (downloaded) | Indicators | Sex | Ages | Transmission category |
|---|---|---|---|---|
| October 22, 2026 | HIV diagnoses, HIV prevalence, AIDS classifications, viral suppression, receipt of care, linkage to care | Male | 13+ | Male-to-male sexual contact; male-to-male sexual contact and injection drug use |
| October 22, 2026 | HIV diagnoses, HIV prevalence, receipt of care, linkage to care | Both sexes | 13+ | All |
| October 25, 2026 | Estimated HIV incidence, estimated HIV prevalence (undiagnosed and diagnosed), knowledge of status (2017-2022) | Both sexes | 13+ | All |
| October 25, 2026 | PrEP coverage and number of persons prescribed (2017-2023) | Male | 13+ | All |
| October 25, 2026 | PrEP coverage and number of persons prescribed (2017-2023) | Both sexes | 13-24, 25-34, 35-44, 45-54 | All |

The raw exports and the script that combined them (`combine_atlasplus.py`) are kept outside the
repository in the project archive (`EEE-SD-Model-2026-archive/inputs_cdc_raw_atlasplus_20261025/`).
The combined file reproduces the stocks built from the earlier downloads exactly (October 25, 2026
check), apart from empty national placeholder rows that the older download carried.

## AIDSVu_State_PrEP_<year>_20240522.xlsx

AIDSVu (https://aidsvu.org) state PrEP data, 2017-2023, downloaded May 22, 2024: PrEP users by
state, sex and age group. Used for male PrEP users (state PrEP stocks) and users by age (national
PrEP targets).

## Other inputs (inputs/)

- `msm_state_population.csv`: estimated MSM population by state.
- `state_msm_margins.csv`, `national_msm_margins.csv`: demographic margins for the state and
  national raking (scripts/01, scripts/02).
