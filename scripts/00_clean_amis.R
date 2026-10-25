# Construct the analytic AMIS respondent file from annual restricted-use
# Stata files.
#
# DATA STRUCTURE
# AMIS is a repeated cross-sectional survey; respondents are not followed
# longitudinally. Survey cycles included here are 2017, 2018, 2019, 2021,
# 2022, 2023, and 2024. AMIS 2015–2016 are excluded because their stigma items
# do not ask whether the experience occurred in the last 6 months. AMIS 2020 is
# excluded because COVID-19 disrupted recruitment and fielding.
#
# Survey year denotes the AMIS cycle represented by each raw file and differs
# from the file-submission year. For the primary state analysis, cycles are
# assigned to four period labels:
#   2017–2018: AMIS 2017 and 2018
#   2019–2020: AMIS 2019 only
#   2021–2022: AMIS 2021 and 2022
#   2023–2024: AMIS 2023 and 2024
#
# The 2019–2020 label is retained for alignment with the common two-year
# period structure; it does not imply that AMIS 2020 observations were used.
#
# ELIGIBILITY AND HARMONIZATION
# The analytic file retains respondents who:
#   1. reported HIV status and did not report a positive HIV diagnosis;
#   2. were aged 18–99 years; and
#   3. resided in one of the 50 U.S. states.
#
# Race/ethnicity, education, insurance, and other respondent characteristics
# are harmonized across cycle-specific variable definitions. The 2018
# education coding incorporates the youth-specific education item.
#
# MEASURE CONSTRUCTION
# Eight model measures are constructed:
#   anticipated healthcare stigma, general social stigma, family stigma,
#   disclosure to a healthcare provider, having seen a healthcare provider
#   in the past year, past-year PrEP use, HIV testing frequency (tests in the
#   past two years, capped at four, scaled to 0-1; never-testers 0), and
#   HIV-related risk behavior.
#
# General coding principles:
#   - legitimate questionnaire skips are coded as the implied substantive
#     value where supported by the survey logic;
#   - refusals, "don't know" responses, and items not administered remain
#     missing unless the survey logic supports an implied value;
#   - for stigma checklists, a construct is coded 1 if any available component
#     indicates a recent experience, 0 if no component indicates a recent
#     experience and at least one component contains a substantive negative
#     response, and missing if all components are unavailable or non-substantive.
#
# Measure-specific coding rules are implemented below. AMIS 2024 did not
# administer the sexual-violence stigma items, so its general-social-stigma
# composite is calculated from the four available components.
#
# OUTPUTS
#   results/00_amis/amis_analytic.csv.gz
#     Respondent-level analytic file used by the weighting and bootstrap stages.
#
#   results/00_amis/exclusions.csv
#     Sequential counts documenting application of the eligibility criteria.
#
# Run from the repository root:
#   Rscript scripts/00_clean_amis.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(haven)
})

raw_dir <- "inputs/restricted/amis_raw"
output_dir <- "results/00_amis"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# read in full AMIS data 
cycles <- c(
  amis_17 = "sitq2017_analysis_noid.dta",
  amis_18 = "sitq2018_analysis_noid.dta",
  amis_19 = "sitq2019_analysis_noid.dta",
  amis_21 = "sitq2021_analysis_noid.dta",
  amis_22 = "sitq2022_analysis_fulldata_noid.dta",
  amis_23 = "sitq2023_analysis_final_noid.dta",
  amis_24 = "sitq2024_analysis_final_NOID.dta"
)

# pull out relevant data
raw_vars <- c(
  "vdatesub", "state_calc", "_censdiv", "region", "_hivstat", "age", "_raceomb", "_educat",
  "hleducat_le24", "_inscat", "_nchs_rural2013", "homeless_p12m", "illicit", "heard_prep", "prep_ever", "prep_used",
  "prep_current", "seehcp", "out_gi", "out_gid", "evertest", "test2yrs", "_lasthivtest12", "_mo_lasthivtest",
  "m_m1uas", "m_muauhs", "m_muahp", "m_mm1hsk", "m_m1hst",
  paste0("stigma_", c("a1", "a2", "b1", "b2", "c1", "c2", "c3", "d1", "d2", "d3", "d4"))
)

# read all data cycles, keeping only the variables of interest and adding a survey name and row number
read_cycle <- function(survey, file) {
  path <- file.path(raw_dir, file)
  available <- names(read_dta(path, n_max = 0))
  df <- read_dta(path, col_select = all_of(available[tolower(available) %in% raw_vars]))
  names(df) <- tolower(names(df))
  df %>%
    zap_labels() %>%
    mutate(survey = survey, row_in_file = row_number(), .before = 1)
}

# Stigma composite: 1 if any component is 1 (yes, last 6 months); 0 if any
# component is 0 or 2 (no, or not in the last 6 months); otherwise missing.
recent_any <- function(...) {
  items <- cbind(...)
  any_recent <- rowSums(items == 1, na.rm = TRUE) > 0
  any_answer <- rowSums(items == 0 | items == 2, na.rm = TRUE) > 0
  case_when(any_recent ~ 1, any_answer ~ 0)
}

# Violence attributed to having sex with men, both in the last 6 months. The
# attribution item is asked only after violence was reported. Unknown is returned
# as 7 (non-response) so the composite treats it like an unanswered item.
recent_attributed <- function(event, attributed) {
  case_when(
    event %in% c(0, 2) ~ 0,
    event == 1 & attributed == 1 ~ 1,
    event == 1 & attributed %in% c(0, 2) ~ 0,
    TRUE ~ 7
  )
}

raw <- imap_dfr(cycles, ~ read_cycle(.y, .x))

# Race/ethnicity: AMIS 2017-2019 codes 1 AI/AN, 2 Asian/NHPI, 3 Black, 4 Hispanic,
# 5 White, 6 Other/multiple; AMIS 2021-2023 codes 1 Black, 2 Hispanic, 3 AI/AN,
# 4 Asian/NHPI, 5 MENA, 6 Other, 7 White, 8 Unknown. "other" matches the state
# raking target (all other races).
recoded <- raw %>%
  mutate(
    vdatesub = as.Date(vdatesub),
    submission_year = as.integer(format(vdatesub, "%Y")),
    survey_year = 2000L + as.integer(sub("amis_", "", survey)),
    year_group = case_when(
      survey_year %in% 2017:2018 ~ "2017-2018", survey_year %in% 2019:2020 ~ "2019-2020",
      survey_year %in% 2021:2022 ~ "2021-2022", survey_year %in% 2023:2024 ~ "2023-2024"
    ),
    rowID = paste(survey, row_in_file, sep = "_"),
    rdrace = if_else(
      survey %in% c("amis_17", "amis_18", "amis_19"),
      case_when(`_raceomb` == 3 ~ "black", `_raceomb` == 4 ~ "hisp", `_raceomb` == 5 ~ "white",
                `_raceomb` %in% c(1, 2, 6) ~ "other"),
      case_when(`_raceomb` == 1 ~ "black", `_raceomb` == 2 ~ "hisp", `_raceomb` == 7 ~ "white",
                `_raceomb` %in% 3:6 ~ "other")
    ),
    agecat = cut(age, c(17, 24, 34, 44, 54, Inf), labels = c("18_24", "25_34", "35_44", "45_54", "55_up")),
    # Respondents aged 18-24 in AMIS 2018 have only the youth education item;
    # its codes map to _educat as observed in AMIS 2019, where both exist.
    school4 = coalesce(
      c("lths", "hs", "college", "grad")[`_educat`],
      case_when(hleducat_le24 %in% 2:3 ~ "lths", hleducat_le24 == 4 ~ "hs",
                hleducat_le24 == 5 ~ "college", hleducat_le24 %in% 6:7 ~ "grad")
    ),
    insur_bin = case_when(`_inscat` %in% 2:4 ~ 1, `_inscat` == 1 ~ 0),
    urbrur = case_when(`_nchs_rural2013` %in% 1:4 ~ 1, `_nchs_rural2013` %in% 5:6 ~ 0),
    homeless_p12m = if_else(homeless_p12m %in% 0:1, homeless_p12m, NA_real_),
    illicit = if_else(illicit %in% 0:1, illicit, NA_real_),

    stigma_family = recent_any(stigma_a1, stigma_a2),
    stigma_ahs = recent_any(stigma_b1, stigma_b2),
    stigma_gss = recent_any(stigma_c1, stigma_c2, stigma_c3,
                            recent_attributed(stigma_d1, stigma_d2),
                            recent_attributed(stigma_d3, stigma_d4)),
    out_gid = case_when(
      out_gi == 0 ~ 0,
      out_gi == 1 & out_gid == 1 ~ 1,
      out_gi == 1 & out_gid %in% c(0, 2) ~ 0
    ),
    seehcp = if_else(seehcp %in% 0:1, seehcp, NA_real_),
    risk_multi = case_when(
      m_muauhs == 1 | m_muahp == 1 ~ 1,
      m_muauhs == 0 & m_muahp %in% c(0, 8) ~ 0
    ),
    risk_single = case_when(
      m_mm1hsk == 0 | m_m1hst %in% c(2, 3) ~ 1,
      m_m1hst == 1 ~ 0
    ),
    risk_behavior = case_when(
      is.na(m_m1uas) | m_m1uas %in% c(0, 8) ~ 0,
      m_m1uas == 1 & !is.na(m_muauhs) ~ risk_multi,
      m_m1uas == 1 ~ risk_single
    ),
    prep_used = case_when(
      prep_current == 1 | prep_used == 1 ~ 1,
      heard_prep == 0 | prep_ever == 0 | prep_used == 0 ~ 0
    ),
    hivtest12_binary = if_else(`_lasthivtest12` %in% 0:1, `_lasthivtest12`, NA_real_),
    hivtest12 = case_when(
      evertest == 0 ~ 0,
      test2yrs >= 0 & test2yrs <= 100 ~ pmin(test2yrs, 4) / 4,
      evertest == 1 & `_mo_lasthivtest` > 24 ~ 0
    )
  )

# ---- Exclusions ------------------------------------------------------------------
# Apply the eligibility criteria in order, logging the respondents removed at each step.
steps <- list(
  "HIV-positive or HIV status missing" = quote(!is.na(`_hivstat`) & `_hivstat` != 2),
  "Aged under 18 or over 99" = quote(age >= 18 & age <= 99),
  "DC or outside the 50 states" = quote(!state_calc %in% c("DC", "") & !is.na(state_calc) & `_censdiv` %in% 1:9)
)
analytic <- recoded
exclusions <- tibble(step = "Respondents in AMIS 2017-2019, 2021-2024", removed = NA_integer_, remaining = nrow(analytic))
for (step in names(steps)) {
  keep <- analytic %>% filter(!!steps[[step]])
  exclusions <- add_row(exclusions, step = step, removed = nrow(analytic) - nrow(keep), remaining = nrow(keep))
  analytic <- keep
}

output_columns <- c("survey", "rowID", "vdatesub", "submission_year", "survey_year", "year_group", "state_calc", "_censdiv", "region",
                    "_hivstat", "age", "agecat", "rdrace", "school4", "insur_bin", "urbrur", "homeless_p12m",
                    "illicit", "stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "risk_behavior",
                    "prep_used", "hivtest12", "hivtest12_binary", "evertest", "test2yrs")
write_csv(exclusions, file.path(output_dir, "exclusions.csv"))
write_csv(select(analytic, all_of(output_columns)), gzfile(file.path(output_dir, "amis_analytic.csv.gz")))

print(exclusions)
print(analytic %>% group_by(survey) %>% summarise(n = n(), across(c(stigma_ahs, stigma_gss, stigma_family, prep_used, hivtest12), ~ round(100 * mean(.x, na.rm = TRUE), 1))))
