# State-year CDC surveillance and PrEP stocks for MSM (formerly `EEE Stocks 7.csv`).
#
# Replaces three notebooks in EEE/EEE Feb 2025: `EEE CDC Clean.ipynb` (AtlasPlus download -> CDC Data
# Raw), `AIDSVU PrEP Cleaning/PrEP Cleaning.ipynb` (AIDSVu state PrEP files -> PrEP Data) and
# `EEE Stocks Final 2026 - Stocks 7.ipynb`. Ported from Python (2026-10-27; retained indicators identical).
#
# Inputs (inputs/cdc_raw/, public): AtlasPlus state HIV indicators, the original AIDSVu state PrEP
# workbooks for 2017-2023, and state MSM population estimates (inputs/msm_state_population.csv).
# AtlasPlus indicators come from the reproducibly combined atlasplus_combined.csv (October 2026 exports;
# see inputs/cdc_raw/SOURCES.md). Years after 2024 are excluded because AtlasPlus marks 2025-2026 as
# preliminary.
#
# PrEP stocks (MSM):
#   PrEP           AIDSVu male PrEP users x 0.98 (MSM share of male users)
#   PrEP Eligible  CDC-implied men with PrEP indications (persons prescribed / coverage), x 2 (denominator
#                  scale); missing state-years are imputed (within-state mean of available years, or the
#                  cross-state mean eligible/MSM-population ratio when a state has none); 2023 is the
#                  2017-2022 mean
# Surveillance stocks (MSM): diagnoses and diagnosed prevalence sum the two MSM transmission categories.
# Estimated incidence and estimated prevalence are available only for the all-population series and are
# allocated to MSM using the corresponding MSM share of diagnoses or diagnosed prevalence. Viral suppression
# sums the two MSM transmission categories and is imputed, where missing, as a bounded rate of estimated MSM
# prevalence. After the last CDC estimate year, estimated MSM prevalence is diagnosed MSM prevalence times
# the state's last observed estimated-to-diagnosed prevalence ratio; estimated incidence is not extended.
# Only indicators consumed by the weighting or epidemiological-model pipeline are written.
#
# National PrEP targets by age group (for the national raking, scripts/5_demographic/01):
#   target_age = target_men x coverage_age / coverage_all
#   target_men     national PrEP / PrEP Eligible from the state stocks above
#   coverage_age   CDC persons prescribed / persons with indications by age (both sexes, age rows of
#                  atlasplus_combined.csv, ages 16-54, states with data)
#   coverage_all   all ages combined, weighting each age group by its AIDSVu users
# Ages 55+ have no CDC indication estimate and are assumed to have the coverage of ages 45-54; the 35+ group
# combines 35-44, 45-54 and 55+ by AIDSVu users. 2023 (no CDC age data): indications are the 2017-2022 mean
# by age, with AIDSVu 2023 users, as for the state eligible counts.
#
# Each stock carries its imputation_method where it is built: "none" (observed), "missing", or the rule
# that produced the value.
#
# Outputs: results/00_cdc/cdc_stocks.csv (St_Abbr, Year, Cases, Population, Indicator, imputation_method),
#          results/00_cdc/national_prep_age_targets.csv
#
# Run from the repository root: Rscript scripts/1_data/02_build_cdc_stocks.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
})

# ---- Settings ---------------------------------------------------------------------------------

raw_dir <- "inputs/cdc_raw"
output_path <- "results/00_cdc/cdc_stocks.csv"
age_targets_path <- "results/00_cdc/national_prep_age_targets.csv"

msm_share <- 0.98                  # share of male PrEP users who are MSM
eligible_scale <- 2.0              # PrEP-eligible denominator scale
last_surveillance_year <- 2024     # later AtlasPlus years are preliminary (12-month reporting delay)
eligible_years <- 2017:2022
prep_years <- c(eligible_years, 2023)
msm_tcs <- c("Male-to-male sexual contact", "Male-to-male sexual contact and injection drug use")
non_states <- c("DC", "PR", "District of Columbia")
prep_indicator <- "PrEP coverage and number of persons prescribed"
age_cols <- c("Age LE 24 PrEP Users" = "<=24", "Age 25-34 PrEP Users" = "25-34", "Age 35-44 PrEP Users" = "35-44",
              "Age 45-54 PrEP Users" = "45-54", "Age 55+ PrEP Users" = "55+")

state_abbr <- c(
  "Alabama" = "AL", "Alaska" = "AK", "American Samoa" = "AS", "Arizona" = "AZ", "Arkansas" = "AR",
  "California" = "CA", "Colorado" = "CO", "Connecticut" = "CT", "Delaware" = "DE",
  "District of Columbia" = "District of Columbia", "Florida" = "FL", "Georgia" = "GA", "Guam" = "GU",
  "Hawaii" = "HI", "Idaho" = "ID", "Illinois" = "IL", "Indiana" = "IN", "Iowa" = "IA", "Kansas" = "KS",
  "Kentucky" = "KY", "Louisiana" = "LA", "Maine" = "ME", "Marshall Islands" = "MH", "Maryland" = "MD",
  "Massachusetts" = "MA", "Michigan" = "MI", "Micronesia" = "FM", "Minnesota" = "MN", "Mississippi" = "MS",
  "Missouri" = "MO", "Montana" = "MT", "Nebraska" = "NE", "Nevada" = "NV", "New Hampshire" = "NH",
  "New Jersey" = "NJ", "New Mexico" = "NM", "New York" = "NY", "North Carolina" = "NC",
  "North Dakota" = "ND", "Northern Marianas" = "MP", "Ohio" = "OH", "Oklahoma" = "OK", "Oregon" = "OR",
  "Palau" = "PW", "Pennsylvania" = "PA", "Puerto Rico" = "PR", "Rhode Island" = "RI",
  "South Carolina" = "SC", "South Dakota" = "SD", "Tennessee" = "TN", "Texas" = "TX", "Utah" = "UT",
  "Vermont" = "VT", "Virgin Islands" = "VI", "Virginia" = "VA", "Washington" = "WA",
  "West Virginia" = "WV", "Wisconsin" = "WI", "Wyoming" = "WY", "USA" = "USA"
)

# ---- Helpers ----------------------------------------------------------------------------------

# Sum that is NA when every value is NA (pandas sum(min_count=1)).
sum_or_na <- function(x) if (all(is.na(x))) NA_real_ else sum(x, na.rm = TRUE)

# Sum of an indicator over the given transmission categories, by state and year.
series <- function(base, indicator, tc) {
  base %>%
    filter(Indicator == indicator, `Transmission Category` %in% tc) %>%
    group_by(St_Abbr, Year) %>%
    summarise(Cases = sum_or_na(Cases), .groups = "drop")
}

# Rows in the output layout. Rows without an imputation_method get "none" (observed) or "missing".
stock <- function(df, indicator) {
  if (!"imputation_method" %in% names(df)) df$imputation_method <- if_else(is.na(df$Cases), "missing", "none")
  df %>%
    mutate(Population = "MSM", Indicator = indicator) %>%
    select(St_Abbr, Year, Cases, Population, Indicator, imputation_method)
}

# All-population total x (MSM part / all part), clipped to [0, 1]; missing if the MSM part is.
allocate_to_msm <- function(total_all, msm_part, all_part) {
  total_all %>%
    select(St_Abbr, Year, Total = Cases) %>%
    left_join(select(msm_part, St_Abbr, Year, MSM = Cases), by = c("St_Abbr", "Year")) %>%
    left_join(select(all_part, St_Abbr, Year, ALL = Cases), by = c("St_Abbr", "Year")) %>%
    mutate(ALL = na_if(coalesce(ALL, 0), 0), Cases = Total * pmin(pmax(MSM / ALL, 0), 1)) %>%
    select(St_Abbr, Year, Cases)
}

# Linear interpolation by position within a series; ends take the nearest observed value
# (pandas interpolate(method="linear", limit_direction="both")).
interpolate_both <- function(x) {
  ok <- which(!is.na(x))
  if (length(ok) == 0) return(x)
  if (length(ok) == 1) return(rep(x[ok], length(x)))
  approx(ok, x[ok], xout = seq_along(x), rule = 2)$y
}

# ---- Raw sources ------------------------------------------------------------------------------

atlasplus <- read_csv(file.path(raw_dir, "atlasplus_combined.csv"), col_types = cols(fips = col_character()),
                      show_col_types = FALSE)

# State-level rows (ages 13+) through the last surveillance year: numeric cases (estimates keep the point
# value), zeros as missing. PrEP coverage is the men's series; the both-sexes total and age rows are not used.
cdc <- atlasplus %>%
  filter(age_group == "13+", year <= last_surveillance_year, indicator != prep_indicator | sex == "Male") %>%
  transmute(Indicator = indicator, Year = as.integer(year), `Transmission Category` = transmission_category,
            Cases = na_if(cases, 0), Percent = percent,
            St_Abbr = unname(state_abbr[str_remove(geography, "[\\^*]+$")]))   # strip footnote markers

# AIDSVu state PrEP workbooks, one per year: male users and users by age group (numbers stored as text).
aidsvu <- map_dfr(prep_years, function(year) {
  read_excel(file.path(raw_dir, sprintf("AIDSVu_State_PrEP_%d_20240522.xlsx", year)), skip = 3, col_types = "text") %>%
    select(St_Abbr = `State Abbreviation`, male_users = `Male PrEP Users`, all_of(names(age_cols))) %>%
    mutate(Year = as.integer(year), across(-c(St_Abbr, Year), ~ suppressWarnings(as.numeric(.x))))
})

msm_pop <- read_csv("inputs/msm_state_population.csv", show_col_types = FALSE, name_repair = "unique_quiet") %>%
  select(St_Abbr = state, msm_pop = mean)

# ---- PrEP stocks --------------------------------------------------------------------------------

prep_users <- aidsvu %>%
  filter(!St_Abbr %in% non_states) %>%
  transmute(St_Abbr, Year, Cases = male_users * msm_share) %>%
  stock("PrEP")

# PrEP-eligible MSM: men with indications implied by CDC coverage, imputed where missing, 2023 as the
# 2017-2022 mean, then scaled. Each state-year's label records how its value was obtained.
implied <- cdc %>%
  filter(Indicator == prep_indicator, !is.na(St_Abbr)) %>%
  mutate(eligible = if_else(Percent > 0, Cases / (Percent / 100), NA_real_)) %>%
  group_by(St_Abbr, Year) %>%
  summarise(eligible = mean(eligible, na.rm = TRUE), .groups = "drop") %>%
  mutate(eligible = if_else(is.nan(eligible), NA_real_, eligible))
m <- sapply(eligible_years, function(y) {          # state (msm_pop order) x year
  w <- filter(implied, Year == y)
  w$eligible[match(msm_pop$St_Abbr, w$St_Abbr)]
})
raw_missing <- is.na(m)
n_missing <- rowSums(raw_missing)
all_missing <- n_missing == length(eligible_years)
part_missing <- n_missing > 0 & !all_missing
mean_ratio <- mean(rowMeans(m[n_missing == 0, , drop = FALSE]) / msm_pop$msm_pop[n_missing == 0])
m[all_missing, ] <- msm_pop$msm_pop[all_missing] * mean_ratio
for (i in which(part_missing)) m[i, is.na(m[i, ])] <- mean(m[i, ], na.rm = TRUE)
m <- cbind(m, rowMeans(m)) * eligible_scale
method <- cbind(ifelse(raw_missing & all_missing, "prep_eligible_cross_state_mean_proportion",
                       ifelse(raw_missing & part_missing, "prep_eligible_within_state_available_year_mean", "none")),
                "prep_eligible_2017_2022_mean")
prep_eligible <- tibble(St_Abbr = rep(msm_pop$St_Abbr, length(prep_years)),
                        Year = rep(as.integer(prep_years), each = nrow(msm_pop)),
                        Cases = as.vector(m), imputation_method = as.vector(method)) %>%
  filter(!St_Abbr %in% non_states) %>%
  stock("PrEP Eligible")

# ---- Surveillance stocks ----------------------------------------------------------------------

base <- cdc %>%
  filter(!is.na(St_Abbr), !is.na(Indicator), !is.na(`Transmission Category`)) %>%
  group_by(St_Abbr, Year, Indicator, `Transmission Category`) %>%
  summarise(Cases = sum_or_na(Cases), .groups = "drop")
missing_tcs <- setdiff(msm_tcs, unique(base$`Transmission Category`))
if (length(missing_tcs) > 0) stop("Missing MSM transmission categories: ", paste(missing_tcs, collapse = ", "))

# Directly observed MSM stocks: sum the two AtlasPlus MSM transmission categories.
msm_diag <- series(base, "HIV diagnoses", msm_tcs)
msm_prev <- series(base, "HIV prevalence", msm_tcs)
diagnoses <- stock(msm_diag, "HIV diagnoses")

# CDC publishes estimated incidence and prevalence only for the full population. Allocate them to MSM using
# MSM's observed share of diagnoses or diagnosed prevalence in the same state-year.
incidence <- allocate_to_msm(series(base, "Estimated HIV incidence", "All"), msm_diag,
                             series(base, "HIV diagnoses", "All")) %>%
  stock("Estimated HIV incidence (MSM)")
estimated <- allocate_to_msm(series(base, "Estimated HIV prevalence (undiagnosed and diagnosed)", "All"), msm_prev,
                             series(base, "HIV prevalence", "All"))

# After the last estimate year: diagnosed MSM prevalence x the state's last estimated/diagnosed ratio.
last_estimate <- max(cdc$Year[cdc$Indicator == "Estimated HIV prevalence (undiagnosed and diagnosed)" & !is.na(cdc$Cases)])
ratio <- estimated %>%
  filter(!is.na(Cases)) %>%
  group_by(St_Abbr) %>%
  slice_max(Year, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  inner_join(rename(msm_prev, diagnosed = Cases), by = c("St_Abbr", "Year")) %>%
  transmute(St_Abbr, last_year = Year, ratio = Cases / diagnosed)
extended <- msm_prev %>%
  inner_join(ratio, by = "St_Abbr") %>%
  filter(Year > last_year) %>%
  transmute(St_Abbr, Year, Cases = Cases * ratio)
prevalence <- bind_rows(estimated, extended) %>%
  mutate(imputation_method = case_when(is.na(Cases) ~ "missing",
                                       Year > last_estimate ~ sprintf("diagnosed_x_%d_estimated_to_diagnosed_ratio", last_estimate),
                                       TRUE ~ "none")) %>%
  stock("Estimated HIV prevalence (MSM)")

# Viral suppression: missing counts filled as a bounded rate of estimated MSM prevalence. Rates are
# interpolated within state, then filled with the year median, then the overall median.
suppression <- series(base, "HIV viral suppression", msm_tcs) %>%
  left_join(select(prevalence, St_Abbr, Year, prev = Cases), by = c("St_Abbr", "Year")) %>%
  mutate(raw_rate = if_else(!is.na(Cases) & !is.na(prev) & prev > 0, pmin(pmax(Cases / prev, 0), 1), NA_real_)) %>%
  arrange(St_Abbr, Year) %>%
  group_by(St_Abbr) %>%
  mutate(observed_state = any(!is.na(raw_rate)), rate = interpolate_both(raw_rate)) %>%
  group_by(Year) %>%
  mutate(rate = coalesce(rate, median(raw_rate, na.rm = TRUE))) %>%
  ungroup() %>%
  mutate(rate = pmin(pmax(coalesce(rate, median(raw_rate, na.rm = TRUE)), 0), 1),
         fill = is.na(Cases) & !is.na(prev) & prev > 0 & !is.na(rate),
         imputation_method = case_when(fill & observed_state ~ "rate_impute_vs_within_state",
                                       fill ~ "rate_impute_vs_year_median",
                                       is.na(Cases) ~ "missing",
                                       TRUE ~ "none"),
         Cases = if_else(fill, rate * prev, Cases)) %>%
  stock("HIV viral suppression")

# ---- National PrEP targets by age group -------------------------------------------------------

coverage <- atlasplus %>%
  filter(indicator == prep_indicator, sex == "Both sexes", age_group != "13+", !is.na(cases), percent > 0,
         !geography %in% c("District of Columbia", "Puerto Rico")) %>%
  mutate(indicated = cases / (percent / 100), age = recode(age_group, "13-24" = "<=24")) %>%
  group_by(Year = as.integer(year), age) %>%
  summarise(coverage = sum(cases) / sum(indicated), .groups = "drop") %>%
  pivot_wider(names_from = age, values_from = coverage) %>%
  mutate(`55+` = `45-54`)
users <- aidsvu %>%
  filter(!St_Abbr %in% non_states) %>%
  group_by(Year) %>%
  summarise(across(all_of(names(age_cols)), ~ sum(.x, na.rm = TRUE)), .groups = "drop") %>%
  rename_with(~ unname(age_cols[.x]), all_of(names(age_cols)))
ages <- unname(age_cols)
u <- as.matrix(users[ages]); rownames(u) <- users$Year
cov <- as.matrix(coverage[ages]); rownames(cov) <- coverage$Year
indicated <- u[as.character(eligible_years), ] / cov[as.character(eligible_years), ages]
indicated <- rbind(indicated, "2023" = colMeans(indicated))
groups <- list("<=24" = "<=24", "25-34" = "25-34", "35+" = c("35-44", "45-54", "55+"))
cov_group <- sapply(groups, function(a) rowSums(u[, a, drop = FALSE]) / rowSums(indicated[, a, drop = FALSE]))
cov_all <- rowSums(u) / rowSums(indicated)
men <- bind_rows(prep_users, prep_eligible) %>%
  group_by(Year, Indicator) %>%
  summarise(Cases = sum(Cases, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = Indicator, values_from = Cases)
target_men <- setNames(men$PrEP / men$`PrEP Eligible`, men$Year)

age_targets <- expand_grid(Year = prep_years, age_group = names(groups)) %>%
  mutate(y = as.character(Year),
         coverage_ratio = map2_dbl(y, age_group, ~ cov_group[.x, .y] / cov_all[[.x]]),
         target_men = unname(target_men[y]),
         prep_prop = target_men * coverage_ratio,
         prep_eligible = map2_dbl(y, age_group, ~ eligible_scale * sum(indicated[.x, groups[[.y]]])),
         prep_users = prep_prop * prep_eligible) %>%
  select(Year, age_group, prep_prop, prep_eligible, prep_users, coverage_ratio, target_men)

# ---- Write ----------------------------------------------------------------------------------------

stocks <- bind_rows(prep_users, prep_eligible, diagnoses, incidence, suppression, prevalence) %>%
  arrange(St_Abbr, Population, Indicator, Year)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_csv(stocks, output_path, na = "")
write_csv(age_targets, age_targets_path, na = "")
message(sprintf("Wrote %d rows to %s", nrow(stocks), output_path))
print(age_targets %>% select(Year, age_group, prep_prop) %>% pivot_wider(names_from = age_group, values_from = prep_prop))
