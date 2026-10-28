# AMIS 2022 change to the anticipated healthcare stigma (AHS) items: the national series for Fig. 1B and
# the evidence that the wording adjustment (scripts/1_data/03) is conservative.
#
# Both AHS items (stigma_b1, b2) changed in 2022 from "worry someone may learn you have sex with men" to
# "worry someone may treat you poorly because you have sex with men". Three measures did not change:
# experienced healthcare stigma (b3: heard providers gossiping about you; b4: not treated well in a health
# center because someone knew), general social stigma and family stigma. All are "in the last 6 months"
# composites (yes to any item), coded as in scripts/1_data/01.
#
# Outputs (results/02_inputs/wording/):
#   items_by_year.csv  national % by survey year (national weights, scripts/5_demographic/01) with survey-weighted 95%
#                      intervals: AHS as measured, AHS adjusted as in the model (the scripts/1_data/03 shift for
#                      2022 onward; its interval is the measured one moved by the same shift), and
#                      experienced healthcare stigma (Fig. 1B)
#   shift_2022.csv     2022 vs 2021 log-odds shift for AHS and the three unchanged measures, adjusted for
#                      state and age with state-period weights, estimated as in scripts/1_data/03 (the AHS row
#                      reproduces the shift the model removes). Unchanged measures rising by a similar
#                      amount indicate that part of the AHS rise is real, so removing all of it is
#                      conservative.
#
# Reads the restricted AMIS files in place (inputs/restricted/amis_raw) and writes aggregates only.
# Run from the repository root: Rscript scripts/2_evidence/02_wording_items.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(haven)
  library(survey)
})

output_dir <- "results/02_inputs/wording"
raw_dir <- "inputs/restricted/amis_raw"
new_wording_first_year <- 2022
cycles <- c(amis_17 = "sitq2017_analysis_noid.dta", amis_18 = "sitq2018_analysis_noid.dta",
            amis_19 = "sitq2019_analysis_noid.dta", amis_21 = "sitq2021_analysis_noid.dta",
            amis_22 = "sitq2022_analysis_fulldata_noid.dta", amis_23 = "sitq2023_analysis_final_noid.dta",
            amis_24 = "sitq2024_analysis_final_NOID.dta")
wording_shift <- read_csv("results/01_weights/ahs_wording_shift.csv", show_col_types = FALSE)$shift_log_odds
measures <- c(ahs = "stigma_ahs", ehs = "ehs", gss = "stigma_gss", family = "stigma_family")

# Reconstruct a six-month composite from its component items. AMIS codes 1 as "yes", 0 as "no", and 2
# as an experience that did not occur within the last six months. Thus:
#   composite = 1  if at least one answered item is a recent experience;
#   composite = 0  if no item is recent and at least one answered item is no/not recent;
#   composite = NA if every component is missing.
# This matches the composite-variable construction in scripts/1_data/01_clean_amis.R.
recent_any <- function(...) {
  items <- cbind(...)
  case_when(rowSums(items == 1, na.rm = TRUE) > 0 ~ 1, rowSums(items == 0 | items == 2, na.rm = TRUE) > 0 ~ 0)
}

# Experienced healthcare stigma (EHS) is not part of the eight-variable model input, so reconstruct it from
# the unchanged raw b3/b4 items. read_dta(..., n_max = 0) first reads only the column names because their
# capitalization differs across survey cycles. rowID combines the AMIS cycle and original row number, as in
# scripts/1_data/01_clean_amis.R, allowing these reconstructed values to be joined to the weighted analytic data.
ehs <- imap_dfr(cycles, function(file, survey) {
  path <- file.path(raw_dir, file)
  available <- names(read_dta(path, n_max = 0))                 # variable-name case differs across cycles
  read_dta(path, col_select = all_of(available[tolower(available) %in% c("stigma_b3", "stigma_b4")])) %>%
    rename_with(tolower) %>%
    zap_labels() %>%
    transmute(rowID = paste(survey, row_number(), sep = "_"), ehs = recent_any(stigma_b3, stigma_b4))
})

# ---- National series by year (Fig. 1B) ---------------------------------------------------------

# Use the national raking weights from scripts/5_demographic/01. The inner join deliberately retains only respondents
# whose raw records could be linked to the analytic file; measure-specific missing values are then removed
# separately below so AHS and EHS each use all linked respondents observed for that measure.
national <- readRDS("results/01_weights/national_raked.rds")$variables %>%
  select(rowID, survey_year, stigma_ahs, national_weight) %>%
  inner_join(ehs, by = "rowID")

# Estimate one measure's national weighted prevalence and 95% confidence interval separately by survey
# year. The logit method is used only to keep confidence limits within [0, 1]; the point estimate is the
# ordinary survey-weighted proportion.
annual_prevalence <- function(data, variable) {
  complete_data <- data %>%
    filter(!is.na(.data[[variable]]))

  design <- svydesign(
    ids = ~1,
    weights = ~national_weight,
    data = complete_data
  )

  estimates <- svyby(
    reformulate(variable),
    ~survey_year,
    design,
    svyciprop,
    vartype = "ci",
    method = "logit"
  )

  tibble(
    survey_year = estimates$survey_year,
    value = as.numeric(coef(estimates)),
    low = estimates$ci_l,
    high = estimates$ci_u
  )
}

# Estimate the two annual series separately so it is explicit which measure each table represents.
ahs_by_year <- annual_prevalence(national, "stigma_ahs") %>%
  rename(stigma_ahs = value, stigma_ahs_low = low, stigma_ahs_high = high)

ehs_by_year <- annual_prevalence(national, "ehs") %>%
  rename(ehs = value, ehs_low = low, ehs_high = high)

# For 2022 onward, move a measured AHS probability back by the wording-shift estimate used in the main
# model: adjusted probability = expit[logit(measured probability) - wording_shift]. The wording shift is
# treated as fixed here, consistent with the model-input adjustment in scripts/1_data/03_state_weights.R.
adjust_ahs_wording <- function(probability, year) {
  if_else(
    year >= new_wording_first_year,
    plogis(qlogis(probability) - wording_shift),
    probability
  )
}

by_year <- ahs_by_year %>%
  inner_join(ehs_by_year, by = "survey_year") %>%
  mutate(
    stigma_ahs_adjusted = adjust_ahs_wording(stigma_ahs, survey_year),
    stigma_ahs_adjusted_low = adjust_ahs_wording(stigma_ahs_low, survey_year),
    stigma_ahs_adjusted_high = adjust_ahs_wording(stigma_ahs_high, survey_year)
  ) %>%
  select(
    survey_year,
    stigma_ahs,
    stigma_ahs_low,
    stigma_ahs_high,
    stigma_ahs_adjusted,
    stigma_ahs_adjusted_low,
    stigma_ahs_adjusted_high,
    ehs,
    ehs_low,
    ehs_high
  )

# ---- 2022 vs 2021 shifts --------------------------------------------------------------------------

# Restrict to the paired 2021-2022 model period so that "post" compares the two adjacent AMIS cycles.
# The state-period weights come from scripts/1_data/03; the reconstructed EHS value is joined by respondent rowID.
state <- readRDS("results/01_weights/state_raked.rds")$variables %>%
  filter(year_group == "2021-2022") %>%
  select(rowID, survey_year, state_calc, agecat, all_of(setdiff(measures, "ehs")), raked_weight) %>%
  inner_join(ehs, by = "rowID")

# Estimate the 2022-versus-2021 change separately for AHS and each unchanged comparison measure. As in
# scripts/1_data/03_state_weights.R, multiply the within-state normalized raking weights by the state's retained
# sample size. This leaves weighted estimates within a state unchanged while giving states influence in
# proportion to their respondent counts.
#
# For binary measure Y, the quasi-binomial logit model is:
#   logit P(Y_i = 1) = alpha + beta_post I(year_i = 2022) + state_i + age_i.
# beta_post is therefore the adjusted 2022 change in log odds. If unchanged measures have a similar positive
# beta_post to AHS, the observed AHS rise cannot plausibly be attributed entirely to its wording change.
shift <- function(measure) {
  d <- state %>%
    filter(!is.na(.data[[measure]])) %>%
    group_by(state_calc) %>%
    mutate(weight = raked_weight * n()) %>%
    ungroup() %>%
    mutate(post = as.integer(survey_year == 2022), agecat = factor(agecat), state_calc = factor(state_calc))
  fit <- svyglm(reformulate(c("post", "state_calc", "agecat"), measure),
                design = svydesign(ids = ~1, weights = ~weight, data = d), family = quasibinomial())

  # Return the log-odds change and its design-based standard error; confidence limits are added below.
  tibble(estimate = coef(fit)[["post"]], se = sqrt(vcov(fit)["post", "post"]), n = nrow(d))
}
shifts <- imap_dfr(measures, ~ shift(.x) %>% mutate(measure = .y, .before = 1)) %>%

  # item_changed distinguishes AHS from the three negative-control-like comparison measures whose wording
  # did not change. These comparisons diagnose a survey-wide temporal change; they are not causal controls.
  mutate(item_changed = measure == "ahs", low = estimate - 1.96 * se, high = estimate + 1.96 * se)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(by_year, file.path(output_dir, "items_by_year.csv"))
write_csv(shifts, file.path(output_dir, "shift_2022.csv"))
print(by_year %>% mutate(across(-survey_year, ~ round(100 * .x, 1))), width = Inf)
print(shifts %>% mutate(across(where(is.double), ~ round(.x, 3))), width = Inf)
