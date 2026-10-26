# AMIS 2022 change to the anticipated healthcare stigma (AHS) items: was the 2022 rise in AHS a wording
# effect? Supplementary wording analyses and the series for the measurement check in Fig. 1.
#
# Both AHS items (stigma_b1, b2) changed in 2022 from "worry someone may learn you have sex with men"
# to "worry someone may treat you poorly because you have sex with men". Three sets of items did not
# change and serve as controls: experienced healthcare stigma (b3: heard providers gossiping about you;
# b4: not treated well in a health center because someone knew), general social stigma and family
# stigma. All are "in the last 6 months" composites (yes to any item), coded as in scripts/00.
#
# Outputs (results/02_inputs/wording/):
#   items_by_year.csv       national % by survey year (national weights, scripts/02): AHS items and
#                           composite, AHS adjusted as in the model (scripts/01 shift for 2022+),
#                           experienced healthcare stigma items and composite, general social, family;
#                           survey-weighted 95% intervals for AHS (measured and adjusted) and experienced stigma
#   shift_2022.csv          2022 vs 2021 log-odds shift (state and age adjusted, state weights, as in
#                           scripts/01) for each measure; AHS minus each control (difference in
#                           differences, respondents clustered); the same contrast for 2018 vs 2017 and
#                           2024 vs 2023 (no wording change) as placebos
#   shift_2022_by_prep.csv  the 2022 vs 2021 shift among PrEP users and non-users, and the difference
#
# Reads the restricted AMIS files in place (inputs/restricted/amis_raw) and writes aggregates only.
# Run from the repository root: Rscript scripts/08c_wording_items.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(haven)
  library(survey)
})

output_dir <- "results/02_inputs/wording"
raw_dir <- "inputs/restricted/amis_raw"
cycles <- c(amis_17 = "sitq2017_analysis_noid.dta", amis_18 = "sitq2018_analysis_noid.dta",
            amis_19 = "sitq2019_analysis_noid.dta", amis_21 = "sitq2021_analysis_noid.dta",
            amis_22 = "sitq2022_analysis_fulldata_noid.dta", amis_23 = "sitq2023_analysis_final_noid.dta",
            amis_24 = "sitq2024_analysis_final_NOID.dta")
wording_shift <- read_csv("results/01_weights/ahs_wording_shift.csv", show_col_types = FALSE)$shift_log_odds
measures <- c(ahs = "stigma_ahs", ehs = "ehs", gss = "stigma_gss", family = "stigma_family")

# 1 = any item yes in the last 6 months; 0 = all answered items no or not in the last 6 months.
recent_any <- function(...) {
  items <- cbind(...)
  case_when(rowSums(items == 1, na.rm = TRUE) > 0 ~ 1, rowSums(items == 0 | items == 2, na.rm = TRUE) > 0 ~ 0)
}

# Raw healthcare items, linked to the analytic file by rowID (survey and raw row number, scripts/00).
items <- imap_dfr(cycles, function(file, survey) {
  path <- file.path(raw_dir, file)
  available <- names(read_dta(path, n_max = 0))
  read_dta(path, col_select = all_of(available[tolower(available) %in% paste0("stigma_b", 1:4)])) %>%
    rename_with(tolower) %>%
    zap_labels() %>%
    mutate(rowID = paste(survey, row_number(), sep = "_"))
}) %>%
  transmute(rowID, ahs_b1 = recent_any(stigma_b1), ahs_b2 = recent_any(stigma_b2),
            ehs_b3 = recent_any(stigma_b3), ehs_b4 = recent_any(stigma_b4), ehs = recent_any(stigma_b3, stigma_b4))

# ---- National series by year -------------------------------------------------------------

national <- readRDS("results/01_weights/national_raked.rds")$variables %>%
  select(rowID, survey_year, stigma_ahs, stigma_gss, stigma_family, national_weight) %>%
  inner_join(items, by = "rowID")
by_year <- national %>%
  group_by(survey_year) %>%
  summarise(n = n(), across(c(ahs_b1, ahs_b2, stigma_ahs, ehs_b3, ehs_b4, ehs, stigma_gss, stigma_family),
                            ~ weighted.mean(.x, national_weight, na.rm = TRUE)), .groups = "drop") %>%
  mutate(stigma_ahs_adjusted = if_else(survey_year >= 2022, plogis(qlogis(stigma_ahs) - wording_shift), stigma_ahs),
         new_wording = survey_year >= 2022)

# Survey-weighted 95% intervals (logit scale, national weights) for the series shown in Fig. 1B; the
# adjusted AHS interval is the measured one moved by the same shift.
ci <- map_dfr(c("stigma_ahs", "ehs"), function(v) {
  d <- national %>% filter(!is.na(.data[[v]]))
  est <- svyby(reformulate(v), ~survey_year, svydesign(ids = ~1, weights = ~national_weight, data = d),
               svyciprop, vartype = "ci", method = "logit")
  tibble(survey_year = est$survey_year, measure = v, low = est$ci_l, high = est$ci_u)
}) %>%
  pivot_wider(names_from = measure, values_from = c(low, high), names_glue = "{measure}_{.value}")
by_year <- by_year %>%
  left_join(ci, by = "survey_year") %>%
  mutate(stigma_ahs_adjusted_low = if_else(survey_year >= 2022, plogis(qlogis(stigma_ahs_low) - wording_shift), stigma_ahs_low),
         stigma_ahs_adjusted_high = if_else(survey_year >= 2022, plogis(qlogis(stigma_ahs_high) - wording_shift), stigma_ahs_high))

# ---- 2022 vs 2021 shifts and difference in differences ------------------------------------

state <- readRDS("results/01_weights/state_raked.rds")$variables %>%
  select(rowID, survey, survey_year, year_group, state_calc, agecat, prep_used, stigma_ahs, stigma_gss,
         stigma_family, raked_weight) %>%
  inner_join(items %>% select(rowID, ehs), by = "rowID")

pair_data <- function(old, new) {
  state %>%
    filter(survey_year %in% c(old, new)) %>%
    group_by(state_calc) %>%
    mutate(weight = raked_weight * n()) %>%          # each state weighted to its sample size, as in scripts/01
    ungroup() %>%
    mutate(post = as.integer(survey_year == new), agecat = factor(agecat), state_calc = factor(state_calc))
}

shift <- function(d, measure) {
  d <- filter(d, !is.na(.data[[measure]]))
  fit <- svyglm(reformulate(c("post", "state_calc", "agecat"), measure),
                design = svydesign(ids = ~1, weights = ~weight, data = d), family = quasibinomial())
  tibble(estimate = coef(fit)[["post"]], se = sqrt(vcov(fit)["post", "post"]), n = nrow(d))
}

# AHS minus a control: both measures stacked per respondent, measure-specific state and age terms,
# respondents clustered.
did <- function(d, controls) {
  long <- d %>%
    select(rowID, weight, post, state_calc, agecat, all_of(c("stigma_ahs", controls))) %>%
    pivot_longer(-c(rowID, weight, post, state_calc, agecat), names_to = "measure", values_to = "y") %>%
    filter(!is.na(y)) %>%
    mutate(ahs = as.integer(measure == "stigma_ahs"), measure = factor(measure))
  fit <- svyglm(y ~ measure * (state_calc + agecat) + post + post:ahs,
                design = svydesign(ids = ~rowID, weights = ~weight, data = long), family = quasibinomial())
  tibble(estimate = coef(fit)[["post:ahs"]], se = sqrt(vcov(fit)["post:ahs", "post:ahs"]), n = n_distinct(long$rowID))
}

contrasts <- list(c(2021, 2022), c(2017, 2018), c(2023, 2024))
shifts <- map_dfr(contrasts, function(years) {
  d <- pair_data(years[1], years[2])
  single <- imap_dfr(measures, ~ shift(d, .x) %>% mutate(term = paste0("shift_", .y)))
  differences <- list(gss = "stigma_gss", family = "stigma_family", ehs = "ehs",
                      gss_family = c("stigma_gss", "stigma_family")) %>%
    imap_dfr(~ did(d, .x) %>% mutate(term = paste0("ahs_minus_", .y)))
  bind_rows(single, differences) %>% mutate(contrast = paste(years[2], "vs", years[1]), .before = 1)
}) %>%
  mutate(low = estimate - 1.96 * se, high = estimate + 1.96 * se, p = 2 * pnorm(-abs(estimate / se)))

by_prep <- {
  d <- pair_data(2021, 2022)
  expand_grid(measure = names(measures), prep_used = c(0, 1)) %>%
    mutate(fit = map2(measure, prep_used, ~ shift(filter(d, prep_used == .y), measures[[.x]]))) %>%
    unnest(fit) %>%
    group_by(measure) %>%
    group_modify(~ bind_rows(
      .x %>% mutate(group = if_else(prep_used == 1, "PrEP users", "non-users")),
      tibble(group = "users minus non-users", estimate = diff(.x$estimate), se = sqrt(sum(.x$se^2)))
    )) %>%
    ungroup() %>%
    select(measure, group, estimate, se, n) %>%
    mutate(low = estimate - 1.96 * se, high = estimate + 1.96 * se, p = 2 * pnorm(-abs(estimate / se)))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(by_year, file.path(output_dir, "items_by_year.csv"))
write_csv(shifts, file.path(output_dir, "shift_2022.csv"))
write_csv(by_prep, file.path(output_dir, "shift_2022_by_prep.csv"))

print(by_year %>% mutate(across(where(is.double) & !survey_year, ~ round(100 * .x, 1))), width = Inf)
print(shifts %>% filter(contrast == "2022 vs 2021") %>% mutate(across(where(is.double), ~ round(.x, 3))), width = Inf)
print(by_prep %>% mutate(across(where(is.double), ~ round(.x, 3))), width = Inf)
