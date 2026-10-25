# Individual-level, access-adjusted association between anticipated healthcare stigma (AHS) and
# past-year PrEP use, by state. It is the evidence target for the total AHS -> PrEP response of the
# dynamic model (pipeline/state_fit.py, models/sbm/calibration.py) and the reference for the
# model-vs-evidence diagnostic (scripts/13c).
#
# For each state passing the median-55 precision screen (a superset of the primary and strict
# screens), a survey-weighted Poisson model of PrEP use on AHS, adjusted for confounders only:
# insurance, age, race/ethnicity, education, risk behavior (a proxy for PrEP indication) and AMIS
# cycle. Healthcare contact, disclosure and HIV testing are not adjusted for: they lie on pathways
# from AHS to PrEP, and the association is matched to the model's total AHS -> PrEP response. State
# log prevalence ratios are pooled toward their random-effects mean (DerSimonian-Laird, empirical
# Bayes); the mean is used for divisions and the nation. se_pooled is the posterior SD of each
# pooled estimate (including the uncertainty of the mean); the fit draws each unit's ratio from
# it (log scale).
#
# Wording: AMIS changed the AHS item in 2022. The new wording classifies more men as having AHS
# (about 10% -> 16%) and its association with PrEP is weaker (year-specific ratios about 0.5 before,
# 0.7-0.85 after; ahs_prep_evidence_by_year.csv). The model's AHS is adjusted to the old wording
# (scripts/01), so the primary target uses the old-wording cycles (2017-2021); the all-years
# estimate is a sensitivity (and the target for the unadjusted-wording fit). The by-year and
# wording-step models pool the median-55 states with state fixed effects.
#
# Inputs: results/01_weights/state_raked.rds (scripts/01),
#         results/02_inputs/state_screen/median55_states.csv (scripts/07)
# Outputs (results/02_inputs/):
#   ahs_prep_evidence.csv               old-wording cycles; one row per state, plus "POOLED" (primary)
#   ahs_prep_evidence_all_years.csv     all cycles; same layout (sensitivity)
#   ahs_prep_evidence_by_year.csv       pooled AHS effect by survey year: ratio and difference (pp)
#   ahs_prep_evidence_wording_step.csv  pooled AHS x (year trend + new wording) interaction
#
# Run from the repository root: Rscript scripts/08b_ahs_prep_evidence.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(survey)
})

output_dir <- "results/02_inputs"
old_wording_last_year <- 2021
confounders <- c("insur_bin", "agecat", "rdrace", "school4", "risk_behavior", "cycle")

states <- read_csv("results/02_inputs/state_screen/median55_states.csv", show_col_types = FALSE)$state
amis <- readRDS("results/01_weights/state_raked.rds")$variables %>%
  filter(state_calc %in% states) %>%
  group_by(state_calc, year_group) %>%
  mutate(weight = raked_weight * n()) %>%          # each state-period weighted to its sample size
  ungroup() %>%
  mutate(cycle = factor(survey_year), across(c(insur_bin, agecat, rdrace, school4), factor))

fit_state <- function(df) {
  df <- df %>% select(prep_used, stigma_ahs, all_of(confounders), weight) %>% drop_na()
  terms <- c("stigma_ahs", confounders[map_lgl(confounders, ~ n_distinct(df[[.x]]) >= 2)])
  design <- svydesign(ids = ~1, weights = ~weight, data = df)
  fit <- svyglm(reformulate(terms, "prep_used"), design = design, family = quasipoisson())
  tibble(n = nrow(df), n_ahs = sum(df$stigma_ahs == 1),
         log_apr = coef(fit)[["stigma_ahs"]], se = sqrt(vcov(fit)["stigma_ahs", "stigma_ahs"]))
}

# State estimates and their random-effects pooling (DerSimonian-Laird).
state_evidence <- function(data) {
  estimates <- data %>%
    group_by(state = state_calc) %>%
    group_modify(~ fit_state(.x)) %>%
    ungroup()
  w <- 1 / estimates$se^2
  fixed <- sum(w * estimates$log_apr) / sum(w)
  q <- sum(w * (estimates$log_apr - fixed)^2)
  tau2 <- max(0, (q - (nrow(estimates) - 1)) / (sum(w) - sum(w^2) / sum(w)))
  mu <- sum(estimates$log_apr / (estimates$se^2 + tau2)) / sum(1 / (estimates$se^2 + tau2))
  mu_se <- sqrt(1 / sum(1 / (estimates$se^2 + tau2)))
  message(sprintf("  random-effects mean aPR %.2f (95%% CI %.2f-%.2f); between-state SD of log aPR %.2f",
                  exp(mu), exp(mu - 1.96 * mu_se), exp(mu + 1.96 * mu_se), sqrt(tau2)))
  # Empirical-Bayes estimate: precision-weighted between the state estimate and the mean; its
  # posterior variance adds the mean's uncertainty through the weight on the mean.
  shrink <- if (tau2 > 0) (1 / tau2) / (1 / estimates$se^2 + 1 / tau2) else rep(1, nrow(estimates))
  estimates %>%
    mutate(log_apr_pooled = shrink * mu + (1 - shrink) * log_apr,
           se_pooled = sqrt(if (tau2 > 0) 1 / (1 / se^2 + 1 / tau2) + shrink^2 * mu_se^2 else rep(mu_se^2, n())),
           apr = exp(log_apr), apr_low = exp(log_apr - 1.96 * se), apr_high = exp(log_apr + 1.96 * se),
           apr_pooled = exp(log_apr_pooled)) %>%
    bind_rows(tibble(state = "POOLED", n = sum(estimates$n), n_ahs = sum(estimates$n_ahs), log_apr = mu, se = mu_se,
                     log_apr_pooled = mu, se_pooled = mu_se, apr = exp(mu), apr_low = exp(mu - 1.96 * mu_se),
                     apr_high = exp(mu + 1.96 * mu_se), apr_pooled = exp(mu)))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
message("Old-wording cycles (", paste(sort(unique(amis$survey_year[amis$survey_year <= old_wording_last_year])), collapse = ", "), "):")
write_csv(state_evidence(filter(amis, survey_year <= old_wording_last_year)), file.path(output_dir, "ahs_prep_evidence.csv"))
message("All cycles:")
write_csv(state_evidence(amis), file.path(output_dir, "ahs_prep_evidence_all_years.csv"))

# ---- Change over time and the wording step (pooled, state fixed effects) --------------------

pooled <- amis %>%
  mutate(state_calc = factor(state_calc), new_wording = survey_year > old_wording_last_year) %>%
  select(prep_used, stigma_ahs, survey_year, new_wording, all_of(confounders), state_calc, weight) %>%
  drop_na()
design <- svydesign(ids = ~1, weights = ~weight, data = pooled)
rhs <- paste(c(confounders, "state_calc"), collapse = " + ")
scales <- list(ratio = quasipoisson(), difference_pp = gaussian())
to_scale <- list(ratio = exp, difference_pp = function(x) 100 * x)

ahs_share <- svyby(~stigma_ahs, ~cycle, design, svymean)
by_year <- imap_dfr(scales, function(family, scale) {
  fit <- svyglm(as.formula(paste("prep_used ~ cycle:stigma_ahs +", rhs)), design = design, family = family)
  k <- grep("stigma_ahs", names(coef(fit)))
  b <- coef(fit)[k]
  se <- sqrt(diag(vcov(fit))[k])
  tr <- to_scale[[scale]]
  tibble(scale = scale, survey_year = as.integer(sub("cycle(\\d+):stigma_ahs", "\\1", names(b))),
         estimate = tr(b), low = tr(b - 1.96 * se), high = tr(b + 1.96 * se))
}) %>%
  left_join(tibble(survey_year = as.integer(as.character(ahs_share$cycle)), ahs_pct = 100 * ahs_share$stigma_ahs),
            by = "survey_year") %>%
  mutate(new_wording = survey_year > old_wording_last_year)
write_csv(by_year, file.path(output_dir, "ahs_prep_evidence_by_year.csv"))

step <- imap_dfr(scales, function(family, scale) {
  fit <- svyglm(as.formula(paste("prep_used ~ stigma_ahs * (I(survey_year - 2020) + new_wording) +", rhs)),
                design = design, family = family)
  terms <- c(per_year = "stigma_ahs:I(survey_year - 2020)", new_wording = "stigma_ahs:new_wordingTRUE")
  b <- coef(fit)[terms]
  se <- sqrt(diag(vcov(fit))[terms])
  tr <- to_scale[[scale]]
  tibble(scale = scale, term = names(terms), estimate = tr(b), low = tr(b - 1.96 * se), high = tr(b + 1.96 * se),
         p_value = 2 * pnorm(-abs(b / se)))
})
write_csv(step, file.path(output_dir, "ahs_prep_evidence_wording_step.csv"))
print(by_year, n = Inf)
print(step)
