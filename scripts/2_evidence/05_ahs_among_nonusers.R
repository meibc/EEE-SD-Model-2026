# Who remains without PrEP: anticipated healthcare stigma (AHS) among PrEP non-users, by AMIS cycle.
#
# As PrEP use rises, a lower PrEP prevalence among men with AHS (prevalence ratio r < 1; scripts/2_evidence/01)
# makes them a growing share of the men still not using PrEP. This script measures that composition in each AMIS
# cycle, the observed counterpart of the projection (scripts/4_results/figS_nonusers.py):
#   share among non-users  Pr(AHS | no PrEP)
#   ratio                  Pr(AHS | no PrEP) / Pr(AHS), how overrepresented men with AHS are among non-users
# AHS as measured: the items were reworded in 2022 (scripts/2_evidence/02), which shifts its level from 2022 on;
# the ratio compares AHS within a cycle, so it is less affected than the level.
#
# National weights (scripts/1_data/07), respondents with both AHS and PrEP use observed. Intervals: survey-weighted
# 95%, logit for the proportions, delta method for the ratio.
#
# Output: results/02_inputs/ahs_among_nonusers.csv (aggregates only).
# Run from the repository root: Rscript scripts/2_evidence/05_ahs_among_nonusers.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(survey)
})

output_csv <- "results/02_inputs/ahs_among_nonusers.csv"

national <- readRDS("results/01_weights/national_raked.rds")$variables %>%
  select(survey_year, stigma_ahs, prep_used, national_weight) %>%
  filter(!is.na(stigma_ahs), !is.na(prep_used)) %>%
  mutate(nonuser = as.numeric(prep_used == 0), ahs_nonuser = stigma_ahs * nonuser, ahs_user = stigma_ahs * (1 - nonuser))

proportion <- function(design, variable) {
  e <- svyciprop(reformulate(variable), design, method = "logit")
  c(value = as.numeric(coef(e)), low = unname(attr(e, "ci")[1]), high = unname(attr(e, "ci")[2]))
}

by_year <- map_dfr(sort(unique(national$survey_year)), function(year) {
  d <- svydesign(ids = ~1, weights = ~national_weight, data = filter(national, survey_year == year) %>% mutate(one = 1))
  a_all <- proportion(d, "stigma_ahs")
  p_all <- proportion(d, "prep_used")
  a_non <- proportion(subset(d, nonuser == 1), "stigma_ahs")
  a_use <- proportion(subset(d, nonuser == 0), "stigma_ahs")
  # ratio = [total(AHS and non-user) / total(non-user)] / [total(AHS) / total(everyone)]
  totals <- svytotal(~ahs_nonuser + nonuser + stigma_ahs + one, d)
  r <- svycontrast(totals, quote((ahs_nonuser / nonuser) / (stigma_ahs / one)))
  r_est <- as.numeric(coef(r))
  r_se <- as.numeric(SE(r))
  tibble(survey_year = year, respondents = nrow(d$variables),
         prep = p_all[["value"]], ahs = a_all[["value"]], ahs_low = a_all[["low"]], ahs_high = a_all[["high"]],
         ahs_nonusers = a_non[["value"]], ahs_nonusers_low = a_non[["low"]], ahs_nonusers_high = a_non[["high"]],
         ahs_users = a_use[["value"]],
         ratio = r_est, ratio_low = r_est - 1.96 * r_se, ratio_high = r_est + 1.96 * r_se)
})

write_csv(by_year, output_csv)
print(by_year %>% mutate(across(c(prep, starts_with("ahs")), ~ round(100 * .x, 1)), across(starts_with("ratio"), ~ round(.x, 2))))
