# Inputs for the PrEP diffusion model (config/specifications.py, prep_ceiling "unit" and prep_diffusion): each
# unit's PrEP ceiling and the prior for the growth rate of PrEP toward it.
#
# 1. Ceiling: the share of men without anticipated healthcare stigma (AHS) who use PrEP or would be willing to.
# Ceiling = users + (1 - users) x willing non-users, among respondents without AHS, AMIS 2022-2024, state
# weights. State estimates are shrunk toward the primary-state value on the logit scale (empirical Bayes:
# sampling variance 1 / (n_eff K (1 - K)), Kish effective n; between-state variance by method of moments, primary
# states). States without an estimate strictly between 0 and 1 take the primary-state value.
# Divisions and the nation are respondent-count-weighted averages of their states' shrunk ceilings, as the
# model inputs average states into divisions (data/workbook.py).
#
# 2. Growth-rate prior: in the model logit(P0 / ceiling) rises by the unit's PrEP drift each two-year step, P0 being
# PrEP among men without AHS. For each primary state, P0 = P / (1 - A (1 - r)) in each AMIS cycle 2017-2024 (P, A
# weighted state values; A wording-adjusted from 2022 as in scripts/1_data/03; r the state's prevalence ratio,
# scripts/2_evidence/01) and logit(P0 / ceiling) is regressed on year (weights 1 / sampling variance, delta method,
# Kish effective n). The prior mean is the random-effects mean of the state slopes (DerSimonian-Laird), per
# two-year step; the between-state SD is reported (the model keeps a weak prior SD so states follow their data).
#
# Inputs: results/01_weights/state_raked.rds, results/01_weights/state_model_inputs.xlsx (states sheet),
#         results/01_weights/ahs_wording_shift.csv, results/02_inputs/ahs_prep_evidence.csv,
#         results/02_inputs/state_screen/primary_states.csv, inputs/restricted/amis_raw (willing_prep, 2022-2024)
# Outputs: results/02_inputs/prep_ceiling.csv (unit_id, unit_kind, ceiling, and for states the unshrunk estimate,
#          effective n and shrinkage weight); results/02_inputs/prep_growth.csv (state slopes and, in the row
#          "prior", the prior mean and SD). Aggregates only.
#
# Run from the repository root: Rscript scripts/2_evidence/04_prep_diffusion.R

suppressPackageStartupMessages({
  library(haven)
  library(readxl)
  library(tidyverse)
})

output_path <- "results/02_inputs/prep_ceiling.csv"
growth_path <- "results/02_inputs/prep_growth.csv"
step_years <- 2
raw_files <- c(amis_22 = "sitq2022_analysis_fulldata_noid.dta", amis_23 = "sitq2023_analysis_final_noid.dta",
               amis_24 = "sitq2024_analysis_final_NOID.dta")

# Willingness (1 willing, 0 not; other codes missing), matched to the analytic file by row ID
willing <- imap_dfr(raw_files, function(file, survey) {
  path <- file.path("inputs/restricted/amis_raw", file)
  column <- names(read_dta(path, n_max = 0))
  d <- read_dta(path, col_select = all_of(column[tolower(column) == "willing_prep"]))
  tibble(rowID = paste(survey, seq_len(nrow(d)), sep = "_"), willing = as.numeric(zap_labels(d[[1]])))
}) %>% mutate(willing = if_else(willing %in% c(0, 1), willing, NA_real_))

men <- readRDS("results/01_weights/state_raked.rds")$variables %>%
  filter(survey_year >= 2022, stigma_ahs == 0, !is.na(prep_used)) %>%
  left_join(willing, by = "rowID") %>% mutate(w = raked_weight)

ceiling_of <- function(d) {
  users <- weighted.mean(d$prep_used, d$w)
  nonusers <- filter(d, prep_used == 0, !is.na(willing))
  tibble(users = users, willing_nonusers = weighted.mean(nonusers$willing, nonusers$w),
         n_eff = sum(d$w)^2 / sum(d$w^2)) %>%
    mutate(ceiling_raw = users + (1 - users) * willing_nonusers)
}

primary <- read_csv("results/02_inputs/state_screen/primary_states.csv", show_col_types = FALSE)$state
national <- ceiling_of(filter(men, state_calc %in% primary))$ceiling_raw

# States: empirical-Bayes shrinkage of logit(ceiling) toward the primary-state value
all_states <- read_excel("results/01_weights/state_model_inputs.xlsx", sheet = "states")$state_calc
states <- men %>% group_by(unit_id = state_calc) %>% group_modify(~ ceiling_of(.x)) %>% ungroup() %>%
  right_join(tibble(unit_id = all_states), by = "unit_id") %>%
  mutate(y = qlogis(ceiling_raw), v = 1 / (n_eff * ceiling_raw * (1 - ceiling_raw)))
mu <- qlogis(national)
tau2 <- with(filter(states, unit_id %in% primary), max(0, mean((y - mu)^2 - v)))
states <- states %>%
  mutate(shrinkage = if_else(is.finite(v), v / (v + tau2), 1),
         ceiling = if_else(is.finite(v), plogis(shrinkage * mu + (1 - shrinkage) * y), national), unit_kind = "state")

# Divisions and nation: respondent-count-weighted averages of state ceilings (data/workbook.py)
counts <- read_excel("results/01_weights/state_model_inputs.xlsx", sheet = "states") %>%
  select(unit_id = state_calc, n, Census_div) %>% inner_join(select(states, unit_id, ceiling), by = "unit_id")
divisions <- counts %>% group_by(Census_div) %>% summarise(ceiling = weighted.mean(ceiling, n)) %>%
  transmute(unit_id = paste0("div_", Census_div), unit_kind = "division", ceiling)
nation <- tibble(unit_id = "USA", unit_kind = "nation", ceiling = weighted.mean(counts$ceiling, counts$n))

out <- bind_rows(nation, divisions,
                 select(states, unit_id, unit_kind, ceiling, ceiling_raw, n_eff, shrinkage))
write_csv(out, output_path)

cat(sprintf("Primary-state ceiling (pooled, unshrunk): %.3f; between-state SD (logit): %.3f\n", national, sqrt(tau2)))
print(out %>% filter(unit_kind != "state" | unit_id %in% primary) %>% mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
        as.data.frame(), row.names = FALSE)

# ---- 2. Growth-rate prior ---------------------------------------------------------------------------
shift <- read_csv("results/01_weights/ahs_wording_shift.csv", show_col_types = FALSE)$shift_log_odds
ratio <- read_csv("results/02_inputs/ahs_prep_evidence.csv", show_col_types = FALSE) %>% select(state, r = apr_pooled)
annual <- readRDS("results/01_weights/state_raked.rds")$variables %>%
  filter(state_calc %in% primary, !is.na(prep_used), !is.na(stigma_ahs)) %>%
  group_by(state = state_calc, year = survey_year) %>%
  summarise(P = weighted.mean(prep_used, raked_weight), A = weighted.mean(stigma_ahs, raked_weight),
            n_eff = sum(raked_weight)^2 / sum(raked_weight^2), .groups = "drop") %>%
  left_join(ratio, by = "state") %>% left_join(select(states, state = unit_id, ceiling), by = "state") %>%
  mutate(A = if_else(year >= 2022, plogis(qlogis(A) - shift), A), m = 1 - A * (1 - r), q = P / m / ceiling,
         z = qlogis(q), se_z = sqrt(P * (1 - P) / n_eff) / (m * ceiling * q * (1 - q)))
slopes <- annual %>% group_by(state) %>% group_modify(function(g, k) {
  fit <- summary(lm(z ~ I(year - 2020), data = g, weights = 1 / se_z^2))$coefficients
  tibble(slope = step_years * fit[2, 1], se = step_years * fit[2, 2], n_years = nrow(g))
}) %>% ungroup()
w <- 1 / slopes$se^2
fixed_mean <- sum(w * slopes$slope) / sum(w)
q_stat <- sum(w * (slopes$slope - fixed_mean)^2)
tau2_growth <- max(0, (q_stat - (nrow(slopes) - 1)) / (sum(w) - sum(w^2) / sum(w)))
w_re <- 1 / (slopes$se^2 + tau2_growth)
prior <- tibble(state = "prior", slope = sum(w_re * slopes$slope) / sum(w_re), se = sqrt(tau2_growth), n_years = NA)
write_csv(bind_rows(slopes, prior), growth_path)
cat(sprintf("\nPrEP growth per %d-year step, logit(P0 / ceiling): prior mean %.3f, between-state SD %.3f\n",
            step_years, prior$slope, prior$se))
print(slopes %>% arrange(slope) %>% mutate(across(where(is.numeric), ~ round(.x, 3))) %>% as.data.frame(), row.names = FALSE)
