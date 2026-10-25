# Model-vs-evidence diagnostic for the PrEP effect of anticipated healthcare stigma (AHS), by
# state: the individual-level expected value and ceiling, compared with the dynamic model.
#
# The access-adjusted state prevalence ratios (aPR) come from scripts/08b_ahs_prep_evidence.R
# (survey-weighted Poisson, confounders only, pooled by random effects); they also centre the
# model's AHS -> PrEP priors, so agreement here shows how far each state's dynamics moved away
# from that centre.
#
# Per state, for removing an AHS share s:
#   expected gain = s x PrEP among men without AHS x (1 - aPR)   adjusted association
#   ceiling       = s x PrEP among men without AHS               men with AHS use no PrEP
# compared with the model's AHS-scenario PrEP gain in 2036 (s = the scenario's AHS reduction;
# PrEP among men without AHS from the baseline projection).
#
# Inputs: results/02_inputs/ahs_prep_evidence.csv (scripts/08b),
#         results/01_weights/state_raked.rds (scripts/01),
#         results/02_inputs/state_screen/primary_states.csv (scripts/07),
#         results/04_scenarios/primary/scenario_endpoint_sem_draws.csv (scripts/11)
# Output: results/07_structural/ahs_prep_individual_bounds.csv
#
# Run from the repository root: Rscript scripts/13c_ahs_prep_individual_bounds.R

suppressPackageStartupMessages({
  library(tidyverse)
})

output_csv <- "results/07_structural/ahs_prep_individual_bounds.csv"

primary <- read_csv("results/02_inputs/state_screen/primary_states.csv", show_col_types = FALSE)$state
estimates <- read_csv("results/02_inputs/ahs_prep_evidence.csv", show_col_types = FALSE) %>%
  filter(state %in% primary)
amis <- readRDS("results/01_weights/state_raked.rds")$variables %>%
  filter(state_calc %in% primary) %>%
  group_by(state_calc, year_group) %>%
  mutate(weight = raked_weight * n()) %>%
  ungroup()

# Observed 2023-2024 AHS and PrEP among men without AHS (state weights).
observed <- amis %>%
  filter(year_group == "2023-2024", !is.na(stigma_ahs), !is.na(prep_used)) %>%
  group_by(state = state_calc) %>%
  summarise(ahs_2024 = weighted.mean(stigma_ahs, weight),
            prep_no_ahs_2024 = weighted.mean(prep_used[stigma_ahs == 0], weight[stigma_ahs == 0]),
            prep_ahs_2024 = weighted.mean(prep_used[stigma_ahs == 1], weight[stigma_ahs == 1]),
            .groups = "drop")

# Model: AHS-scenario reduction in AHS and gain in PrEP, 2036 (state medians across draws).
model <- read_csv("results/04_scenarios/primary/scenario_endpoint_sem_draws.csv", show_col_types = FALSE) %>%
  filter(scenario == "s1_reduce_ahs", year == 2036, variable %in% c("stigma_ahs", "prep_used")) %>%
  group_by(state = unit_id, variable) %>%
  summarise(baseline = median(baseline), delta = median(delta), .groups = "drop") %>%
  pivot_wider(names_from = variable, values_from = c(baseline, delta)) %>%
  transmute(state, ahs_baseline_2036 = baseline_stigma_ahs, ahs_cut_2036 = -delta_stigma_ahs,
            prep_baseline_2036 = baseline_prep_used,
            model_prep_gain_2036 = delta_prep_used)

bounds <- estimates %>%
  left_join(observed, by = "state") %>%
  left_join(model, by = "state") %>%
  mutate(
    # PrEP among men without AHS in the 2036 baseline: average = p0 x (1 - AHS share x (1 - aPR))
    prep_no_ahs_2036 = prep_baseline_2036 / (1 - ahs_baseline_2036 * (1 - apr_pooled)),
    expected_gain_2036 = ahs_cut_2036 * prep_no_ahs_2036 * (1 - apr_pooled),
    ceiling_2036 = ahs_cut_2036 * prep_no_ahs_2036,
    model_vs_expected = model_prep_gain_2036 / expected_gain_2036,
    model_vs_ceiling = model_prep_gain_2036 / ceiling_2036
  ) %>%
  arrange(desc(model_vs_ceiling))

write_csv(bounds, output_csv)
print(bounds %>% transmute(state, aPR = round(apr, 2), aPR_pooled = round(apr_pooled, 2),
                           expected_pp = round(100 * expected_gain_2036, 2), ceiling_pp = round(100 * ceiling_2036, 2),
                           model_pp = round(100 * model_prep_gain_2036, 2),
                           model_vs_expected = round(model_vs_expected, 1), model_vs_ceiling = round(model_vs_ceiling, 2)),
      n = Inf)
