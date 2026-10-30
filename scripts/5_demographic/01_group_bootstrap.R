# Weighted annual trajectories of the eight measures for six demographic groups,
# with joint respondent-bootstrap replicates.
#
# Groups: Black non-Hispanic, Hispanic/Latino, White non-Hispanic; ages <=24,
# 25-34, >=35. Each annual estimate is the weighted proportion in that wave, using the
# national raking weights (scripts/1_data/07); n_eff is the cell's Kish effective sample
# size. There is no 2020 wave. Respondents are resampled within state x survey-year
# strata with weights held fixed.
#
# Outputs (results/02_inputs/group_bootstrap/):
#   group_trajectories.csv      point estimates (scripts/5_demographic/02)
#   group_draws.csv.gz          bootstrap replicates (scripts/5_demographic/02)
#
# Run from the repository root: Rscript scripts/5_demographic/01_group_bootstrap.R [n_boot]

suppressPackageStartupMessages({
  library(tidyverse)
})

args <- commandArgs(trailingOnly = TRUE)
n_boot <- if (length(args) > 0) as.integer(args[1]) else 500L
seed <- 20261006
if (is.na(n_boot) || n_boot < 1) stop("n_boot must be a positive integer")

output_points <- "results/02_inputs/group_bootstrap/group_trajectories.csv"
output_draws <- "results/02_inputs/group_bootstrap/group_draws.csv.gz"
dir.create("results/02_inputs/group_bootstrap", recursive = TRUE, showWarnings = FALSE)

model_vars <- c(
  "stigma_ahs", "stigma_gss", "stigma_family", "out_gid",
  "seehcp", "risk_behavior", "prep_used", "hivtest12"
)

# Each row defines one marginal population. Race/ethnicity and age analyses are
# separate, so respondents can contribute to one group of each type.
groups <- tribble(
  ~population_id, ~group_variable, ~group_value,
  "black_nh", "race_ethnicity", "Black, non-Hispanic",
  "hispanic_latino", "race_ethnicity", "Hispanic/Latino",
  "white_nh", "race_ethnicity", "White, non-Hispanic",
  "age_under_24", "age_group", "<=24",
  "age_25_34", "age_group", "25-34",
  "age_35_plus", "age_group", "35+"
)

# ---- Respondents ---------------------------------------------------------------

amis <- readRDS("results/01_weights/national_raked.rds")$variables %>%
  transmute(
    state_calc,
    Year = as.integer(survey_year),
    race_ethnicity = as.character(race_ethnicity),
    age_group = as.character(age_group),
    weight = national_weight,
    across(all_of(model_vars))
  )

# Keep the original number of respondents in every state x survey-year cell.
# A bootstrap replicate samples respondent row numbers within each cell.
stratum_indices <- split(
  seq_len(nrow(amis)),
  interaction(amis$state_calc, amis$Year, drop = TRUE, lex.order = TRUE)
)

# ---- Estimator -----------------------------------------------------------------

estimate_trajectories <- function(data) {

  # Reshape once, then use the same long dataset for all six populations.
  long_data <- data %>%
    pivot_longer(
      cols = all_of(model_vars),
      names_to = "variable",
      values_to = "response"
    )

  pmap_dfr(groups, function(population_id, group_variable, group_value) {
    long_data %>%
      filter(.data[[group_variable]] == group_value, !is.na(response)) %>%
      group_by(Year, variable) %>%
      summarise(
        # Survey-weighted prevalence in this population, wave and measure.
        value = weighted.mean(response, weight),

        # Kish effective sample size associated with that weighted estimate.
        n_eff = sum(weight)^2 / sum(weight^2),
        .groups = "drop"
      ) %>%
      mutate(population_id = population_id, .before = 1)
  })
}

point_estimates <- estimate_trajectories(amis)

# ---- Joint respondent bootstrap ------------------------------------------------

resample_within_strata <- function(data, indices) {
  sampled_rows <- map(
    indices,
    ~ sample(.x, size = length(.x), replace = TRUE)
  ) %>%
    unlist(use.names = FALSE)

  data[sampled_rows, , drop = FALSE]
}

set.seed(seed)
bootstrap_draws <- map_dfr(seq_len(n_boot), function(draw) {
  if (draw %% 50 == 0) message("Replicate ", draw, " of ", n_boot)

  resampled_amis <- resample_within_strata(amis, stratum_indices)
  estimate_trajectories(resampled_amis) %>%
    mutate(draw = draw, .before = 1)
})

expected_rows <- n_boot * nrow(point_estimates)
if (nrow(bootstrap_draws) != expected_rows) {
  stop(
    "Bootstrap output is incomplete: expected ", expected_rows,
    " rows but obtained ", nrow(bootstrap_draws)
  )
}

# ---- Outputs -------------------------------------------------------------------

write_csv(point_estimates, output_points)
write_csv(bootstrap_draws, gzfile(output_draws))
message("Wrote ", n_boot, " replicates for ", nrow(groups), " groups to ", dirname(output_points))
