# Weighted annual trajectories of the eight measures for six demographic groups,
# with joint respondent-bootstrap replicates.
#
# Groups: Black non-Hispanic, Hispanic/Latino, White non-Hispanic; ages <=24,
# 25-34, >=35. Each annual estimate uses the national raking weights (scripts/1_data/07)
# and is smoothed on the logit scale with its adjacent observed waves (weights:
# focal wave 1, neighbours 0.5, each times the cell's Kish n_eff), so endpoint
# years borrow from one neighbour. There is no 2020 wave. Respondents are
# resampled within state x survey-year strata with weights held fixed.
#
# Outputs (results/02_inputs/group_bootstrap/):
#   group_trajectories.csv      point estimates (scripts/5_demographic/03)
#   group_draws.csv.gz          bootstrap replicates (scripts/5_demographic/03)
#
# Run from the repository root: Rscript scripts/5_demographic/02_group_bootstrap.R [n_boot]

suppressPackageStartupMessages({
  library(tidyverse)
  library(survey)
})


args <- commandArgs(trailingOnly = TRUE)
n_boot <- if (length(args) > 0) as.integer(args[1]) else 500L
seed <- 20261006
eps <- 1e-6

output_points <- "results/02_inputs/group_bootstrap/group_trajectories.csv"
output_draws <- "results/02_inputs/group_bootstrap/group_draws.csv.gz"
dir.create("results/02_inputs/group_bootstrap", recursive = TRUE, showWarnings = FALSE)

model_vars <- c("stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "risk_behavior", "prep_used", "hivtest12")
groups <- tribble(
  ~population_id, ~column, ~value,
  "black_nh", "race_ethnicity", "Black, non-Hispanic",
  "hispanic_latino", "race_ethnicity", "Hispanic/Latino",
  "white_nh", "race_ethnicity", "White, non-Hispanic",
  "age_under_24", "age_group", "<=24",
  "age_25_34", "age_group", "25-34",
  "age_35_plus", "age_group", "35+"
)

national <- readRDS("results/01_weights/national_raked.rds")
amis <- national$variables %>%
  transmute(
    state_calc, Year = as.integer(survey_year), race_ethnicity = as.character(race_ethnicity),
    age_group = as.character(age_group), weight = national_weight,
    across(all_of(model_vars))
  )
strata <- split(seq_len(nrow(amis)), interaction(amis$state_calc, amis$Year, drop = TRUE, lex.order = TRUE))

# Kish n_eff-weighted logit smooth over adjacent observed waves.
smooth <- function(value, n_eff) {
  z <- qlogis(pmin(pmax(value, eps), 1 - eps))
  smoothed <- map_dbl(seq_along(z), function(i) {
    nb <- intersect(seq_along(z), i + -1:1)
    w <- ifelse(nb == i, 1, 0.5) * n_eff[nb]
    sum(z[nb] * w) / sum(w)
  })
  pmin(pmax(plogis(smoothed), eps), 1 - eps)
}

trajectories <- function(data) {
  pmap_dfr(groups, function(population_id, column, value) {
    data[data[[column]] == value, ] %>%
      pivot_longer(all_of(model_vars), names_to = "variable", values_to = "y") %>%
      filter(!is.na(y)) %>%
      group_by(variable, Year) %>%
      summarise(value = sum(y * weight) / sum(weight), n_eff = sum(weight)^2 / sum(weight^2), .groups = "drop_last") %>%
      arrange(Year, .by_group = TRUE) %>%
      mutate(value = smooth(value, n_eff)) %>%
      ungroup() %>%
      mutate(population_id = population_id, .before = 1)
  })
}

points <- trajectories(amis)

set.seed(seed)
draws <- map_dfr(seq_len(n_boot), function(b) {
  if (b %% 50 == 0) message("Replicate ", b, " of ", n_boot)
  rows <- unlist(lapply(strata, function(idx) idx[sample.int(length(idx), replace = TRUE)]), use.names = FALSE)
  trajectories(amis[rows, ]) %>% mutate(draw = b, .before = 1)
})
stopifnot(nrow(draws) == n_boot * nrow(points))

write_csv(points, output_points)
write_csv(draws, gzfile(output_draws))
message("Wrote ", n_boot, " replicates for ", nrow(groups), " groups to ", dirname(output_points))
