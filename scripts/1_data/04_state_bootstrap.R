# Joint respondent bootstrap for the eight state dynamic-model inputs.
#
# Resamples respondents with replacement within each state x period stratum,
# carrying each respondent's fixed state-period raking weight, outcomes, and
# missingness jointly through all eight measures. Each replicate reconstructs:
#   1. state estimates as respondent-level raking-weighted means;
#   2. Census-division estimates as respondent-count-weighted means of states;
#   3. the national estimate as an MSM-population-weighted mean of states.
#
# One bootstrap replicate consists of one complete respondent dataset: every
# state-period stratum is resampled once, and all state, division, and national
# estimates are then calculated from that dataset. Replicate cell estimates are
# used directly as dynamic-model input draws. Raking weights are held fixed
# rather than re-estimated within replicates. Periods are sampled independently
# because AMIS is repeated cross-sectional, not longitudinal.
#
# AHS state estimates are adjusted for the AMIS 2022 wording change exactly as in scripts/1_data/03
# (shift read from results/01_weights/ahs_wording_shift.csv); AHS_WORDING=unadjusted writes the
# unadjusted sensitivity version to results/02_inputs/state_bootstrap_ahs_unadjusted/.
#
# Outputs (results/02_inputs/state_bootstrap/):
#   state_input_draws.csv.gz  draw x unit x period x measure estimates (scripts/3_model/01)
#   cell_precision.csv        original estimate, Kish n_eff, bootstrap mean/sd per cell (scripts/1_data/05)
#   division_fallback_audit.csv
#                             state cells replaced by the matching division draw
#
# Run from the repository root: Rscript scripts/1_data/04_state_bootstrap.R [n_boot]

suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
})


args <- commandArgs(trailingOnly = TRUE)
n_boot <- if (length(args) > 0) as.integer(args[1]) else 500L
seed <- 20261002
eps <- 1e-6

# AHS wording adjustment as in scripts/1_data/03 (AHS_WORDING=unadjusted writes the sensitivity version).
ahs_wording <- Sys.getenv("AHS_WORDING", "adjusted")
stopifnot(ahs_wording %in% c("adjusted", "unadjusted"))
suffix <- if (ahs_wording == "adjusted") "" else "_ahs_unadjusted"
input_rds <- "results/01_weights/state_raked.rds"
workbook <- paste0("results/01_weights/state_model_inputs", suffix, ".xlsx")
output_dir <- paste0("results/02_inputs/state_bootstrap", suffix)
output_draws <- file.path(output_dir, "state_input_draws.csv.gz")
output_cells <- file.path(output_dir, "cell_precision.csv")
output_fallback_audit <- file.path(output_dir, "division_fallback_audit.csv")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
wording_shift <- if (ahs_wording == "adjusted") {
  read_csv("results/01_weights/ahs_wording_shift.csv", show_col_types = FALSE)$shift_log_odds
} else {
  0
}

model_vars <- c(
  "stigma_ahs",
  "stigma_gss",
  "stigma_family",
  "out_gid",
  "seehcp",
  "risk_behavior",
  "prep_used",
  "hivtest12"
)

periods <- c(
  "2017-2018" = 2018L,
  "2019-2020" = 2020L,
  "2021-2022" = 2022L,
  "2023-2024" = 2024L
)

clamp <- function(p) pmin(pmax(p, eps), 1 - eps)

# ---- Inputs --------------------------------------------------------------------

# The RDS contains the state-weighted survey design produced by script 01.
# Extract its respondent table so each bootstrap replicate carries forward the
# already-estimated state-period raking weights.
state_design <- readRDS(input_rds)

amis <- state_design$variables %>%
  transmute(
    state_calc,
    census_division = as.integer(`_censdiv`),
    year_group,
    year = periods[year_group],
    new_wording = survey_year >= 2022,
    raked_weight,
    across(all_of(model_vars))
  ) %>%
  mutate(
    stratum = interaction(
      state_calc,
      year_group,
      drop = TRUE,
      lex.order = TRUE
    )
  )

states <- read_excel(workbook, sheet = "states") %>%
  transmute(
    state_calc,
    census_division = as.integer(Census_div),
    state_n = n
  ) %>%
  inner_join(
    read_csv(
      "inputs/msm_state_population.csv",
      show_col_types = FALSE
    ) %>%
      transmute(state_calc = state, msm_pop = rounded),
    by = "state_calc"
  )

stopifnot(setequal(states$state_calc, unique(amis$state_calc)))

# ---- Estimator -----------------------------------------------------------------

# Reconstruct the complete geographic hierarchy from one respondent dataset.
estimate <- function(data) {
  # State estimates preserve respondent-level raking weights and calculate
  # Kish effective sample size separately for every measure-specific cell.
  state <- data %>%
    pivot_longer(
      all_of(model_vars),
      names_to = "variable",
      values_to = "response"
    ) %>%
    filter(!is.na(response)) %>%
    group_by(unit = state_calc, census_division, year_group, year, variable) %>%
    summarise(
      prop = sum(response * raked_weight) / sum(raked_weight),
      n_eff = sum(raked_weight)^2 / sum(raked_weight^2),
      share_new = sum(new_wording * raked_weight) / sum(raked_weight),
      .groups = "drop"
    ) %>%
    # Move AHS back by the wording shift times the share of respondents with the new wording.
    mutate(prop = if_else(variable == "stigma_ahs", plogis(qlogis(prop) - wording_shift * share_new), prop)) %>%
    select(-share_new) %>%
    left_join(
      states %>% select(-census_division),
      by = c("unit" = "state_calc")
    )

  # Division estimates weight state estimates by the total retained respondent
  # count recorded in the model-input workbook.
  division <- state %>%
    group_by(census_division, year_group, year, variable) %>%
    summarise(
      prop = weighted.mean(clamp(prop), state_n),
      .groups = "drop"
    ) %>%
    mutate(
      unit = paste0("div_", census_division),
      geography_type = "division_model",
      value_source = "division_estimate"
    )

  # National estimates weight state estimates by estimated state MSM population.
  nation <- state %>%
    group_by(year_group, year, variable) %>%
    summarise(
      prop = weighted.mean(prop, msm_pop),
      .groups = "drop"
    ) %>%
    mutate(
      unit = "USA",
      geography_type = "nation_model",
      value_source = "national_estimate"
    )

  # Complete the expected state-period-measure grid. Cells without a direct
  # state estimate inherit the matching division estimate from the same
  # bootstrap respondent dataset. Keeping this rule here ensures that the
  # statistical preparation is complete before the estimates enter Python.
  state_grid <- crossing(
    unit = states$state_calc,
    year_group = names(periods),
    variable = model_vars
  ) %>%
    mutate(year = unname(periods[year_group])) %>%
    left_join(
      states %>% rename(unit = state_calc),
      by = "unit"
    )

  completed_state <- state_grid %>%
    left_join(
      state %>% select(unit, year_group, year, variable, prop, n_eff),
      by = c("unit", "year_group", "year", "variable")
    ) %>%
    left_join(
      division %>%
        select(census_division, year_group, year, variable, division_prop = prop),
      by = c("census_division", "year_group", "year", "variable")
    ) %>%
    mutate(
      value_source = if_else(
        is.na(prop),
        "division_fallback",
        "state_estimate"
      ),
      prop = coalesce(prop, division_prop),
      geography_type = "state"
    )

  if (anyNA(completed_state$prop)) {
    unavailable <- completed_state %>%
      filter(is.na(prop)) %>%
      select(unit, year_group, variable, census_division)
    stop(
      "Division estimates were unavailable for one or more state cells:\n",
      paste(capture.output(print(unavailable)), collapse = "\n")
    )
  }

  bind_rows(completed_state, division, nation) %>%
    select(
      unit,
      geography_type,
      year_group,
      year,
      variable,
      prop,
      n_eff,
      value_source
    )
}

# ---- Original estimates and reproduction check --------------------------------

original <- estimate(amis)

trends <- read_excel(workbook, sheet = "trends") %>%
  transmute(
    unit = Geography,
    year = as.integer(Year),
    variable = Indicator,
    workbook = Value
  )

reproduction <- original %>%
  filter(
    geography_type == "nation_model" |
      value_source == "state_estimate"
  ) %>%
  inner_join(trends, by = c("unit", "year", "variable"))

expected_reproduction_rows <- nrow(trends)

stopifnot(nrow(reproduction) == expected_reproduction_rows)
stopifnot(max(abs(reproduction$prop - reproduction$workbook)) < 1e-10)

# ---- Bootstrap -----------------------------------------------------------------

# Create one complete bootstrap respondent dataset. Sampling occurs separately
# within each state-period stratum, and each stratum retains its original size.
# Complete rows are selected, so a respondent's outcomes, missingness, and fixed
# raking weight always travel together.
resample <- function(data) {
  # This is a named list of original data-row numbers. For example, the element
  # named "CA.2021-2022" contains the row numbers of all respondents in that
  # state-period stratum; it does not contain a separate copy of their data.
  stratum_rows <- split(
    seq_len(nrow(data)),
    data$stratum,
    drop = TRUE
  )

  # Within each list element, sample as many row positions as the stratum
  # originally contained. Replacement permits duplicate respondents and leaves
  # some original respondents unselected. unlist() then combines the sampled
  # row numbers from all strata into one index for the complete bootstrap sample.
  sampled_rows <- unlist(
    lapply(
      stratum_rows,
      function(row_index) {
        row_index[sample.int(length(row_index), replace = TRUE)]
      }
    ),
    use.names = FALSE
  )

  data[sampled_rows, ]
}

set.seed(seed)

# Repeat the complete resampling process n_boot times. Each replicate produces
# one estimate for every geography-period-measure cell. With 500 replicates,
# 49 states + 9 divisions + 1 nation, 4 periods, and 8 measures, draws contains
# 500 x 59 x 4 x 8 = 944,000 rows.
draws <- map_dfr(
  seq_len(n_boot),
  function(bootstrap_draw) {
    if (bootstrap_draw %% 50 == 0) {
      message("Replicate ", bootstrap_draw, " of ", n_boot)
    }

    estimate(resample(amis)) %>%
      mutate(draw = bootstrap_draw, .before = 1)
  }
) %>%
  mutate(model_draw = clamp(prop)) %>%
  select(
    draw,
    unit,
    geography_type,
    year_group,
    year,
    variable,
    model_draw,
    value_source
  )

# Collapse the replicate dimension. Each output row is one
# geography-period-measure cell containing its bootstrap mean, bootstrap
# standard deviation, and estimate from the original respondent dataset.
# With complete data, cells contains 59 x 4 x 8 = 1,888 rows.
cells <- draws %>%
  group_by(unit, geography_type, year_group, year, variable) %>%
  summarise(
    bootstrap_mean = mean(model_draw),
    bootstrap_sd = sd(model_draw),
    division_fallback_share = mean(value_source == "division_fallback"),
    .groups = "drop"
  ) %>%
  left_join(
    original %>%
      rename(
        original_estimate = prop,
        original_value_source = value_source
      ),
    by = c(
      "unit",
      "geography_type",
      "year_group",
      "year",
      "variable"
    )
  )

# Sorting changes only output order; it does not alter any estimates.
ordered_draws <- draws %>%
  arrange(draw, geography_type, unit, year, match(variable, model_vars))

fallback_audit <- draws %>%
  filter(geography_type == "state") %>%
  group_by(unit, year_group, year, variable) %>%
  summarise(
    n_draws = n(),
    n_division_fallback = sum(value_source == "division_fallback"),
    share_division_fallback = mean(value_source == "division_fallback"),
    .groups = "drop"
  )

write_csv(ordered_draws, gzfile(output_draws))
write_csv(cells, output_cells)
write_csv(fallback_audit, output_fallback_audit)

maximum_centering_difference <- max(
  abs(cells$bootstrap_mean - clamp(cells$original_estimate))
)

message(
  "Bootstrap centering, max |mean - original|: ",
  signif(maximum_centering_difference, 3),
  "; wrote ",
  n_boot,
  " replicates to ",
  dirname(output_draws)
)
