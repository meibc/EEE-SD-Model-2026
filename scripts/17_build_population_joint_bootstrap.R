#!/usr/bin/env Rscript

# Joint respondent-bootstrap trajectories for the six national demographic
# groups and the California Hispanic/Latino bridge analysis.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(survey)
})

input_rds <- Sys.getenv(
  "POP_BOOT_INPUT_RDS",
  "/Users/meibinchen/Documents/Codex/2026-08-13/d/outputs/amis_national_priority_population_raked_PRIMARY.rds"
)
output_dir <- Sys.getenv(
  "POP_BOOT_OUTPUT_DIR",
  "/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/output/population_joint_bootstrap_n500"
)
n_boot <- as.integer(Sys.getenv("POP_BOOT_N", "500"))
seed <- as.integer(Sys.getenv("POP_BOOT_SEED", "20261006"))
eps <- 1e-6

variables <- c(
  "stigma_ahs", "stigma_gss", "stigma_family", "out_gid",
  "seehcp", "risk_behavior", "prep_used", "hivtest12"
)
observed_years <- c(2017L, 2018L, 2019L, 2021L, 2022L, 2023L, 2024L)
period_year <- c(
  "2017-2018" = 2018L,
  "2019-2020" = 2020L,
  "2021-2022" = 2022L,
  "2023-2024" = 2024L
)

weighted_mean_safe <- function(x, w) {
  keep <- is.finite(x) & is.finite(w) & w > 0
  if (!any(keep)) return(NA_real_)
  sum(x[keep] * w[keep]) / sum(w[keep])
}

kish_neff_safe <- function(x, w) {
  keep <- is.finite(x) & is.finite(w) & w > 0
  if (!any(keep)) return(NA_real_)
  sw <- sum(w[keep])
  sw2 <- sum(w[keep]^2)
  if (!is.finite(sw2) || sw2 <= 0) return(NA_real_)
  sw^2 / sw2
}

summarize_population <- function(data, population_id, population_label, time_col, analysis) {
  values <- data %>%
    group_by(.data[[time_col]]) %>%
    summarise(
      across(all_of(variables), ~ weighted_mean_safe(.x, analysis_weight)),
      .groups = "drop"
    ) %>%
    rename(Year = all_of(time_col)) %>%
    pivot_longer(all_of(variables), names_to = "variable", values_to = "value")

  neff <- data %>%
    group_by(.data[[time_col]]) %>%
    summarise(
      across(all_of(variables), ~ kish_neff_safe(.x, analysis_weight)),
      .groups = "drop"
    ) %>%
    rename(Year = all_of(time_col)) %>%
    pivot_longer(all_of(variables), names_to = "variable", values_to = "n_eff")

  values %>%
    left_join(neff, by = c("Year", "variable")) %>%
    mutate(
      population_id = population_id,
      population_label = population_label,
      analysis = analysis,
      .before = 1
    )
}

smooth_one_trajectory <- function(data) {
  data <- data %>% arrange(Year)
  z <- qlogis(pmin(pmax(data$value, eps), 1 - eps))
  out <- numeric(length(z))
  for (i in seq_along(z)) {
    neighbors <- intersect(seq_along(z), c(i - 1L, i, i + 1L))
    kernel <- ifelse(neighbors == i, 1.0, 0.5)
    weights_local <- kernel * data$n_eff[neighbors]
    valid <- is.finite(z[neighbors]) & is.finite(weights_local) & weights_local > 0
    if (!any(valid)) {
      out[i] <- z[i]
    } else {
      out[i] <- sum(z[neighbors][valid] * weights_local[valid]) / sum(weights_local[valid])
    }
  }
  data$value <- pmin(pmax(plogis(out), eps), 1 - eps)
  data
}

summarize_all_populations <- function(data) {
  groups <- list(
    list("black_nh", "Black, non-Hispanic", data %>% filter(race_ethnicity == "Black, non-Hispanic")),
    list("hispanic_latino", "Hispanic/Latino", data %>% filter(race_ethnicity == "Hispanic/Latino")),
    list("white_nh", "White, non-Hispanic", data %>% filter(race_ethnicity == "White, non-Hispanic")),
    list("age_under_24", "<=24", data %>% filter(age_group == "<=24")),
    list("age_25_34", "25-34", data %>% filter(age_group == "25-34")),
    list("age_35_plus", "35+", data %>% filter(age_group == "35+"))
  )

  national <- bind_rows(lapply(groups, function(g) {
    summarize_population(g[[3]], g[[1]], g[[2]], "survey_year", "national_annual_smoothed")
  })) %>%
    group_by(population_id, population_label, analysis, variable) %>%
    group_modify(~ smooth_one_trajectory(.x)) %>%
    ungroup()

  california <- data %>%
    filter(state_calc == "CA", race_ethnicity == "Hispanic/Latino") %>%
    summarize_population(
      "ca_hispanic_latino", "California Hispanic/Latino",
      "period_year", "california_two_year_grouped"
    )

  bind_rows(national, california) %>%
    arrange(population_id, Year, match(variable, variables))
}

if (!file.exists(input_rds)) stop("Missing input RDS: ", input_rds)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

design <- readRDS(input_rds)
if (!is.list(design) || !"variables" %in% names(design)) {
  stop("Expected a survey design object with a $variables data frame")
}
raw <- as_tibble(design$variables)
raw$analysis_weight <- as.numeric(weights(design))

required <- c(
  "rowID", "state_calc", "year_group", "survey_year", "survey", "_hivstat",
  "age", "rdrace", "test2yrs", "analysis_weight", variables
)
missing <- setdiff(required, names(raw))
if (length(missing) > 0) stop("Missing required fields: ", paste(missing, collapse = ", "))

analysis_data <- raw %>%
  filter(`_hivstat` == 1, !survey %in% c("amis_15", "amis_16")) %>%
  mutate(
    survey_year = as.integer(survey_year),
    state_calc = as.character(state_calc),
    year_group = as.character(year_group),
    period_year = unname(as.integer(period_year[year_group])),
    race_ethnicity = case_when(
      rdrace == "black" ~ "Black, non-Hispanic",
      rdrace == "white" ~ "White, non-Hispanic",
      rdrace == "hisp" ~ "Hispanic/Latino",
      TRUE ~ NA_character_
    ),
    age_group = case_when(
      !is.na(age) & age <= 24 ~ "<=24",
      !is.na(age) & age >= 25 & age <= 34 ~ "25-34",
      !is.na(age) & age >= 35 ~ "35+",
      TRUE ~ NA_character_
    ),
    # Match the state factor-analysis definition: reported HIV tests during
    # the preceding two years, capped at four tests and scaled to [0, 1].
    # Values outside the documented nonnegative count range are missing.
    test2yrs_raw = suppressWarnings(as.numeric(as.character(test2yrs))),
    test2yrs_num = if_else(
      test2yrs_raw >= 0 & test2yrs_raw <= 100,
      test2yrs_raw,
      NA_real_
    ),
    hivtest12 = pmin(test2yrs_num, 4) / 4,
    stratum = interaction(state_calc, survey_year, drop = TRUE, lex.order = TRUE)
  ) %>%
  filter(
    survey_year %in% observed_years,
    !is.na(period_year), !is.na(state_calc),
    is.finite(analysis_weight), analysis_weight > 0
  ) %>%
  mutate(across(all_of(variables), ~ suppressWarnings(as.numeric(as.character(.x)))))

strata <- split(seq_len(nrow(analysis_data)), analysis_data$stratum, drop = TRUE)
set.seed(seed)

original <- summarize_all_populations(analysis_data) %>% mutate(draw = 0L, .before = 1)
draws <- vector("list", n_boot)
for (b in seq_len(n_boot)) {
  sampled_rows <- unlist(
    lapply(strata, function(index) sample(index, length(index), replace = TRUE)),
    use.names = FALSE
  )
  draws[[b]] <- summarize_all_populations(analysis_data[sampled_rows, , drop = FALSE]) %>%
    mutate(draw = b, .before = 1)
  if (b %% 25L == 0L || b == n_boot) {
    message("Completed population bootstrap replicate ", b, " of ", n_boot)
  }
}

bootstrap <- bind_rows(draws)
rm(draws)
gc()

expected_cells <- 6L * 7L * length(variables) + 4L * length(variables)
coverage <- bootstrap %>% count(draw, name = "n_cells")
if (nrow(coverage) != n_boot || any(coverage$n_cells != expected_cells)) {
  stop("Incomplete bootstrap replicate coverage")
}
if (any(!is.finite(bootstrap$value)) || any(bootstrap$value <= 0 | bootstrap$value >= 1)) {
  stop("Bootstrap trajectories contain invalid probability values")
}

centering <- bootstrap %>%
  group_by(population_id, population_label, analysis, Year, variable) %>%
  summarise(
    bootstrap_mean = mean(value),
    bootstrap_sd = sd(value),
    min_n_eff = min(n_eff, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(
    original %>% select(population_id, analysis, Year, variable, original_value = value),
    by = c("population_id", "analysis", "Year", "variable")
  ) %>%
  mutate(
    centering_difference = bootstrap_mean - original_value,
    absolute_centering_difference = abs(centering_difference)
  )

metadata <- tibble(
  field = c(
    "artifact_type", "input_rds", "n_boot", "seed", "resampling_strata",
    "weight_handling", "national_time_points", "national_smoothing",
    "california_time_points", "hiv_testing_definition"
  ),
  value = c(
    "joint respondent bootstrap for national demographic and California Hispanic/Latino trajectories",
    input_rds, as.character(n_boot), as.character(seed), "state_calc x survey_year",
    "final age-PrEP calibrated weights carried with resampled complete respondent records",
    paste(observed_years, collapse = ","),
    "within-replicate ESS-weighted local logit smoother; focal kernel 1.0, adjacent observed waves 0.5",
    paste(unname(period_year), collapse = ","),
    "test2yrs count capped at 4 and divided by 4, matching the state factor-analysis frequency construct"
  )
)

saveRDS(bootstrap, file.path(output_dir, "population_joint_bootstrap_draws_n500.rds"), compress = "xz")
write_csv(bootstrap, gzfile(file.path(output_dir, "population_joint_bootstrap_draws_n500.csv.gz")))
write_csv(original, file.path(output_dir, "population_joint_bootstrap_point_trajectories.csv"))
write_csv(centering, file.path(output_dir, "population_joint_bootstrap_centering_audit.csv"))
write_csv(metadata, file.path(output_dir, "population_joint_bootstrap_metadata.csv"))

message("Saved population bootstrap outputs to ", output_dir)
