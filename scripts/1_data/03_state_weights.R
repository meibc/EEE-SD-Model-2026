# Construct state-period AMIS raking weights and dynamic-model input files.
#
# Periods pair AMIS cycles (year_group from scripts/1_data/01_clean_amis.R): 2017-2018
# (AMIS 2017, 2018), 2019-2020 (AMIS 2019), 2021-2022 (AMIS 2021, 2022), 2023-2024
# (AMIS 2023, 2024). For each state and period, this script:
#   1. begins with equal respondent weights;
#   2. rakes respondents to state-specific MSM demographic margins for age,
#      race/ethnicity, and education;
#   3. when a unique target and both response categories are available, also
#      rakes to a state-period PrEP target calculated as pooled CDC PrEP users
#      divided by pooled PrEP-eligible MSM over the matching calendar years
#      with CDC data (2023 only for 2023-2024);
#   4. winsorizes the resulting weights at the state-period 1st and 99th
#      percentiles; and
#   5. rescales the final weights to sum to 1 within each state-period.
#
# State demographic margins come from:
#   inputs/state_msm_margins.csv
# and were prepared by:
#   MSM-Population-Estimates/raking_targets/build_amis_raking_targets.R
# PrEP targets come from:
#   results/00_cdc/cdc_stocks.csv (scripts/1_data/02_build_cdc_stocks.R)
#
# Demographic categories absent from a state-period AMIS sample are omitted
# from that cell's raking margins, and the remaining target proportions are
# renormalized to sum to 1.
#
# Respondent inputs come from scripts/1_data/01_clean_amis.R. Respondents missing age,
# race/ethnicity, education, or PrEP-use status are excluded before raking.
# State-period cells with fewer than 30 complete respondents are excluded.
#
# Outputs:
#   results/01_weights/state_raked.rds
#       Respondent-level survey design containing final state-period weights.
#
#   results/01_weights/state_raking_diagnostics.csv
#       State-period sample sizes, effective sample sizes, weight variability,
#       winsorization frequency, convergence indicators, and PrEP calibration.
#
#   results/01_weights/ahs_wording_shift.csv
#       Estimated AHS wording shift (log-odds, AMIS 2022 vs 2021 within state-periods).
#
#   results/01_weights/state_model_inputs.xlsx
#       trends:
#         Weighted state-period means of the eight modeled AMIS measures. AHS is
#         adjusted for the AMIS 2022 wording change (shift estimated below; set
#         AHS_WORDING=unadjusted for the unadjusted sensitivity workbook).
#         USA estimates are constructed by weighting state estimates by each
#         state's estimated MSM population; they are not obtained from a
#         separate national respondent-level raking procedure.
#       states:
#         Census division and total retained respondent-row count for each
#         state across the included periods.
#       cov:
#         Pairwise-complete covariance matrix for the eight AMIS measures,
#         pooled across all states and periods, using the final state-period
#         weights rescaled by the state-level respondent count.
#
# Adapted from: EEE/AMIS - Factor Analysis v4.Rmd
#
# Run from the repository root:
#   Rscript scripts/1_data/03_state_weights.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(survey)
  library(srvyr)
  library(openxlsx2)
})

model_periods <- c(
  "2017-2018" = 2018L,
  "2019-2020" = 2020L,
  "2021-2022" = 2022L,
  "2023-2024" = 2024L
)

# AHS question wording changed in AMIS 2022. AHS_WORDING=unadjusted (sensitivity analysis) skips the
# adjustment below and writes the workbook as state_model_inputs_ahs_unadjusted.xlsx.
ahs_wording <- Sys.getenv("AHS_WORDING", "adjusted")
stopifnot(ahs_wording %in% c("adjusted", "unadjusted"))
output_xlsx <- if (ahs_wording == "adjusted") {
  "results/01_weights/state_model_inputs.xlsx"
} else {
  "results/01_weights/state_model_inputs_ahs_unadjusted.xlsx"
}
output_wording <- "results/01_weights/ahs_wording_shift.csv"
output_rds <- "results/01_weights/state_raked.rds"
output_diagnostics <- "results/01_weights/state_raking_diagnostics.csv"
dir.create("results/01_weights", recursive = TRUE, showWarnings = FALSE)

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
rake_control <- list(maxit = 100, epsilon = 1e-7)
cap_quantiles <- c(0.01, 0.99)
min_period_n <- 30

# ---- Raking targets ----------------------------------------------------------

msm_counts <- read_csv("inputs/state_msm_margins.csv", show_col_types = FALSE)
cdc_stocks <- read_csv("results/00_cdc/cdc_stocks.csv", show_col_types = FALSE)

# PrEP target: CDC PrEP / PrEP-eligible, pooled over each period's calendar years.
prep_targets <- cdc_stocks %>%
  filter(Population == "MSM", Indicator %in% c("PrEP", "PrEP Eligible")) %>%
  mutate(
    year_group = case_when(
      Year %in% 2017:2018 ~ "2017-2018",
      Year %in% 2019:2020 ~ "2019-2020",
      Year %in% 2021:2022 ~ "2021-2022",
      Year %in% 2023:2024 ~ "2023-2024"
    )
  ) %>%
  filter(!is.na(year_group)) %>%
  group_by(state_calc = St_Abbr, year_group, Indicator) %>%
  summarise(Cases = sum(Cases, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = Indicator, values_from = Cases) %>%
  filter(`PrEP Eligible` > 0) %>%
  transmute(
    state_calc,
    year_group,
    target_prep = pmin(pmax(`PrEP` / `PrEP Eligible`, 0), 1)
  )

# ---- Respondents ---------------------------------------------------------------

amis <- read_csv("results/00_amis/amis_analytic.csv.gz", show_col_types = FALSE) %>%
  filter(!is.na(agecat), !is.na(rdrace), !is.na(school4), !is.na(prep_used)) %>%
  mutate(
    prep_rake_status = factor(
      if_else(prep_used == 1, "prep_used", "prep_not_used"),
      levels = c("prep_not_used", "prep_used")
    )
  )

# ---- Raking --------------------------------------------------------------------

# Demographic margins for one state; categories absent from the sample are
# dropped and the remaining shares renormalized.
demographic_margin <- function(label, varname, state, present) {
  msm_counts %>%
    filter(ST_ABBREV == state, Label == label, Category %in% present) %>%
    transmute(
      !!varname := Category,
      Freq = group_sum_msm_share / sum(group_sum_msm_share)
    )
}

# Create raking weights for one state-period.
rake_period <- function(df, state, yg) {
  # Construct the demographic targets and their matching sample formulas.
  population_margins <- list(
    demographic_margin("Age", "agecat", state, df$agecat),
    demographic_margin("Race", "rdrace", state, df$rdrace),
    demographic_margin("Educ", "school4", state, df$school4)
  )
  sample_margins <- list(~agecat, ~rdrace, ~school4)

  # Add PrEP calibration only when exactly one target and both response
  # categories are available for the state-period.
  prep_target <- prep_targets$target_prep[
    prep_targets$state_calc == state &
      prep_targets$year_group == yg
  ]

  has_prep_margin <- length(prep_target) == 1 &&
    n_distinct(df$prep_rake_status) == 2

  if (has_prep_margin) {
    prep_population_margin <- tibble(
      prep_rake_status = factor(
        c("prep_not_used", "prep_used"),
        levels = levels(df$prep_rake_status)
      ),
      Freq = c(1 - prep_target, prep_target)
    )

    population_margins <- append(population_margins, list(prep_population_margin))
    sample_margins <- append(sample_margins, list(~prep_rake_status))
  }

  initial_design <- svydesign(ids = ~1, data = df, weights = ~1)

  # Retain a convergence flag while suppressing only the corresponding
  # survey::rake() warning.
  converged <- TRUE
  raked_design <- withCallingHandlers(
    rake(
      design = initial_design,
      sample.margins = sample_margins,
      population.margins = population_margins,
      control = rake_control
    ),
    warning = function(w) {
      if (grepl("did not converge", conditionMessage(w))) {
        converged <<- FALSE
        invokeRestart("muffleWarning")
      }
    }
  )

  raw_weights <- weights(raked_design)
  cap_values <- quantile(raw_weights, cap_quantiles)
  capped_weights <- pmin(pmax(raw_weights, cap_values[1]), cap_values[2])

  df %>%
    mutate(
      raked_weight_raw = raw_weights,
      raked_weight = capped_weights / sum(capped_weights),
      rake_converged = converged,
      prep_margin = has_prep_margin
    )
}

# Rake each state-period with at least min_period_n complete respondents.
raked_respondents <- amis %>%
  group_by(state_calc, year_group) %>%
  filter(n() >= min_period_n) %>%
  group_modify(~ rake_period(.x, .y$state_calc, .y$year_group)) %>%
  ungroup()

# Save the raked survey design for later use in the model.
state_design <- as_survey_design(
  raked_respondents,
  ids = 1,
  weights = raked_weight
)

saveRDS(state_design, output_rds)

# ---- Diagnostics  ------------------------------------------------------
diagnostics <- raked_respondents %>%
  group_by(state_calc, year_group) %>%
  summarise(
    n = n(),
    n_eff = sum(raked_weight)^2 / sum(raked_weight^2),
    weight_cv = sd(raked_weight) / mean(raked_weight),
    share_capped = mean(
      raked_weight_raw < quantile(raked_weight_raw, cap_quantiles[1]) |
        raked_weight_raw > quantile(raked_weight_raw, cap_quantiles[2])
    ),
    rake_converged = first(rake_converged),
    prep_margin = first(prep_margin),
    unweighted_prep = mean(prep_used),
    weighted_prep = weighted.mean(prep_used, raked_weight),
    .groups = "drop"
  ) %>%
  left_join(prep_targets, by = c("state_calc", "year_group"))
write_csv(diagnostics, output_diagnostics)

nonconverged_cells <- diagnostics %>%
  filter(!rake_converged) %>%
  transmute(label = paste(state_calc, year_group)) %>%
  pull(label)

message(
  "Raked ",
  nrow(diagnostics),
  " state-periods; not converged: ",
  paste(nonconverged_cells, collapse = ", ")
)

# ---- Model input workbook ------------------------------------------------------

# MSM population targets
msm_pop <- read_csv("inputs/msm_state_population.csv", col_select = c(state, rounded), name_repair = "unique_quiet",
                    show_col_types = FALSE) %>%
  transmute(state_calc = state, msm_pop = rounded)
missing_pop <- setdiff(
  unique(raked_respondents$state_calc),
  msm_pop$state_calc
)
if (length(missing_pop) > 0) {
  stop(
    "Missing MSM population estimates for: ",
    paste(missing_pop, collapse = ", ")
  )
}

# States sheet: total retained respondent-row count for each state across the included periods.
sheet_states <- raked_respondents %>%
  count(state_calc, Census_div = `_censdiv`) %>%
  select(state_calc, n, Census_div)

# AHS wording change (AMIS 2022: "worry someone may learn you have sex with men" became "worry
# someone may treat you poorly because you have sex with men"). The shift is the age-adjusted
# log-odds difference in AHS between AMIS 2022 and AMIS 2021 respondents in the same state and
# period (2021-2022), using the state-period weights. State-period AHS estimates are moved back by
# the shift times the weighted share of AHS respondents from AMIS 2022 or later (scripts/1_data/04 applies
# the same adjustment to every bootstrap replicate).
wording_data <- raked_respondents %>%
  filter(year_group == "2021-2022", !is.na(stigma_ahs)) %>%
  group_by(state_calc) %>%
  mutate(weight = raked_weight * n()) %>%
  ungroup() %>%
  mutate(amis_2022 = as.integer(survey_year == 2022), agecat = factor(agecat), state_calc = factor(state_calc))
wording_fit <- svyglm(stigma_ahs ~ amis_2022 + state_calc + agecat,
                      design = svydesign(ids = ~1, weights = ~weight, data = wording_data),
                      family = quasibinomial())
wording_shift <- coef(wording_fit)[["amis_2022"]]
wording_se <- sqrt(vcov(wording_fit)["amis_2022", "amis_2022"])
write_csv(tibble(shift_log_odds = wording_shift, se = wording_se,
                 n_amis_2021 = sum(wording_data$amis_2022 == 0), n_amis_2022 = sum(wording_data$amis_2022 == 1)),
          output_wording)
message(sprintf("AHS wording shift: %.3f log-odds (SE %.3f); %s", wording_shift, wording_se, ahs_wording))
applied_shift <- if (ahs_wording == "adjusted") wording_shift else 0

ahs_new_wording_share <- raked_respondents %>%
  filter(!is.na(stigma_ahs)) %>%
  group_by(Geography = state_calc, year_group) %>%
  summarise(share_new = weighted.mean(survey_year >= 2022, raked_weight), .groups = "drop")

# AMIS trends sheet: weighted state-period means of the eight measures, plus USA (MSM-population weighted).
state_trends <- raked_respondents %>%
  group_by(Geography = state_calc, year_group) %>%
  summarise(
    across(
      all_of(model_vars),
      ~ weighted.mean(.x, raked_weight, na.rm = TRUE)
    ),
    .groups = "drop"
  ) %>%
  pivot_longer(
    all_of(model_vars),
    names_to = "Indicator",
    values_to = "Value"
  ) %>%
  left_join(ahs_new_wording_share, by = c("Geography", "year_group")) %>%
  mutate(Value = if_else(Indicator == "stigma_ahs",
                         plogis(qlogis(Value) - applied_shift * coalesce(share_new, 0)), Value)) %>%
  transmute(
    Geography,
    Year = model_periods[year_group],
    Indicator,
    Value,
    Source = "AMIS"
  )

# USA trends: weighted average of the state-period means, weighted by the MSM population in each state.
usa_trends <- state_trends %>%
  inner_join(msm_pop, by = c("Geography" = "state_calc")) %>%
  filter(!is.na(Value)) %>%
  group_by(Year, Indicator) %>%
  summarise(Value = sum(Value * msm_pop) / sum(msm_pop), .groups = "drop") %>%
  transmute(Geography = "USA", Year, Indicator, Value, Source = "AMIS")

# combine state and USA trends into a single sheet
sheet_trends <- bind_rows(state_trends, usa_trends)

# Pooled weighted covariance across all states and periods (pairwise complete).
# Each state-period's weights sum to 1, so scaling by the state's
# respondent count weights states by sample size.
# Cov (X, Y) = E[XY] - E[X]E[Y], with E[.] = sum(w * .) / sum(w)
# Cov sheet: row and column names are the model_vars, values are the covariances.
state_sample_size <- sheet_states$n[
  match(raked_respondents$state_calc, sheet_states$state_calc)
]

covariance_weight <-
  raked_respondents$raked_weight * state_sample_size

weighted_pairwise_covariance <- function(i, j) {
  x <- raked_respondents[[model_vars[i]]]
  y <- raked_respondents[[model_vars[j]]]
  complete_pair <- !is.na(x) & !is.na(y)

  pair_weights <- covariance_weight[complete_pair]
  pair_weights <- pair_weights / sum(pair_weights)

  weighted_cross_product <- sum(
    pair_weights * x[complete_pair] * y[complete_pair]
  )

  weighted_x_mean <- sum(pair_weights * x[complete_pair])
  weighted_y_mean <- sum(pair_weights * y[complete_pair])

  weighted_cross_product - weighted_x_mean * weighted_y_mean
}

sheet_cov <- outer(
  seq_along(model_vars),
  seq_along(model_vars),
  Vectorize(weighted_pairwise_covariance)
)
dimnames(sheet_cov) <- list(model_vars, model_vars)

# create workbook and write the three sheets
wb <- wb_workbook()

workbook_data <- list(
  trends = sheet_trends,
  states = sheet_states,
  cov = sheet_cov
)

for (sheet in c("trends", "states", "cov")) {
  wb$add_worksheet(sheet)
  wb$add_data(sheet, workbook_data[[sheet]])
}

wb$save(output_xlsx, overwrite = TRUE)
message("Wrote ", output_xlsx)
