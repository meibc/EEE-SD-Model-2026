# Data-derived model constants reported in the supplement (Table S4), from the weighted state-period
# estimates of the primary precision-screen states over the four model periods.
#
# - Reference probabilities (S2.2): for each measure, the 5th percentile of the weighted state-period
#   point estimates. Under state anchoring they no longer centre the dynamics (each state is centred at
#   its own mean); they set the AHS benchmark and convert the AHS-PrEP prevalence ratio to the model's
#   logit scale.
# - PrEP drift regularization target (S2.3): unweighted mean of the three two-year changes in logit PrEP.
# - PrEP persistence target (S2.4): pooled AR(1) slope of logit PrEP, x[t+1] = a + phi * x[t] (the slope
#   does not depend on centring). The intercept a is not reported: its value depends on the centring
#   constant, so it has no stable meaning; PrEP growth is the drift target above.
# - Within-state PrEP persistence: the same AR(1) on each state's departures from its own mean, the scale
#   of the state-anchored model. Reported for comparison, not used: with three transitions per state it is
#   biased downward.
#
# Inputs: results/01_weights/state_model_inputs.xlsx (sheet "trends"; scripts/1_data/03),
#         results/02_inputs/state_screen/primary_states.csv (scripts/1_data/05)
# Output: results/02_inputs/reported_constants.csv
# With --write-config, also writes the references, the drift target (3 decimals) and the persistence target
# (phi rounded to the nearest 0.05) to config/dynamic_variables.csv and config/dynamic_persistence_priors.csv.
#
# Run from the repository root after scripts 01-07 and before fitting:
#   Rscript scripts/1_data/06_reported_constants.R [--write-config]

suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
})

workbook <- "results/01_weights/state_model_inputs.xlsx"
primary_states_path <- "results/02_inputs/state_screen/primary_states.csv"
variable_config_path <- "config/dynamic_variables.csv"
diagonal_config_path <- "config/dynamic_persistence_priors.csv"
output_path <- "results/02_inputs/reported_constants.csv"
write_config <- "--write-config" %in% commandArgs(trailingOnly = TRUE)
years <- c(2018, 2020, 2022, 2024)

logit <- function(p) {
  p <- pmin(pmax(p, 0.01), 0.99)                   # clipping used by the model (models/dynamic/estimation/transforms.py)
  log(p / (1 - p))
}

states <- read_csv(primary_states_path, show_col_types = FALSE)$state
trends <- read_excel(workbook, sheet = "trends") %>%
  filter(Geography %in% states, Year %in% years)
if (any(count(trends, Indicator)$n != length(states) * length(years))) {
  stop("Primary states lack a complete set of state-period estimates")
}

# get the 5th percentile of each measure's weighted state-period estimates
variable_config <- read_csv(variable_config_path, show_col_types = FALSE)
references <- trends %>%
  group_by(Indicator) %>%
  summarise(value = quantile(Value, 0.05, type = 7), .groups = "drop") %>%
  deframe()

# Logit PrEP by state (rows) and period (columns, in order).
# The drift target is the unweighted mean of all 3 * n_states adjacent two-year
# changes in logit PrEP:
#   drift = mean_g,t(x_g,t+1 - x_g,t).
# With a complete balanced panel, this is equivalently:
#   [mean_g(x_g,2024) - mean_g(x_g,2018)] / 3.
prep <- trends %>%
  filter(Indicator == "prep_used") %>%
  mutate(x = logit(Value)) %>%
  select(Geography, Year, x) %>%
  pivot_wider(names_from = Year, values_from = x) %>%
  select(all_of(as.character(years))) %>%
  as.matrix()
drift_target <- mean(prep[, -1] - prep[, -ncol(prep)])

# Convert the state x period matrix into the 3 * n_states paired transitions used by lm():
#   x_t    = (x_g,2018, x_g,2020, x_g,2022)
#   x_next = (x_g,2020, x_g,2022, x_g,2024).
# as.vector() stacks both matrices in the same column-major order, so each state's value at t
# remains paired with that same state's value at t + 1.
transitions <- function(x) tibble(x_t = as.vector(x[, -ncol(x)]), x_next = as.vector(x[, -1]))

# Pooled persistence across all state-period transitions. With c = logit(reference PrEP), define
#   z_g,t = x_g,t - c
# and estimate
#   z_g,t+1 = a_pooled + phi_pooled * z_g,t + error_g,t.
# Only phi_pooled is used: it does not depend on c, whereas a_pooled does.
pooled <- coef(summary(lm(x_next ~ x_t, data = transitions(prep - logit(references[["prep_used"]])))))

# Within-state persistence. For each state g, define its four-period mean
#   xbar_g = (1 / 4) * sum_t x_g,t
# and its departure from that mean
#   d_g,t = x_g,t - xbar_g.
# Estimate
#   d_g,t+1 = a_within + phi_within * d_g,t + error_g,t.
# This removes stable differences in average PrEP levels between states. 
within <- coef(summary(lm(x_next ~ x_t, data = transitions(prep - rowMeans(prep)))))

# create csv of constants for the supplement 
n_states <- length(states)
constants <- bind_rows(
  tibble(constant = "prep_drift_target", value = drift_target, se = NA_real_, n_states = n_states,
         n_obs = 3L * n_states, definition = "mean two-year change in logit PrEP use, primary-screen states"),
  tibble(constant = "prep_persistence_phi", value = pooled["x_t", "Estimate"], se = pooled["x_t", "Std. Error"],
         n_states = n_states, n_obs = 3L * n_states,
         definition = "pooled centered-logit AR(1) slope with intercept, primary-screen states"),
  tibble(constant = "prep_persistence_phi_within_state", value = within["x_t", "Estimate"], se = within["x_t", "Std. Error"],
         n_states = n_states, n_obs = 3L * n_states,
         definition = "AR(1) slope of departures from each state's mean logit PrEP (state-anchored scale); biased downward with three transitions; not used"),
  tibble(constant = paste0("reference_", variable_config$variable), value = unname(references[variable_config$variable]),
         se = NA_real_, n_states = n_states, n_obs = n_states * length(years),
         definition = "5th percentile of weighted state-period estimates, primary-screen states")
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_csv(constants, output_path, na = "")
print(constants %>% select(constant, value, se), n = Inf)

# to account for the new PrEP drift target and persistence target, update the config files if requested
if (write_config) {
  persistence_target <- round(pooled["x_t", "Estimate"] * 20) / 20
  variable_config %>%
    mutate(reference_prob = unname(references[variable]),
           reference_note = sprintf("Fifth percentile of weighted point estimates across the %d primary states and four periods.", n_states),
           drift_prior_mean = if_else(variable == "prep_used", round(drift_target, 3), drift_prior_mean)) %>%
    write_csv(variable_config_path, na = "")
  read_csv(diagonal_config_path, show_col_types = FALSE) %>%
    mutate(prior_mean = if_else(variable == "prep_used", persistence_target, prior_mean)) %>%
    write_csv(diagonal_config_path, na = "")
  message(sprintf("Wrote references, PrEP drift target %.3f and persistence target %.2f to config/",
                  round(drift_target, 3), persistence_target))
}
