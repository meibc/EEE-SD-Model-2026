# State precision screen for the primary comparative analysis.
#
# Inclusion depends only on AMIS input precision and completeness: the cell-level Kish effective sample
# sizes of the 8 measures x 4 two-year periods for each state. The input CSV is not an external source:
# scripts/1_data/04_state_bootstrap.R creates it from the original state-period estimates before resampling. For
# each measure-specific cell, n_eff = (sum of raking weights)^2 / sum of squared raking weights. The same
# file also records the original estimate and the mean and SD across respondent-bootstrap replicates.
# This screen is therefore computed before any model is fitted. Cells filled from the Census division by
# scripts/1_data/04 have no state-level n_eff and do not count as complete.
#
# Screens
#   primary   all 32 cells; median n_eff >= 60; 25th percentile n_eff >= 40
#   median55  sensitivity: the same rule with a median cutoff of 55
#   strict    sensitivity: all 32 cells; 25th percentile >= 50; minimum >= 25
#
# Input:  results/02_inputs/state_bootstrap/cell_precision.csv (scripts/1_data/04)
# Output (results/02_inputs/state_screen/): state_effective_n_screen.csv (all states), and primary_states.csv,
#         median55_states.csv, strict_states.csv (the states passing each screen)
#
# Run from the repository root: Rscript scripts/1_data/05_state_precision_screen.R

suppressPackageStartupMessages(library(tidyverse))

cells_path <- "results/02_inputs/state_bootstrap/cell_precision.csv"
output_dir <- "results/02_inputs/state_screen"
periods <- c(2018, 2020, 2022, 2024)
variables <- c("stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "risk_behavior", "prep_used", "hivtest12")
expected_cells <- length(periods) * length(variables)
median_cutoff <- 60
q25_cutoff <- 40

screen <- read_csv(cells_path, show_col_types = FALSE) %>%
  filter(geography_type == "state", year %in% periods, variable %in% variables) %>%
  mutate(n_eff = suppressWarnings(as.numeric(n_eff))) %>%
  group_by(state = as.character(unit)) %>%
  summarise(
    n_expected_cells = expected_cells,
    n_complete_cells = n_distinct(paste(year, variable)[!is.na(n_eff)]),
    n_values = sum(!is.na(n_eff)),
    min_cell_neff = if (n_values > 0) min(n_eff, na.rm = TRUE) else NA_real_,
    q25_cell_neff = if (n_values > 0) quantile(n_eff, 0.25, na.rm = TRUE, type = 7, names = FALSE) else NA_real_,
    median_cell_neff = if (n_values > 0) median(n_eff, na.rm = TRUE) else NA_real_,
    q75_cell_neff = if (n_values > 0) quantile(n_eff, 0.75, na.rm = TRUE, type = 7, names = FALSE) else NA_real_,
    max_cell_neff = if (n_values > 0) max(n_eff, na.rm = TRUE) else NA_real_,
    .groups = "drop"
  ) %>%
  mutate(
    complete_32_cells = n_complete_cells == expected_cells & n_values == expected_cells,
    primary_precision_screen = complete_32_cells & median_cell_neff >= median_cutoff & q25_cell_neff >= q25_cutoff,
    median55_precision_screen = complete_32_cells & median_cell_neff >= 55 & q25_cell_neff >= q25_cutoff,
    strict_precision_screen = complete_32_cells & q25_cell_neff >= 50 & min_cell_neff >= 25,
    across(ends_with("_screen"), ~ coalesce(.x, FALSE))
  ) %>%
  select(state, n_expected_cells, n_complete_cells, complete_32_cells, min_cell_neff, q25_cell_neff,
         median_cell_neff, q75_cell_neff, max_cell_neff, ends_with("_screen")) %>%
  arrange(desc(primary_precision_screen), desc(median_cell_neff), state)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(screen, file.path(output_dir, "state_effective_n_screen.csv"), na = "")
for (name in c("primary", "median55", "strict")) {
  passed <- screen %>% filter(.data[[paste0(name, "_precision_screen")]])
  write_csv(passed, file.path(output_dir, paste0(name, "_states.csv")), na = "")
  message(sprintf("%s (%d): %s", name, nrow(passed), paste(passed$state, collapse = ", ")))
}
