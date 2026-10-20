#!/usr/bin/env Rscript

# National cross-sectional path-model benchmark for the dynamic pathway model.
#
# Purpose
# -------
# This analysis asks a deliberately narrower question than the longitudinal
# dynamic model: after accounting for AHS, measured engagement variables,
# survey period, and respondent characteristics, do general social stigma
# (GSS) and family stigma retain direct cross-sectional associations with
# past-year PrEP use?
#
# It compares a conceptual/mediated model with an expanded model that adds
# direct GSS -> PrEP and family stigma -> PrEP paths.  A second specification
# adds state fixed effects.  These are weighted cross-sectional associations,
# not lagged effects or causal estimates.

suppressPackageStartupMessages({
  library(dplyr)
  library(lavaan)
  library(readr)
  library(tibble)
})

required_packages <- c("dplyr", "lavaan", "readr", "tibble")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop("Install required packages: ", paste(missing_packages, collapse = ", "))
}

input_rds <- Sys.getenv(
  "AMIS_RESPONDENT_RDS",
  "/Users/meibinchen/Documents/GitHub/EEE/amis_svy_df_prep_raked.rds"
)
msm_population_csv <- Sys.getenv(
  "MSM_POPULATION_CSV",
  "/Users/meibinchen/Documents/GitHub/MSM-Population-Estimates/outputs/final/msm_state_summary.csv"
)
output_dir <- Sys.getenv(
  "SEM_OUTPUT_DIR",
  "output/national_cross_sectional_sem"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(input_rds)) stop("Respondent RDS not found: ", input_rds)
if (!file.exists(msm_population_csv)) {
  stop("MSM population file not found: ", msm_population_csv)
}

raw <- readRDS(input_rds)
if (is.list(raw) && "variables" %in% names(raw)) raw <- raw$variables
if (!is.data.frame(raw)) stop("Input RDS is not a respondent data frame.")

required_columns <- c(
  "state_calc", "survey", "_hivstat", "year_group", "raked_weight",
  "stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp",
  "risk_behavior", "prep_used", "test2yrs", "agecat", "rdrace",
  "income_cat", "school4", "insur_bin"
)
missing_columns <- setdiff(required_columns, names(raw))
if (length(missing_columns) > 0) {
  stop("Input data are missing: ", paste(missing_columns, collapse = ", "))
}

msm_population <- read_csv(msm_population_csv, show_col_types = FALSE) %>%
  transmute(state_calc = as.character(state), state_msm_population = as.numeric(rounded))

# Match the production filters and HIV-testing-frequency construction.
analysis <- raw %>%
  filter(
    `_hivstat` != 2,
    !survey %in% c("amis_15", "amis_16"),
    year_group %in% c("2017-2018", "2019-2020", "2021-2022", "2023-2024")
  ) %>%
  transmute(
    state_calc = as.character(state_calc),
    year_group = factor(
      year_group,
      levels = c("2017-2018", "2019-2020", "2021-2022", "2023-2024")
    ),
    raked_weight = as.numeric(raked_weight),
    stigma_ahs = as.numeric(stigma_ahs),
    stigma_gss = as.numeric(stigma_gss),
    stigma_family = as.numeric(stigma_family),
    out_gid = as.numeric(out_gid),
    seehcp = as.numeric(seehcp),
    risk_behavior = as.numeric(risk_behavior),
    prep_used = as.numeric(prep_used),
    # Production definition: reported testing frequency, capped at four tests
    # in two years and rescaled to [0, 1]. Invalid legacy codes are missing.
    test2yrs_num = suppressWarnings(as.numeric(as.character(test2yrs))),
    hiv_testing_frequency = if_else(
      test2yrs_num >= 0 & test2yrs_num <= 100,
      pmin(test2yrs_num, 4) / 4,
      NA_real_
    ),
    agecat = factor(if_else(is.na(agecat), "missing", as.character(agecat))),
    rdrace = factor(if_else(is.na(rdrace), "missing", as.character(rdrace))),
    income_cat = factor(if_else(is.na(income_cat), "missing", as.character(income_cat))),
    school4 = factor(if_else(is.na(school4), "missing", as.character(school4))),
    insurance = factor(
      case_when(
        is.na(insur_bin) ~ "missing",
        as.numeric(insur_bin) == 1 ~ "insured",
        TRUE ~ "uninsured"
      ),
      levels = c("insured", "uninsured", "missing")
    )
  ) %>%
  filter(
    !is.na(state_calc),
    is.finite(raked_weight),
    raked_weight > 0
  ) %>%
  inner_join(msm_population, by = "state_calc") %>%
  group_by(state_calc, year_group) %>%
  mutate(
    # The original raking weights are normalized within state-period. Scaling
    # each cell to its estimated MSM population reconstructs the national
    # aggregation used by the production pipeline.
    national_weight_raw = raked_weight / sum(raked_weight) * first(state_msm_population)
  ) %>%
  ungroup() %>%
  mutate(national_weight = national_weight_raw / mean(national_weight_raw))

if (nrow(analysis) == 0) stop("No eligible observations remained after filtering.")

make_design <- function(data, include_state_fixed_effects = FALSE) {
  rhs <- paste(
    "year_group + agecat + rdrace + income_cat + school4 + insurance",
    if (include_state_fixed_effects) "+ factor(state_calc)" else ""
  )
  mm <- model.matrix(as.formula(paste("~", rhs)), data = data)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  colnames(mm) <- make.names(colnames(mm), unique = TRUE)
  bind_cols(data, as.data.frame(mm))
}

fit_pair <- function(data, spec_label, include_state_fixed_effects = FALSE) {
  focal_variables <- c(
    "stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp",
    "risk_behavior", "prep_used", "hiv_testing_frequency", "national_weight"
  )
  data <- data %>%
    filter(if_all(all_of(focal_variables), ~ !is.na(.x) & is.finite(.x)))
  dat <- make_design(data, include_state_fixed_effects)
  original_names <- names(data)
  adjustment_vars <- setdiff(names(dat), original_names)
  adjustment_rhs <- paste(adjustment_vars, collapse = " + ")

  # Recursive ordering avoids treating reciprocal, lagged relationships in the
  # dynamic model as simultaneous cross-sectional causal paths.
  common_model <- paste0(
    "stigma_ahs ~ a_gss*stigma_gss + a_family*stigma_family + ",
    "a_care*seehcp + ", adjustment_rhs, "\n",
    "out_gid ~ d_ahs*stigma_ahs + ", adjustment_rhs, "\n",
    "prep_used ~ p_ahs*stigma_ahs + p_out*out_gid + p_care*seehcp + ",
    "p_test*hiv_testing_frequency + p_risk*risk_behavior + ", adjustment_rhs, "\n",
    "ind_gss_ahs_prep := a_gss * p_ahs\n",
    "ind_family_ahs_prep := a_family * p_ahs\n",
    "ind_gss_ahs_out_prep := a_gss * d_ahs * p_out\n",
    "ind_family_ahs_out_prep := a_family * d_ahs * p_out\n"
  )

  expanded_model <- sub(
    "prep_used ~ ",
    "prep_used ~ direct_gss*stigma_gss + direct_family*stigma_family + ",
    common_model,
    fixed = TRUE
  )

  fit_one <- function(model_syntax) {
    sem(
      model = model_syntax,
      data = dat,
      ordered = c("stigma_ahs", "out_gid", "prep_used"),
      estimator = "WLSMV",
      parameterization = "theta",
      sampling.weights = "national_weight",
      sampling.weights.normalization = "total",
      # Adjustment variables are fully observed after explicit missing-value
      # categories are created. Treating them as fixed exogenous predictors
      # avoids estimating a very large, unnecessary covariance block.
      fixed.x = TRUE,
      conditional.x = TRUE,
      missing = "listwise",
      meanstructure = TRUE,
      ncpus = 1L
    )
  }

  fits <- list(conceptual = fit_one(common_model), expanded = fit_one(expanded_model))

  fit_table <- bind_rows(lapply(names(fits), function(model_name) {
    values <- fitMeasures(
      fits[[model_name]],
      c("chisq", "df", "chisq.scaled", "df.scaled", "pvalue.scaled", "cfi.scaled",
        "tli.scaled", "rmsea.scaled", "rmsea.ci.lower.scaled",
        "rmsea.ci.upper.scaled", "srmr")
    )
    as_tibble_row(as.list(values)) %>%
      mutate(specification = spec_label, model = model_name, nobs = nobs(fits[[model_name]]),
             .before = 1)
  }))

  coefficients <- bind_rows(lapply(names(fits), function(model_name) {
    parameterEstimates(fits[[model_name]], standardized = TRUE, ci = TRUE) %>%
      as_tibble() %>%
      filter(op %in% c("~", ":=")) %>%
      mutate(specification = spec_label, model = model_name, .before = 1)
  }))

  difference_test <- lavTestLRT(fits$expanded, fits$conceptual) %>%
    as.data.frame() %>%
    rownames_to_column("model") %>%
    slice(2) %>%
    transmute(
      specification = spec_label,
      comparison = "Expanded versus conceptual",
      chi_square_difference = `Chisq diff`,
      df_difference = `Df diff`,
      p_value = `Pr(>Chisq)`
    )

  key_paths <- coefficients %>%
    filter(
      op == ":=" |
        (lhs == "prep_used" & rhs %in% c(
          "stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp",
          "hiv_testing_frequency", "risk_behavior"
        )) |
        (lhs == "stigma_ahs" & rhs %in% c("stigma_gss", "stigma_family", "seehcp")) |
        (lhs == "out_gid" & rhs == "stigma_ahs")
    )

  list(fits = fits, fit_table = fit_table, coefficients = coefficients,
       key_paths = key_paths, difference_test = difference_test,
       conceptual_syntax = common_model,
       expanded_syntax = expanded_model)
}

message("Fitting national pooled models...")
pooled <- fit_pair(analysis, "National pooled", FALSE)
message("Fitting models with state fixed effects...")
state_adjusted <- fit_pair(analysis, "National, state fixed effects", TRUE)

fit_comparison <- bind_rows(pooled$fit_table, state_adjusted$fit_table)
nested_model_comparison <- bind_rows(
  pooled$difference_test,
  state_adjusted$difference_test
)
all_coefficients <- bind_rows(pooled$coefficients, state_adjusted$coefficients)
key_pathways <- bind_rows(pooled$key_paths, state_adjusted$key_paths)

direct_path_summary <- key_pathways %>%
  filter(
    model == "expanded", lhs == "prep_used",
    rhs %in% c("stigma_gss", "stigma_family")
  ) %>%
  transmute(
    specification,
    pathway = paste(rhs, "-> prep_used"),
    probit_coefficient = est,
    standard_error = se,
    p_value = pvalue,
    ci_lower = ci.lower,
    ci_upper = ci.upper,
    standardized_coefficient = std.all
  )

write_csv(fit_comparison, file.path(output_dir, "model_fit_comparison.csv"))
write_csv(nested_model_comparison, file.path(output_dir, "nested_model_comparison.csv"))
write_csv(key_pathways, file.path(output_dir, "key_pathway_estimates.csv"))
write_csv(all_coefficients, file.path(output_dir, "all_parameter_estimates.csv"))
write_csv(direct_path_summary, file.path(output_dir, "direct_broader_stigma_to_prep.csv"))
writeLines(pooled$conceptual_syntax, file.path(output_dir, "conceptual_model.syntax.txt"))
writeLines(pooled$expanded_syntax, file.path(output_dir, "expanded_model.syntax.txt"))

saveRDS(
  list(
    pooled = pooled$fits,
    state_fixed_effects = state_adjusted$fits,
    analysis_metadata = list(
      input_rds = input_rds,
      n_eligible = nrow(analysis),
      production_filters = "HIV-negative/unknown; AMIS 2017-2024; amis_15/amis_16 excluded",
      testing_construct = "test2yrs capped at 4 and divided by 4",
      weight = "state-period raked weights scaled to state MSM population"
    )
  ),
  file.path(output_dir, "national_cross_sectional_sem_fits.rds")
)

cat("\nNational cross-sectional SEM complete.\n")
cat("Eligible respondent records:", format(nrow(analysis), big.mark = ","), "\n")
cat("\nDirect broader-stigma paths in expanded models:\n")
print(direct_path_summary, n = Inf)
cat("\nModel fit comparison:\n")
print(fit_comparison, n = Inf)
cat("\nOutputs written to:", normalizePath(output_dir), "\n")

