# National raking weights, used by the national AHS wording series (scripts/2_evidence/02) and the
# demographic-group analyses (scripts/5_demographic/01).
#
# Rakes AMIS respondents within each model period to national MSM margins (age
# group, race/ethnicity, education, Census region; inputs/national_msm_margins.csv,
# from MSM-Population-Estimates raking_targets/build_amis_raking_targets.R) plus a
# joint age group x PrEP margin (scripts/1_data/02_build_cdc_stocks.R). No trimming.
#
# Periods pair AMIS cycles (year_group from scripts/1_data/01_clean_amis.R): 2017-2018
# (AMIS 2017, 2018), 2019-2020 (AMIS 2019), 2021-2022 (AMIS 2021, 2022), 2023-2024
# (AMIS 2023, 2024). Each period's PrEP target pools CDC/AIDSVu users and eligible
# persons over the matching calendar years (2023-2024: 2023 only, the latest
# available). The demographic targets are the same in every period.
#
# Respondents of other or multiple race/ethnicity (no national target) or missing
# a raking variable are excluded. Few respondents are aged <=24 from AMIS 2021 on
# (median age 42-51 vs 26-35 earlier), so that age group carries large weights in
# 2021-2024.
#
# Run from the repository root: Rscript scripts/1_data/07_national_weights.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(survey)
})

model_periods <- c(
  "2017-2018" = 2018L,
  "2019-2020" = 2020L,
  "2021-2022" = 2022L,
  "2023-2024" = 2024L
)

output_rds <- "results/01_weights/national_raked.rds"
diagnostics_csv <- "results/01_weights/national_raking_diagnostics.csv"
calibration_csv <- "results/01_weights/national_prep_calibration_table.csv"
dir.create("results/01_weights", recursive = TRUE, showWarnings = FALSE)

year_groups <- names(model_periods)
rake_control <- list(maxit = 100, epsilon = 1e-7)

# Demographic variables calibrated separately (each also has a diagnostic below).
raking_domains <- c("age_group", "race_ethnicity", "education", "census_region")

# ---- Targets -------------------------------------------------------------------

demographic_targets <- read_csv("inputs/national_msm_margins.csv", show_col_types = FALSE) %>%
  select(margin_domain, margin_level, target_prop) %>%
  crossing(year_group = year_groups)

age_prep_targets <- read_csv(
  "results/00_cdc/national_prep_age_targets.csv",
  show_col_types = FALSE
) %>%
  mutate(
    year_group = case_when(
      Year %in% 2017:2018 ~ "2017-2018",
      Year %in% 2019:2020 ~ "2019-2020",
      Year %in% 2021:2022 ~ "2021-2022",
      Year %in% 2023:2024 ~ "2023-2024"
    )
  ) %>%
  group_by(year_group, age_group) %>%
  summarise(prep_prop = sum(prep_users) / sum(prep_eligible), .groups = "drop")
stopifnot(setequal(age_prep_targets$year_group, year_groups))

# ---- Respondents ---------------------------------------------------------------

# Read the cleaned AMIS respondent file, harmonize the national raking
# variables, construct the joint age-by-PrEP category, and retain complete
# observations on every calibration dimension.
amis <- read_csv("results/00_amis/amis_analytic.csv.gz", show_col_types = FALSE) %>%
  mutate(
    age_group = factor(
      case_when(
        agecat == "18_24" ~ "<=24",
        agecat == "25_34" ~ "25-34",
        agecat %in% c("35_44", "45_54", "55_up") ~ "35+"
      ),
      levels = c("<=24", "25-34", "35+")
    ),
    race_ethnicity = factor(
      case_when(
        rdrace == "black" ~ "Black, non-Hispanic",
        rdrace == "white" ~ "White, non-Hispanic",
        rdrace == "hisp" ~ "Hispanic/Latino"
      ),
      levels = c(
        "Black, non-Hispanic",
        "White, non-Hispanic",
        "Hispanic/Latino"
      )
    ),
    education = factor(school4, levels = c("lths", "hs", "college", "grad")),
    census_region = factor(
      c("Northeast", "Midwest", "South", "West")[region],
      levels = c("Northeast", "Midwest", "South", "West")
    ),
    prep_status = factor(
      if_else(prep_used == 1, "prep_used", "prep_not_used"),
      levels = c("prep_not_used", "prep_used")
    ),
    age_prep = interaction(age_group, prep_status, sep = " | ", drop = TRUE)
  ) %>%
  filter(
    !is.na(age_group),
    !is.na(race_ethnicity),
    !is.na(education),
    !is.na(census_region),
    !is.na(prep_status)
  )

# ---- Raking --------------------------------------------------------------------

demographic_margin <- function(yg, domain, present) {
  demographic_targets %>%
    filter(year_group == yg, margin_domain == domain, margin_level %in% present) %>%
    transmute(!!domain := margin_level, Freq = target_prop / sum(target_prop))
}

# Joint age x PrEP margin: each age group's population share split by its PrEP target.
age_prep_margin <- function(yg) {
  joint_margin <- demographic_targets %>%
    filter(year_group == yg, margin_domain == "age_group") %>%
    inner_join(
      filter(age_prep_targets, year_group == yg),
      by = c("margin_level" = "age_group")
    ) %>%
    crossing(prep_status = c("prep_not_used", "prep_used")) %>%
    transmute(
      age_prep = paste(margin_level, prep_status, sep = " | "),
      Freq = target_prop * if_else(prep_status == "prep_used", prep_prop, 1 - prep_prop)
    )

  joint_margin %>%
    mutate(Freq = Freq / sum(Freq))
}

rake_period <- function(df, yg) {

  # Construct one population-target table for each demographic variable.
  demographic_margins <- lapply(
    raking_domains,
    function(domain) {
      observed_categories <- as.character(df[[domain]])

      demographic_margin(
        yg,
        domain,
        observed_categories
      )
    }
  )

  # Construct the joint age-group-by-PrEP target table.
  joint_age_prep_margin <- age_prep_margin(yg)

  # Combine all target tables into the list required by survey::rake().
  population_margins <- append(
    demographic_margins,
    list(joint_age_prep_margin)
  )

  # Construct the matching sample-variable formulas.
  demographic_formulas <- lapply(
    raking_domains,
    function(domain) {
      as.formula(paste0("~", domain))
    }
  )

  # Add the formula corresponding to the joint age-by-PrEP target.
  sample_margins <- append(
    demographic_formulas,
    list(~age_prep)
  )

  # Begin with equal respondent weights.
  initial_design <- svydesign(
    ids = ~1,
    data = df,
    weights = ~1
  )

  # Iteratively calibrate the sample to the population targets, retaining a
  # convergence flag while suppressing only the corresponding warning.
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

  # Attach the resulting national weight to each respondent.
  df %>%
    mutate(national_weight = weights(raked_design), rake_converged = converged)
}

raked_respondents <- amis %>%
  group_by(year_group) %>%
  group_modify(~ rake_period(.x, .y$year_group)) %>%
  ungroup()

national_design <- svydesign(
  ids = ~1,
  data = raked_respondents,
  weights = ~national_weight
)

saveRDS(national_design, output_rds)

# ---- Diagnostics ---------------------------------------------------------------

demographic_fit <- map_dfr(
  raking_domains,
  function(domain) {
    raked_respondents %>%
      group_by(
        year_group,
        margin_level = as.character(.data[[domain]])
      ) %>%
      summarise(
        n = n(),
        weight_total = sum(national_weight),
        .groups = "drop_last"
      ) %>%
      mutate(
        observed = weight_total / sum(weight_total),
        margin_domain = domain
      ) %>%
      ungroup() %>%
      select(-weight_total)
  }
) %>%
  left_join(
    demographic_targets,
    by = c("year_group", "margin_domain", "margin_level")
  ) %>%
  group_by(year_group, margin_domain) %>%
  mutate(target = target_prop / sum(target_prop, na.rm = TRUE)) %>%
  ungroup() %>%
  transmute(
    year_group,
    margin_domain,
    margin_level,
    n,
    observed,
    target,
    diff = observed - target
  )

weight_summary <- raked_respondents %>%
  group_by(year_group) %>%
  summarise(
    n = n(),
    n_eff = sum(national_weight)^2 / sum(national_weight^2),
    weight_cv = sd(national_weight) / mean(national_weight),
    weighted_prep = weighted.mean(prep_used, national_weight),
    rake_converged = first(rake_converged),
    .groups = "drop"
  )
if (!all(weight_summary$rake_converged)) {
  warning("National raking did not converge for: ",
          paste(weight_summary$year_group[!weight_summary$rake_converged], collapse = ", "))
}

write_csv(bind_rows(demographic_fit, weight_summary), diagnostics_csv, na = "")

# Supplementary Table S2: effect of PrEP calibration on weighted PrEP prevalence by age.
raked_respondents %>%
  group_by(year_group, age_group = as.character(age_group)) %>%
  summarise(
    eligible_n = n(),
    unweighted_prep = mean(prep_used),
    weighted_prep = weighted.mean(prep_used, national_weight),
    .groups = "drop"
  ) %>%
  left_join(
    age_prep_targets %>% rename(calibration_target = prep_prop),
    by = c("year_group", "age_group")
  ) %>%
  mutate(weighted_minus_unweighted_pp = 100 * (weighted_prep - unweighted_prep)) %>%
  write_csv(calibration_csv)

print(weight_summary)
message("Max absolute demographic margin error: ", signif(max(abs(demographic_fit$diff)), 3))
