# Survey-weighted cross-sectional associations within demographic groups.
#
# For each of six marginal groups (Black non-Hispanic, Hispanic/Latino, White
# non-Hispanic; ages <=24, 25-34, >=35) and each modeled outcome, fits a
# quasibinomial svyglm of the outcome on its permitted predictors (the 18
# pathways of the state model) plus survey year, education, insurance, recent
# homelessness and illicit drug use (urbanicity is not available in AMIS 2023), using the national
# raking weights (scripts/02). HIV testing is the frequency measure defined in
# scripts/00_clean_amis.R.
#
# Output: results/02_inputs/group_associations.xlsx
#   key_terms    pathway predictors (read by scripts/14)
#   all_terms    every coefficient, including covariates
#   diagnostics  fit status and complete-case counts per model
#
# Run from the repository root: Rscript scripts/05_group_associations.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(survey)
  library(broom)
  library(openxlsx2)
})


output_xlsx <- "results/02_inputs/group_associations.xlsx"
dir.create("results/02_inputs", recursive = TRUE, showWarnings = FALSE)

groups <- tribble(
  ~population_id, ~population_label, ~column, ~value,
  "black_nh", "Black, non-Hispanic", "race_ethnicity", "Black, non-Hispanic",
  "hispanic_latino", "Hispanic/Latino", "race_ethnicity", "Hispanic/Latino",
  "white_nh", "White, non-Hispanic", "race_ethnicity", "White, non-Hispanic",
  "age_under_24", "<=24", "age_group", "<=24",
  "age_25_34", "25-34", "age_group", "25-34",
  "age_35_plus", "35+", "age_group", "35+"
)

pathways <- read_csv("config/sem_structural_priors_v3.csv", show_col_types = FALSE) %>%
  transmute(predictor = source, outcome = target, relationship = paste0(source, " -> ", target))
covariates <- c("year_scaled", "school4", "insur_bin", "homeless_p12m", "illicit")

national <- readRDS("results/01_weights/national_raked.rds")
amis <- national$variables %>%
  mutate(
    analysis_weight = national_weight,
    year_scaled = survey_year - 2017,
    across(c(school4, insur_bin, homeless_p12m, illicit), factor)
  )

fit_group <- function(df, outcome) {
  terms <- c(pathways$predictor[pathways$outcome == outcome], covariates)
  terms <- terms[map_lgl(terms, ~ n_distinct(na.omit(df[[.x]])) >= 2)]
  model_df <- df %>% select(all_of(c(outcome, terms, "analysis_weight"))) %>% drop_na()
  counts <- tibble(n_complete = nrow(model_df), events = sum(model_df[[outcome]] == 1),
                   nonevents = sum(model_df[[outcome]] == 0))
  if (counts$n_complete < 30 || counts$events < 10 || counts$nonevents < 10) {
    return(list(terms = tibble(), diagnostic = mutate(counts, status = "skipped: insufficient data")))
  }
  formula <- reformulate(terms, outcome)
  fit <- svyglm(formula, design = svydesign(ids = ~1, weights = ~analysis_weight, data = model_df),
                family = quasibinomial())
  list(
    terms = tidy(fit) %>%
      mutate(odds_ratio = exp(estimate), conf_low_or = exp(estimate - qnorm(0.975) * std.error),
             conf_high_or = exp(estimate + qnorm(0.975) * std.error),
             formula = deparse1(formula), !!!counts),
    diagnostic = mutate(counts, status = "fit", formula = deparse1(formula))
  )
}

fits <- groups %>%
  crossing(outcome = unique(pathways$outcome)) %>%
  mutate(fit = pmap(list(column, value, outcome), function(column, value, outcome) {
    fit_group(amis[amis[[column]] == value, ], outcome)
  }))

all_terms <- fits %>%
  transmute(population_id, population_label, outcome, terms = map(fit, "terms")) %>%
  unnest(terms)
key_terms <- all_terms %>%
  inner_join(pathways, by = c("outcome", "term" = "predictor")) %>%
  select(population_id, population_label, relationship, outcome, term, estimate, std.error, p.value,
         odds_ratio, conf_low_or, conf_high_or, n_complete, events, nonevents) %>%
  arrange(relationship, population_id)
diagnostics <- fits %>%
  transmute(population_id, outcome, diagnostic = map(fit, "diagnostic")) %>%
  unnest(diagnostic)

wb <- wb_workbook()
for (sheet in c("key_terms", "all_terms", "diagnostics")) {
  wb$add_worksheet(sheet)
  wb$add_data(sheet, get(sheet))
}
wb$save(output_xlsx, overwrite = TRUE)
message("Fitted ", sum(diagnostics$status == "fit"), " of ", nrow(diagnostics), " group models; wrote ", output_xlsx)
