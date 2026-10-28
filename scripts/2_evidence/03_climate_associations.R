# Contextual ("climate") stigma associations with PrEP use: the basis for the contextual upper bound
# (scripts/3_model/05 --contextual-prep-pr 0.80 and 0.70) and the reason it is an upper bound.
#
# Survey-weighted Poisson models of past-year PrEP use among respondents in the primary states, with
# the respondent's own anticipated healthcare stigma (AHS) and the stigma climate of their state-period:
# the weighted share of the other respondents in the same state-period reporting AHS (leave-one-out;
# adjusted for the 2022 item change as in scripts/1_data/03), and in the fuller model also general social and
# family stigma. Climate effects are per 10 percentage points. Confounders as in scripts/2_evidence/01:
# insurance, age, race/ethnicity, education, risk behavior, AMIS cycle. Each model is fitted across
# states (climate contrasts between and within states) and within states only (state fixed effects).
# State weights, each state-period weighted to its sample size (as in scripts/2_evidence/01). Standard errors are
# clustered by state: the climate is shared by everyone in a state-period and persists over time.
#
# Inputs: results/01_weights/state_raked.rds, results/01_weights/ahs_wording_shift.csv (scripts/1_data/03),
#         results/02_inputs/state_screen/primary_states.csv (scripts/1_data/05)
# Output: results/02_inputs/climate_associations.csv (model, contrast, term, prevalence ratio, 95% CI)
#
# Run from the repository root: Rscript scripts/2_evidence/03_climate_associations.R

suppressPackageStartupMessages({
  library(tidyverse)
  library(survey)
})

output_path <- "results/02_inputs/climate_associations.csv"
new_wording_first_year <- 2022
confounders <- c("insur_bin", "agecat", "rdrace", "school4", "risk_behavior", "cycle")
wording_shift <- read_csv("results/01_weights/ahs_wording_shift.csv", show_col_types = FALSE)$shift_log_odds
states <- read_csv("results/02_inputs/state_screen/primary_states.csv", show_col_types = FALSE)$state

# Weighted share of the OTHER respondents in the same state-period reporting the measure:
#
#   climate_i = sum_{j != i}(w_j y_j) / sum_{j != i}(w_j).
#
# Removing respondent i prevents their own stigma response from mechanically appearing in both the
# individual exposure and their contextual exposure. With large state-period samples this differs only
# slightly from the ordinary weighted state-period mean, but it keeps the two concepts distinct.
leave_one_out <- function(y, w) {
  ok <- !is.na(y)
  total <- sum(w[ok] * y[ok])
  weight <- sum(w[ok])
  (total - if_else(ok, w * y, 0)) / (weight - if_else(ok, w, 0))
}

amis <- readRDS("results/01_weights/state_raked.rds")$variables %>%
  filter(state_calc %in% states) %>%
  group_by(state_calc, year_group) %>%
  mutate(
         # State-period raking weights sum to one. Multiplying every weight in the cell by its sample size
         # leaves within-cell weighted means unchanged, but gives larger state-period samples more total
         # influence in the pooled respondent-level regression (the same convention as scripts/2_evidence/01).
         weight = raked_weight * n(),

         # Share of the other respondents who received the post-2022 AHS wording. This is one in a cell
         # containing only the new wording, zero with only the old wording, and fractional in a mixed cell.
         new_share = leave_one_out(as.numeric(survey_year >= new_wording_first_year), weight),

         # Leave-one-out state-period AHS prevalence on its observed wording. Move it back to the old-item
         # scale on the logit scale, in proportion to the cell's share receiving the new wording:
         #   adjusted climate = expit[logit(raw climate) - wording_shift * new_share].
         # Multiplication by 10 makes one model unit equal to a 10-percentage-point higher stigma climate.
         ahs_raw = leave_one_out(stigma_ahs, weight),
         climate_ahs = 10 * plogis(qlogis(ahs_raw) - wording_shift * new_share),

         # GSS and family stigma did not undergo the AHS item change. They use their leave-one-out weighted
         # prevalences directly and are also scaled so one unit represents 10 percentage points.
         climate_gss = 10 * leave_one_out(stigma_gss, weight),
         climate_family = 10 * leave_one_out(stigma_family, weight)) %>%
  ungroup() %>%
  mutate(cycle = factor(survey_year), state_calc = factor(state_calc),
         across(c(insur_bin, agecat, rdrace, school4), factor))

# Both specifications include respondents' own AHS, so climate_ahs asks whether the surrounding
# state-period stigma environment is associated with PrEP beyond the respondent's personal AHS report.
# The fuller model asks whether that association remains after accounting for the measured general-social
# and family-stigma climates. Individual GSS and family stigma are not included, so their climate terms do
# not have as clean a within-person/contextual separation as the AHS terms.
models <- list(
  ahs_climate = c("stigma_ahs", "climate_ahs"),
  all_climates = c("stigma_ahs", "climate_ahs", "climate_gss", "climate_family")
)

# "across_states" is the pooled model without state fixed effects: it uses both between-state and
# within-state variation and therefore is not a pure between-state contrast. "within_states" adds a fixed
# intercept for every state, absorbing all stable differences between states. Its climate coefficient is
# consequently identified by changes in a state's climate across periods, net of the common cycle effects.
contrasts <- list(
  across_states = character(),
  within_states = "state_calc"
)

results <- imap_dfr(models, function(terms, model) {
  imap_dfr(contrasts, function(extra, contrast) {
    # Complete-case dataset for this specification. The all-climates model can therefore use fewer
    # respondents than the AHS-only model if GSS or family stigma is missing.
    d <- amis %>% select(prep_used, all_of(c(terms, confounders, "state_calc")), weight) %>% drop_na()

    # Modified Poisson regression for the binary PrEP outcome. Exponentiating a coefficient gives an
    # adjusted prevalence ratio. For a climate term, the PR compares otherwise similar respondents at
    # state-period climates differing by 10 percentage points. State-clustered design-based standard errors
    # allow outcomes to be correlated within states, including across their four periods.
    fit <- svyglm(reformulate(c(terms, confounders, extra), "prep_used"),
                  design = svydesign(ids = ~state_calc, weights = ~weight, data = d), family = quasipoisson())

    # Report only the individual-AHS and climate coefficients, not the adjustment-variable coefficients.
    b <- coef(fit)[terms]
    se <- sqrt(diag(vcov(fit))[terms])
    tibble(model = model, contrast = contrast, term = terms, n = nrow(d),
           pr = exp(b), low = exp(b - 1.96 * se), high = exp(b + 1.96 * se))
  })
})

write_csv(results, output_path)
print(results %>% mutate(across(c(pr, low, high), ~ round(.x, 2))), n = Inf)
