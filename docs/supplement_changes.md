# Supplement: changes to make

Changes needed in the Supplementary Methods draft (EEE_SD_Supplement, October 2026) so that it matches
the final analysis. Details and numbers are in `docs/analysis_decisions.md` (section names given in
brackets). Values marked *regenerate* change with the final outputs; take them from the results files
named, not from the draft.

## S1. AMIS inputs

### S1.1 Sample and periods
- COVID exclusion: the AMIS 2020 cycle is not used (fielded October 2020 - January 2021); no other
  cycle had respondents in the former 1 March 2020 - 1 July 2021 window. Replace the date-window
  sentence. Including AMIS 2020 is a sensitivity analysis (S7). [System decomposition ...; AMIS cleaning]
- Add AMIS 2024 (fielded November 2024 - August 2025). [AMIS 2024 and cycle-based periods]
- Periods follow AMIS cycles, not calendar year: 2017-2018 = AMIS 2017 + 2018; 2019-2020 = AMIS 2019
  only; 2021-2022 = AMIS 2021 + 2022; 2023-2024 = AMIS 2023 + 2024 (data collected through August 2025).
- AMIS 2015-2016 excluded because their stigma items do not ask whether the experience happened in the
  last 6 months (sharpen "different response options").
- Ages above 99 excluded before raking (4 respondents).

### S1.2 Measures and Table S1
- **Anticipated healthcare stigma: give both wordings.** 2017-2021: "...because you worry someone may
  learn you have sex with men"; 2022 onward: "...because you worry someone may treat you poorly because
  you have sex with men". The table shows only the second.
- Fill the stigma "Variable construction" cells:
  - all stigma composites refer to the last 6 months (item value 1);
  - checklist scoring: 1 if any item endorsed, 0 if none endorsed and at least one answered, missing
    only if none answered;
  - general social stigma: physical violence counts only if in the last 6 months and attributed to
    having sex with men;
  - AMIS 2024 lacks the sexual-violence items (stigma_d3, d4), so its general social stigma has four
    components.
- PrEP use: **"prefer not to answer" and "don't know" are missing, not 0** (the table is wrong). PrEP-
  unaware respondents and those never on PrEP are 0.
- Disclosure: "told nobody" and "does not apply" (no provider) are 0.
- Risk behavior: men with one male partner are routed to partner-specific items; they count as at risk
  if condomless with a partner of unknown or positive status. Respondents reporting condomless sex but
  routed to neither path are missing.
- HIV testing: also 0 when the last test was more than 24 months ago and the count is missing.
- General skip rule: a legitimate skip takes the value implied by the gating answer; prefer-not, don't-
  know and split-sample design skips are missing.

### New subsection: the 2022 change to the AHS items
- What changed (both items, AMIS 2022 onward).
- Adjustment: state-period AHS moved back by 0.452 log-odds (SE 0.105; age-adjusted AMIS 2022 vs 2021
  difference within the same state-period) times the share of AMIS 2022+ respondents.
- Why it is conservative: measures whose wording did not change rose as much from 2021 to 2022
  (experienced healthcare stigma +0.59, general social +0.64, family +0.24 log-odds, vs AHS +0.45), so part
  of the AHS rise is probably real and removing all of it understates AHS after 2022, and the scenario
  benefits with it. (scripts/08c; results/02_inputs/wording/shift_2022.csv)
- Unadjusted AHS is a sensitivity analysis (S7).
[The AHS wording shift may be a post-COVID rebound; Wording adjustment kept as the conservative primary]

### S1.3-S1.4 Weighting
- **The state models use state weights**, which the draft does not describe: state x period raking to
  state MSM margins (age, race/ethnicity, education), weights capped at the state-period 1st/99th
  percentiles, no respondents dropped. The drop rule and unadjusted raked weights are weighting
  sensitivity analyses. Add a subsection. [Inputs and weighting; Weighting rule and Georgia]
- National weights (as described) are used only for national series (Fig. 1B, wording analyses) and the
  demographic descriptives.
- Margins rebuilt from the MSM-Population-Estimates repository; education categories now match AMIS
  (less than HS / HS / some college or associate's / bachelor's or higher), men 18+. [Demographic raking
  margins rebuilt]
- Table S2: *regenerate* (2017-2018 eligible N falls with AMIS 2016 removed). Check the 2023-2024 rows:
  the draft says "not applied"; the national weights pool calendar-year PrEP targets with 2023 from the
  2017-2022 mean imputation. Confirm which applies and state it.
- CVs of the final national weights: 1.28-1.51 (*regenerate* the design-effect range).
- PrEP-eligible denominator x 2: rationale still to be documented (S4 reference in the draft points
  to a section that does not exist yet). [CDC stocks rebuilt]
- Your note: S1.3 is the main place to shorten.

### S1.6 State screen and Table S3
- **24 primary states**, same rule (all 32 cells, median n_eff >= 60, 25th percentile >= 40): the draft's
  17 plus AL, CO, MN, MO, SC, UT, WI. List from results/02_inputs/state_screen/primary_states.csv.
- Stricter screen: 22 states (drops MN and UT).
- Table S3: *regenerate* from state_effective_n_screen.csv.

### S1.7
- 24 states; otherwise unchanged.

## S2. Dynamic model (substantial rewrite)

- **State anchoring.** Each measure is centred at the state's own mean logit over the observed periods
  (per draw): a state-specific intercept, so pathways act on departures from a state's usual level. The
  fixed reference probabilities (Table S4 values: AHS 0.053, family 0.136, general social 0.157,
  disclosure 0.535, visit 0.781, risk 0.180, PrEP 0.057, testing 0.395; *regenerate* from
  results/02_inputs/reported_constants.csv) now serve the benchmark and the evidence conversion.
  Rewrite S2.2.1. [State-anchored revision]
- **Drift (S2.3).** PrEP drift centre 0.420 (not 0.468; mean two-year change in logit PrEP across the 24
  states), scale 1.0 (not 0.35): each state has its own PrEP trend. New weak zero-centred AHS drift
  (scale 0.35). Risk drift unchanged. [Loose PrEP drift; State-anchored revision]
- **Structure (S2.4).**
  - Healthcare contact -> AHS and PrEP -> AHS have no sign constraint (centre 0, scale 0.7).
  - The nine PrEP-cluster pathways (PrEP <-> risk, PrEP <-> testing, risk -> disclosure, disclosure ->
    PrEP and testing, risk -> testing, contact -> testing) centred at 0.05 (not 0.15), scale 0.3.
  - Re-count the permitted pathways and update Supplementary Fig. S1.
  - Add the probability-scale table showing what 0.05 / 0.10 / 0.15 mean. [Pathway and persistence priors]
- **Persistence.** PrEP 0.85 (pooled AR(1) 0.846, SE 0.029, 24 states; not 0.75 / 0.746); contact 0.95;
  general social and family 0.85; scale 0.2. Untargeted diagonals (AHS, disclosure, risk, testing) are
  non-negative. AHS has no persistence target by design (state the rationale).
- **New subsection: the evidence constraint.** The model's total AHS -> PrEP response over 6 steps is
  constrained to match each state's survey association: old-wording cycles (2017-2021), access-adjusted
  prevalence ratio, random-effects mean 0.55 (0.45-0.67), state ratios pooled by empirical Bayes,
  converted to a centred-logit slope; per-draw ratio from N(log ratio, posterior SE). It constrains the
  fitted relationships; it does not set the scenario results (S4). [Evidence target on the old AHS
  wording]
- **Settings.** Stability penalty weight 10,000 (not 10); maxiter 5,000 (not 500); diagonal bounds
  [0, 3] on J. [Model specification; Optimizer iteration cap]
- **S2.7.2 starting values.** The code uses three deterministic starts and keeps the lowest loss: (1) a fixed
  default (persistence 0.5-1.0, pathways below 0.2 in their permitted direction, seed 0), (2) the prior centres,
  (3) the parent unit's fitted J (division for a state, nation for a division). The starts differ by more than
  1% of the loss in about 41% of primary state fits (decision log, full rerun 2026-10-22): state the multi-start
  and that individual pathway entries are weakly identified while the total AHS -> PrEP response is pinned by
  the evidence constraint. Optimizer success refers to the best start.
- **S2.6.** The code implements only the base shrinkage rule, weight = 5 x 300 / n (config/shrinkage.py,
  models/sbm/estimation/shrinkage.py); the draft's extra branches (multiplier for n 250-350, 500,000 below 250) do
  not exist. Describe the base rule only. The 24 primary states have n 565 (AL) to 4,522, weights 0.33-2.65.
- Your note: S2.5 and S2.6 reviewed. Checked: the Lyapunov innovation covariance (0.16^2 x diag of the
  covariance target), wX = 10 and wR = 0.01 match the code (config/optimization.py).

## S3. Epidemic model

- Check the calibration years against the final refit (the full rerun fitted 2017-2024 diagnoses; the
  draft says 2017-2022) and the post-2021 diagnosis multiplier.
- *regenerate* all diagnostics (draft R-hat 1.006, ESS 567/760, coverage 96.7%, etc. are from an older
  run; the 2026-10-22 refit: 0 divergences, max R-hat 1.012, min bulk ESS 586). [Full rerun]
- Your notes: MSM naming in S3.1, Fig. Sx, Table Sx, formatting in S3.6.2.

## S4. Scenarios (rewrite)

- **Projection start.** Each draw projects from its own observed 2024 values (not the fitted system from
  2018).
- **Scenario set (replace Table S4).** Halve AHS from its 2024 level; stigma-resilient PrEP delivery
  (half of the PrEP gain from reducing AHS to the benchmark, AHS and everything else at baseline); halve
  general social and family stigma. Sensitivity: AHS reduced to the benchmark (5th percentile, 5.3%).
  Supplement: AHS held at or below its 2024 level. Remove: pathway attenuation (S4.2.2), combined package
  (S4.2.3), fixed doses, healthcare-generated stigma scenarios.
- **Cuts.** Relative to the state's 2024 level, ramped over three steps (2028, 2030, 2032), held, and
  never above the baseline path (the draft says relative to the contemporaneous baseline).
- **PrEP translation (the core of the final design, missing from the draft).** Scenario PrEP =
  P_b (1 - A_s (1 - r)) / (1 - A_b (1 - r)), with the draw's state prevalence ratio r, at the projected
  levels; the epidemic model translates it. [Final design: static evidence translation]
- **Relationship continued / not continued.** Projections with the fitted PrEP -> AHS relationship
  continued (primary) or removed; the prespecified rule for choosing the primary and its result
  (placebo increment 33% of the primary; placebo coefficient -0.04).
- **Contextual upper bound.** Climate prevalence ratio 0.80 / 0.70 per 10 points of AHS applied to
  baseline PrEP.
- **Outcomes.** 2036 incidence (% change), cumulative infections averted 2025-2036 (number and % of
  baseline), infections averted per 100,000 MSM per year (fixed state MSM population x 12 years).
- **Remove S4.5** (response profiles).
- Keep: interpolation of the two-year grid to annual inputs for the epidemic model.

## S5. Uncertainty

- Add the per-draw prevalence ratio (S2 evidence constraint and S4 translation).
- **Rewrite S5.4.** Sample k now uses the same bootstrap replicate and the same epidemic posterior draw
  in every state, so pooled (summed) results are joint realizations. Pooled estimates are sums over the
  24 states within draws, not a separately fitted national model. Limitation: the state prevalence
  ratios are drawn independently, so uncertainty in the pooled mean ratio is not shared across states.
  [Draws synchronized across states, 2026-10-26]
- S5.5: summaries are median, interquartile range and 95% interval (unchanged).

## S6. Population analyses

- **Remove** the demographic-group dynamic models (S6.2-S6.5, including the California Hispanic/Latino
  analysis). [Population analyses]
- Replace with a short descriptive section: weighted 2017-2024 stigma and PrEP trajectories by group with
  bootstrap intervals (F8). The cross-sectional group associations are not reported (decided 2026-10-30).
- Limitations sentence: group models were explored, but annual subgroup data were too imprecise to
  identify group-specific pathway structure robustly.

## S7. Sensitivity analyses (rewrite to match table_sensitivity.csv)

- Relationship not continued; AHS temporal placebo (continued and not continued); association from all
  AMIS cycles; AHS not adjusted for the 2022 item change; lower and higher persistence; AHS reduced to the
  benchmark; contextual upper bound (0.80, 0.70); AMIS 2020 included; stricter state screen; weighting
  (drop rule, unadjusted raked weights).
- Model-specification checks (supplement; they change only the projected baselines under the final design):
  trajectory fit only (covariance weight 0), covariance weight 0.40, no geographic shrinkage, zero PrEP drift.
  pipeline/state_fit.py now defines only these and the five reported specifications; the draft's Supplementary
  Fig. 4 is from the old model.
- Remove: fixed-dose comparison (S7.5), response-profile thresholds (S7.6, second paragraph).

## Supplementary figures and tables: decided set (2026-10-26)

Replaces Data Tables 5-10 and Supplementary Figures 1-4. Numbering is assigned at assembly.

Figures:
- F1 Model framework: AMIS inputs -> dynamic model (baselines) -> prevalence-ratio translation of the
  scenario -> epidemic model. To redraw for the final design.
- F2 Permitted pathways with signs and prior strength. Author's figure; a draft generated from the config
  is in results/sections/supplement/figS_pathways (scripts/4_results/supplement.py).
- F3 Dynamic-model fit: one-step-ahead fitted vs observed, eight measures, 2020/2022/2024
  (results/sections/supplement/figS_dynamic_fit; scripts/4_results/supplement.py). Done. In-sample.
  Mean absolute error 1.0-1.1 pp for PrEP and AHS and 2.2-3.8 pp for the others, at or below the survey
  sampling SD of the observed values (2.6-4.5 pp) for every measure; lower than carrying the last value
  forward (3.2-5.2 pp); 88-100% of observed values consistent with the fit once their own sampling error is
  counted. Text: the remaining scatter is survey noise the model does not (and should not) reproduce.
- F4 Epidemic calibration: posterior vs observed period means, diagnoses and incidence, primary states
  (figS_epi_calibration). Done. Diagnoses in the 90% interval: 100%, 100%, 75% (2017-19, 2021-22,
  2023-24); incidence 88%, 79%; median absolute error 1.9-5.0%. Replaces the draft's S3.8 numbers.
- F5 Observed PrEP-AHS paths by state (results/sections/coevolution/figS_coevolution). Done.
- F6 State effects with intervals: projected 2036 AHS and incidence reduction
  (results/sections/ahs_reduction/figS_state_effects). Done.
- F8 Demographic descriptive trajectories (results/08_demographic/group_trajectories; scripts/14). Done;
  AHS shown as measured with the reworded years (AMIS 2022+) shaded.

Tables:
- T1 Measures and coding (corrected Table S1). Text.
- T2 PrEP calibration (Table S2). Regenerate; candidate for shortening.
- T3 State screen (Table S3). Data in results/02_inputs/state_screen/.
- T4 Model specification (results/sections/supplement/tableS_model_specification.csv). Done.
- T5 Scenario definitions (replaces the draft's Table S4). Text.
- T6 Wording: 2022 vs 2021 shift for AHS (the 0.452 removed by the model) and the three unchanged measures
  (results/02_inputs/wording/shift_2022.csv). Done.
- T7 State effects: 2036 incidence change, infections averted (number, %, per 100,000 MSM per year) with
  95% intervals, and the share of draws with projected 2036 AHS above 30% / 50%, both projections
  (results/sections/ahs_reduction/tableS_state_effects.csv). Done.
- T9 Sensitivity analyses (results/sections/sensitivity/table_sensitivity.csv). Done.
- T11 Climate associations across vs within states (results/02_inputs/climate_associations.csv;
  scripts/08d; standard errors clustered by state). Done. AHS climate per 10 points 0.69 (0.48-0.99) across
  states, 1.04 (0.92-1.18) within; with social and family climate 0.78 (0.53-1.15) vs 1.05 (0.90-1.22). The
  between-state association is not distinguishable from no effect once the other climates are adjusted for,
  and is absent within states: the contextual bound is a hypothetical upper bound (0.80, 0.70).

Not included (decided): sensitivity forest plot (duplicates T9); fitted PrEP -> AHS by state and the AHS
decomposition; full pathway matrix / heatmap and coefficient distributions (weakly identified entries);
intermediate outcomes for all eight measures; dropped scenarios and response profiles; demographic-group
dynamic models and the California Hispanic/Latino analysis; coupling routes, durability and the AHS
history counterfactual.

## Numbers after the synchronized-draw rerun (2026-10-26)

All pooled and state results were regenerated with draws synchronized across states (changes are small;
e.g. halve AHS 2036 incidence -7.5%, 95% -15.3 to -3.9). Take every number from results/sections/,
not from earlier drafts or messages.
