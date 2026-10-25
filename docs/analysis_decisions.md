# Analysis decisions (paper cleanup, October 2026)

Decisions made while consolidating the repository for the paper. Each one
changes the inputs relative to the previously archived results and needs to
be reflected in the supplement.

## Inputs and weighting

| Decision | Previous | Final | Supplement |
|---|---|---|---|
| AMIS input uncertainty | Beta sampling (`sem_input_draws_v5`) in the empirical-reference fit | 500 joint respondent-bootstrap replicates, fixed weights | S5.1 |
| State weighting | State x period raking; 1st/99th percentile trimming that dropped respondents, cutoffs pooled across periods | Same raking; weights capped at state-period 1st/99th percentiles, no respondents dropped | S1.3-S1.4 (state weighting currently undocumented) |
| National weighting | Age x PrEP national raking (demographic groups only) | Unchanged; used only for the six national demographic groups | S1.3-S1.4, S6.1 |
| California Hispanic/Latino weights | National race x PrEP raking | California state raking weights | S6.5 |
| AMIS 2015-2016 respondents | Included in raking, dropped afterward | Excluded before raking (state and national) | S1.1, S1.4 |
| Implausible ages | Retained (max 425) | Ages > 99 excluded before raking (4 respondents) | S1.1 |
| HIV testing in population regressions and covariance | Binary 12-month testing item | min(test2yrs, 4) / 4, matching S1.2 | S6.2 |
| Unused 2020 testing sheet | Built by the state workbook | Removed | - |

## State selection and references

| Decision | Final | Supplement |
|---|---|---|
| Primary precision screen | All 32 cells, median n_eff >= 60, 25th percentile n_eff >= 40 (the original rule). Under the final weights (rebuilt AMIS file, AMIS-compatible margins): 22 states - CA TX NY FL GA PA NC OH VA IL NJ MI MD WA MA IN TN MO AL CO WI MN. MN (60.2) and WI (60.4) are borderline. (Interim: median 55 was used while MA sat at 57.9 under the earlier weights.) | S1.6, Table S3 (regenerate) |
| Stricter screen | Unchanged rule (25th percentile >= 50, minimum >= 25); 21 states (adds SC; drops AL, MN) | S1.6, S7.6 |
| Additional sensitivity | Median cutoff 55: 24 states (adds SC, UT) | S7.6 |
| Centered-logit references | Recomputed: 5th percentile of weighted point estimates across the primary states and four periods (regenerate for the 22 states) | S2.2, Table S4 |

### Weighting rule and Georgia

Capping is retained as the primary rule because it keeps every respondent. In GA 2019-2020,
the original drop rule removes five AMIS 2019 respondents carrying the period's
highest weights (about 9% of period weight), including three PrEP users of "Other"
race. Dropping them lowers GA 2019-2020 PrEP from 12.5% to 8.4%, which creates a
12-point 2020-2022 PrEP gain immediately after GA's stigma decline and produces
GA's strong AHS -> PrEP pathway in the draft results. With them retained, GA's
PrEP gains are more even (+5, +7, +3 points) and the pathway is uncertain (at its
zero boundary in about half of draws). Report the drop rule and unadjusted raked
weights as weighting sensitivity analyses (relevant to co-author comment JS26).

Primary endpoint remains 2036 (full implementation from 2032 plus time for PrEP to
respond); 2032 and cumulative infections averted are reported as supplementary outcomes.

## Data-derived model constants (`scripts/08_derive_reported_constants.py`)

| Constant | Previous | Final | Supplement |
|---|---|---|---|
| PrEP drift regularization target | 0.468 (46 states, from an older `amis_summary_stats_v4.csv` weighting run) | 0.404: mean two-year change in logit PrEP, the 22 primary states (same states as the references and phi). The PrEP-only AR(1) intercept (0.522) is reported as an alternative; the mean change is the more conservative prior because cross-pathways into PrEP also produce growth, and the prior SD (0.35) lets the fit move between them. | S2.3, Table S4 |
| PrEP persistence target | 0.75, citing phi = 0.746 (SE 0.074) from an old 9-state set | 0.85, from pooled centered-logit AR(1) phi = 0.850 (SE 0.025) across the 22 primary states | S2.4, Table S4 |
| Centered-logit references | Ad hoc values in config | Derived in `scripts/08` (5th percentile, 22 primary states x 4 periods): AHS 0.054, family 0.131, general social 0.152, disclosure 0.537, visit 0.777, risk 0.185, PrEP 0.058, testing 0.395 | S2.2, Table S4 |

Previous Table S3 values also came from `amis_summary_stats_v4.csv`, not the analysed inputs.

## AMIS cleaning rebuilt from the raw files (2026-10-21)

`scripts/00_clean_amis.R` now builds the analytic respondent file from the six raw AMIS Stata
files (AMIS 2017, 2018, 2019, 2021, 2022, 2023; `inputs/restricted/amis_raw/`). It replaces
`EEE/EEE Feb 2025/EEE AMIS Clean.ipynb` -> `AMIS-Variable-Trends/clean_amis.R` ->
`combined_amis_nocovid.csv`. Validated against the old file: respondent counts, age, period,
PrEP use, healthcare visit, disclosure, binary testing, test counts, ever tested and insurance
match exactly in every cycle x state cell; all differences below are deliberate.

- The old file could not be regenerated: the saved notebook defines stigma as "yes, in the last
  6 months", but the file was produced by an earlier version using "ever" (commented-out
  `isin([1, 2])`). Files were selected by position in a directory listing.
- Stigma: now "in the last 6 months" (item value 1), as intended. Primary-state median AHS falls
  from ~24-29% to ~10-16%, family stigma from ~41% to ~20%, general social stigma from ~52-61% to
  ~19-24%. A composite is missing only when no component has a substantive answer (prefer-not
  and don't-know otherwise count as no).
- General social stigma's violence components were `d1 + d2 == 2`, which also counted "violence
  more than 6 months ago, not attributed to MSM" (2 + 0). Now: violence in the last 6 months and
  attributed to having sex with men in the last 6 months (d1 == 1 & d2 == 1; d3 with d4).
- Race: AI/AN, other/multiple and MENA respondents were dropped from the whole analysis (5-7.5% of
  each cycle). They now form the "other" category with Asian/NHPI, matching the state raking target
  (which covers all other races). Unknown race (AMIS 2021+ code 8) is missing. National weights
  still exclude "other" (no national target).
- Education: AMIS 2018 has no calculated education for ages 18-24; the old recode used the youth
  item with codes shifted one level (HS graduates as less than HS, etc.). Now mapped as observed in
  AMIS 2019, where both items exist (2-3 lths, 4 hs, 5 college, 6-7 grad).
- Risk behaviour: prefer-not/don't-know codes (7/9) were summed as "yes" (666 of 9,731 risk = 1).
  Now 1 only for a yes; missing only when both items are prefer-not/don't-know.
- Skip and non-response rules (applied to every measure): a legitimate skip takes the value
  implied by the gating answer; prefer-not-to-answer, don't-know and design skips (split-sample
  modules) are missing; stigma composites use checklist scoring (1 if any item endorsed, 0 if
  none endorsed and at least one answered, missing only if none answered). Consequences:
  - Disclosure to provider: "told nobody" (out_gi = 0, skips the "who" item) and "does not
    apply" (no provider) are 0. Levels fall 6-10 pp (e.g. AMIS 2023 83.6% -> 76.6%); missing
    falls to ~1%, except AMIS 2017-2018 (~36%), where about a third of respondents who had told
    someone were not asked whom (split-sample module).
  - Risk behaviour: respondents with one male partner are routed to partner-specific items and
    were never asked m_muauhs/m_muahp; they now count as at risk if condomless and the partner's
    status was unknown (m_mm1hsk = 0) or positive/indeterminate (m_m1hst). Levels rise 2.5-5.6
    pp. Respondents reporting condomless anal sex but routed to neither path (no partner count;
    ~15% of AMIS 2017, ~6% of AMIS 2018) are missing rather than 0.
  - PrEP use: 0 for PrEP-unaware respondents and those never on PrEP (skip past the item); prefer
    not/don't know now missing (was 0).
  - HIV testing: also 0 when the last test was more than 24 months ago and the count is missing.
  - Healthcare visit: missing for ~12% of AMIS 2017-2018 (not asked); prefer not/don't know missing.
- Respondents with missing HIV status (~135 per cycle) are excluded (as the R stage always did).
- AMIS 2015-2016 are excluded because their stigma items do not ask whether the experience
  occurred in the last 6 months (confirmed 2026-10-22; state in methods).
- COVID: AMIS 2020 (fielded October 2020 - January 2021) is not loaded; no other cycle had
  respondents in the former 2020-03-01 to 2021-07-01 window.
- Urbanicity is absent from AMIS 2023 and is dropped from the group-regression covariates
  (complete-case analysis was silently excluding all AMIS 2023 respondents).
- (Superseded 2026-10-22 by cycle-based periods; see "AMIS 2024 and cycle-based periods".) Periods
  were calendar years of submission: 2023-2024 = second half of AMIS 2022 + AMIS 2023, before AMIS
  2024 was added. AMIS 2022 asked the stigma items of about half its respondents.
- AHS wording changed in AMIS 2022: "worry someone may learn you have sex with men" (2017-2021)
  became "worry someone may treat you poorly because you have sex with men" (2022-2023). Other
  stigma items are unchanged. This coincides with the AHS rise into 2024 (primary-state median
  10.3% in 2022 -> 15.7% in 2024). Decided: report in methods and limitations, and add a sensitivity analysis that removes the
  wording shift (e.g. 2023-2024 AHS matched to the AMIS 2022 level, or fitting through 2022).
- Precision screen after the rebuild: 17 primary states (MA and TN qualify again), median-60
  screen 17, strict screen 14. Downstream references, drift target and all fits must be
  regenerated.

## Demographic raking margins rebuilt (2026-10-21)

State and national demographic margins (age, race/ethnicity, education; national also Census
region) now both come from the MSM-Population-Estimates repository (branch
`amis-compatible-education`, `raking_targets/build_amis_raking_targets.R`), replacing
- state margins from a separate OneDrive build (`GSS and NCHS Data for Marginal Dist/Results/
  state_level_estimates_final_adj.csv`; producing code not found; US total ~1.0M MSM vs 5.6M in
  the repository), and
- national margins hand-assembled in `national_raking_targets.csv`.
`inputs/state_msm_margins.csv` and `inputs/national_msm_margins.csv` are copies of its outputs.

- Education categories did not match AMIS: targets' "college" included bachelor's degrees and
  "grad" meant graduate degrees only, while AMIS's top category is bachelor's or higher; so
  45-66% of AMIS respondents were raked to a 13-19% target (national weights up to 23x the mean).
  Both target sources also used ACS education for men 25+ only, and the repository used
  both-sex ACS columns. Now: men 18+ (ACS S1501 male columns, 18-24 and 25+ rows), categories
  less than HS / HS / some college or associate's / bachelor's or higher, with GSS MSM rates in
  the same categories (degree, years of schooling separating some college from HS). The
  repository's cleaning Rmd carries the same fix.
- Method: the repository's county formula (MSM_total x male[d] x GSS_rate[d, urbanicity],
  normalized within county) evaluated at point estimates with the 100,000-draw county totals;
  it reproduces the simulated state age and race shares (median differences 0.3 and 0.8 pp).
  The repository's saved ethnicity summary predates its current parameters and is stale.
- Race/ethnicity: Hispanic MSM from the ethnicity model; the non-Hispanic remainder split across
  White, Black and other by race-group MSM after removing Hispanic men (ACS B01001A/H/B/I, 2023
  5-year). Hispanic shares are higher than before (national 24% vs 19%), driven by GSS MSM rates
  by ethnicity (small samples); note the uncertainty.
- Primary-state medians: bachelor's or higher 39.6% (was 19.1% "grad"); some college 34.1%; HS
  19.7%; age 18-24 12.7% (was 16.3%).

## AMIS 2024 and cycle-based periods (2026-10-22)

- AMIS 2024 (`sitq2024_analysis_final_NOID.dta`; fielded November 2024 - August 2025) added: 2,747
  analytic respondents. Value codes identical to AMIS 2021-2023; AHS wording as in 2022-2023 (no third
  version); stigma module asked of about half of respondents; the sexual-violence items (stigma_d3,
  d4) are absent, so its general social stigma composite has four components (note in methods).
- Periods now follow AMIS cycles rather than calendar year of submission (decided): each cycle keeps
  its year (`survey_year` = cycle year; `submission_year` kept), and periods pair cycles - 2017-2018
  (AMIS 2017, 2018), 2019-2020 (AMIS 2019), 2021-2022 (AMIS 2021, 2022), 2023-2024 (AMIS 2023, 2024).
  Only two groups moved: AMIS 2022's 2023 submitters (to 2021-2022) and AMIS 2024's 2025 submitters
  (into 2023-2024). Rationale: whole cycles share one instrument version, coding and module design;
  AMIS 2022 (October 2022 - October 2023) is no longer split; the AHS wording change falls at a cycle
  boundary. The 2024 period now includes data collected through August 2025; annual demographic
  series (scripts 05, 06) use cycle year.
- Within AMIS 2024, early (2024) submitters looked more engaged than later (2025) ones (unweighted AHS
  8.3% vs 13.6%, disclosure 83% vs 68%); taking whole cycles avoids depending on fielding timing.
  National 2025-submitter trends (descriptive, demographic raking only): AHS remained elevated (14.3%)
  without rising further, consistent with a level shift at the 2022 wording change.
- National weights (scripts/02) checked against the cycle periods (2026-10-22): raking runs per
  cycle-pair period; PrEP targets pool CDC/AIDSVu calendar years (2023-2024 uses 2023 only); all
  margins met. AMIS respondents are much older from AMIS 2021 on (median age 26 in 2019, 42 in 2021,
  51 in 2024; categories verified against recorded age), so ages <=24 (12% of the target) rest on
  957 respondents in 2021-2022 and 286 in 2023-2024 (n_eff 666 and 155; 2,520 in 2017-2018). Note in
  methods for the descriptive age-group trends (annual 2023: 221, 2024: 125 respondents aged <=24).
- Precision screen: 24 primary states (adds SC; UT at 60.1 and MN 25th percentile 41.1 borderline);
  median-55 screen 25 (adds AZ); strict 22. The 2023-2024 period remains the least precise (median
  primary-state cell n_eff 76 vs 157 in 2021-2022).

## CDC stocks rebuilt from the raw downloads (2026-10-21)

`scripts/00_build_cdc_stocks.py` builds the state-year MSM stocks from the AtlasPlus download and
the seven AIDSVu state PrEP files (`inputs/cdc_raw/`, public) and the state MSM population
estimates. It replaces `EEE CDC Clean.ipynb`, `AIDSVU PrEP Cleaning/PrEP Cleaning.ipynb` and
`EEE Stocks Final 2026 - Stocks 7.ipynb`, and reproduces `EEE Stocks 7.csv` exactly (5,782 rows;
every count, imputation flag and the row order). Two unused notebook inputs (`amis_svy_df.csv`,
`CDC Data.parquet`) are dropped; the MSM population workbook it read is identical to
`inputs/msm_state_population.csv`.

PrEP-eligible denominator scale (x 2): CDC-implied men with PrEP indications (persons prescribed /
coverage) are doubled, so state PrEP targets are about half of CDC-published coverage (primary-
state median 19.5% in 2022 vs 39% published; users / HIV-negative MSM would give 8.3%). The user's
rationale: published coverage overstates PrEP use relative to incidence and diagnoses. The only
documented scale analysis (`AMIS-Variable-Trends/raking_robustness_checks.Rmd`) chose among scales
1-3 by raking fit to the PrEP target and weight variability under the former two-pass,
drop-trimmed raking, not by incidence or diagnoses; target fit improves monotonically with the
scale. Open: document the incidence/diagnosis-based justification (or run it as an explicit
calibration and sensitivity analysis). The national age x PrEP targets
(`inputs/national_raking_targets.csv`, built outside this repository) use the same scale.

## Pipeline fixes

- Fixed-dose sensitivity (scripts/12, 2026-10-22): the script monkeypatched `StateIntervention.apply`
  and inserted codebook entries at runtime. The effect (`probability_absolute_to_baseline`) is now in
  `models/shared/intervention.py` (verified identical to the old patch), the scenarios in
  `config/interventions.py` (`register_fixed_dose_scenarios`), and scripts/12 runs them through `pipeline/sem_scenarios.py`, so they now use
  the same draw pairs as the primary scenarios (previously different per-state seeds). The comparison
  now covers cumulative 2025-2036 incidence as well as 2036 incidence.
- Fixed doses (2026-10-22, decided: matched). 10 and 5 pp were chosen for the "ever" AHS measure
  (~25-30%); under the 6-month definition AHS is ~10-16%, so a 10 pp cut removes most AHS in many
  states. Now derived from the main scenario run in scripts/12: per measure, the matched dose is the
  median percentage-point reduction under s7 (50% of every stigma level) in 2032, when the ramp
  reaches full effect, across the primary states and draws; the half dose is half of it. AHS-only
  scenarios use the AHS dose. Written to `results/05_sensitivity/fixed_dose/doses.csv`. Scenarios
  are registered at run time (`register_fixed_dose_scenarios`), so scripts/11 `--all-scenarios` no
  longer needs to exclude them. Wiring check on the pre-rerun results (40 draws): matched AHS 14.4,
  general social 25.4, family 20.2 pp; values to be reported from the rerun.

- Coupling strength (2026-10-22): the calculation moved from scripts/13 into
  `models/sbm/structural.py` (`coupling_strength`), with unit tests (`tests/test_structural.py`):
  total equals the sum of matrix powers; channels sum to the total; no indirect exits -> all direct;
  no direct edge -> direct channel 0; exits outside the reported channels raise. Script 13 outputs
  identical to before.
- Scenario layer reviewed: baseline and scenario runs are paired (same seed and draw pairs per
  state); the AHS reduction ramps 0% (2026), 33% (2028), 67% (2030), 100% from 2032, relative to each
  draw's own baseline AHS. State the ramp in S4. Removed the unused deterministic runner (with NY
  debug prints), unused result classes, an always-empty USA summary, and stale "standardized v8"
  defaults; scenario outputs identical before and after.
- Decided (2026-10-22): refit the epidemic model. The v7d posterior was calibrated with the earlier
  AMIS PrEP and risk series; risk behaviour changed with the rebuilt cleaning (single-partner fix),
  and incidence scales with (risk / risk0)^alpha. The refit is a pipeline step, so the frozen
  input is no longer used, and Fig S2 is regenerated.
- Epi refit implemented (2026-10-22): `scripts/04b_fit_epi_model.py` -> `pipeline/hierarchical_epi_fit.py`;
  outputs in `results/02_inputs/epi_fit/` (posterior.nc, period_targets.csv, period_fit_summary.csv,
  epi_bundle.npz). Model specification unchanged. Inputs are built in the repo: AMIS PrEP, risk and
  testing from the workbook trends (state-periods without an estimate take the division value via
  the scripts/04 input draws, median); CDC MSM stocks from `results/00_cdc/cdc_stocks.csv`;
  divisions from `config/data.py`; the 2020 Census state populations that scale beta's state
  deviations are kept as documented constants. The fit window is derived from the data (2017 to the
  last year with diagnoses, estimated prevalence and viral suppression for every state; now
  2017-2022), and diagnosis periods (2017-2019, 2021-2022, 2023-2024) enter whenever they lie
  inside it. The bundle carries cdc_beta/alpha/kdx/U0/post_multiplier draws (flattened chain by
  chain), cdc_risk0 = the fit's risk baseline (max(2017 risk, 0.05)), cdc_kappa_prep = 1, and the
  CDC stocks over the fit window (held at the last year to 2036). The research bundle now takes all
  SEM fields from the state fit, so the base bundle carries only epi/CDC fields. Scripts 09 and 10
  default to `results/02_inputs/epi_fit/epi_bundle.npz`; `inputs/epi_v7d/` is kept for regression
  checks and the old Fig S2 diagnostics until script 16 is updated.
  Checks: (1) the converter applied to the original v7d posterior (legacy
  `model_period_v7d_all_states_full.pkl`) reproduces the frozen v9 parameter arrays exactly, and the
  CDC stocks to 5e-10; (2) the rebuilt likelihood data equal the original fit's (all 183 annual
  incidence observations and both diagnosis-period targets), so with the original AMIS inputs the
  fit would be the same; (3) scenario outputs with the converted arrays equal those with v9 (max
  difference 4e-9); (4) a short 5-state sampling run completes. The only input change is AMIS:
  e.g. 2017 risk (risk0) CA 0.258 (was 0.208), TX 0.216, NY 0.265. In v9, CDC incidence for states
  without estimates (DE, HI, IA, KS, NE, NM, OR, RI, UT, WV in some years) had been filled with
  carried values; the bundle now leaves them missing (not used by the fit or the simulator).
- Core model review (2026-10-22): loss terms and gradients (covariance 4*E*J*Sigma, trajectory with
  drift, shrinkage, structural/diagonal/drift priors, spectral-radius penalty with analytic gradient
  for a real dominant eigenvalue and finite differences otherwise), bounds, initialization,
  transforms and the predictor checked; fit and projection use the same centered logits and update
  x[t+1] = J x[t] + c. New test: loss gradient against finite differences (tests/test_loss.py).
  Removed unused code: loss normalization/scaling options (each term was divided by 1 + 1e-8),
  shrinkage boosts for states with n < 350 (fitted states have n >= 565, so the weight is
  5 x 300 / n), the workbook CDC sheet and all CDC handling in `data/` (the epi fit and simulator read
  CDC elsewhere), and unused config fields. The out_gid -> AHS prior sign (+) is intended (confirmed).
- Scenario draws (2026-10-22, decided): each state's random stream depends only on the seed and the
  state code (was seed + position in the state list, so a state's draws changed with the state set,
  e.g. across precision screens), and transition-model and epi draws are taken without replacement
  when there are enough (500 of 500; 500 of 2,000). Verified: CA's draws are identical whether run
  alone or with other states.
- Local optima (2026-10-22, found): the loss is unchanged by the cleanup (to 1e-8), but refitting
  three draws gave a different optimum for 1 of 102 unit fits in the primary model (MA draw 0: loss
  26.77 vs 28.54) and 3 of 102 under no_hierarchy (up to 7%), from rounding-level differences.
  Restarting the optimizer from its solution does not improve the loss, so these are separate local
  minima, not early stops. Fits are deterministic run to run. All of it occurs in fits on the
  stability boundary (spectral radius > 0.998; 65% of fits); fits off the boundary are unaffected.
- Multi-start (2026-10-22, adopted): `JacobianEstimator` fits each unit from three deterministic
  starts (default, structural/diagonal prior means, parent unit's J) and keeps the lowest loss;
  `best_start` and `start_loss_spread` are recorded in the loss-components file. 100-draw check
  (primary): multi-start lowers the loss in 6.4% of state fits (median 0.26%), is higher in 1.1%;
  the starts disagree by > 1% in 6.8%. Effect on reported summaries is small either way: state
  median AHS -> PrEP coupling (range -1.64 to -0.14) moves by at most 0.05 single vs multi (rank rho
  0.997); sensitivity to a 1e-8 loss perturbation halves (max 0.043 -> 0.022; rank rho 0.990 -> 1.000).
  Cost: about 3.3x fitting time (500 draws ~30 min per specification). A smooth stability penalty
  (all eigenvalues above 0.999) was also tried; it did not remove the multiple minima and was not
  adopted.
- Period timing (2026-10-22, decided): each AMIS period stays placed at its end year (2018, 2020, 2022,
  2024) in the transition model, the epi fit and the projections. Mean submission year is 2017.5,
  2019.0, 2021.7 and 2024.1, so the first two periods lead their placement by 0.5-1 year (2019-2020 is
  all AMIS 2019). Kept for one consistent clock between fit and projection; state in methods.
- CDC update (2026-10-22, done): AtlasPlus re-downloaded 2026-10-22 as table exports in
  `inputs/cdc_raw/atlasplus_2026-10/` (MSM and MSM/IDU: diagnoses, prevalence, AIDS
  classifications, viral suppression, care, linkage; "All" both sexes: diagnoses, prevalence,
  care). The first "All" exports were male-only and were replaced, since the "All" series are
  denominators for allocating CDC's all-population estimates to MSM. `00_build_cdc_stocks.py` takes
  every indicator x category in the update from it (2017-2024; 2025-2026 are marked preliminary and
  excluded) and everything else from `atlasplus_hiv.csv` (estimated incidence, estimated prevalence
  and knowledge of status, 2017-2022, the latest CDC estimates; PrEP coverage). After 2022,
  estimated MSM prevalence = diagnosed MSM prevalence x the state's 2022 ratio of estimated to
  diagnosed MSM prevalence; estimated incidence is not extended. Parsing fixes: footnote markers on
  state names ("Mississippi^") and "Data suppressed" cells; missing MSM counts now give missing
  (not zero) allocated values.
  Changes vs the earlier download (2017-2022, model inputs): median revisions <0.5%; PrEP stocks and
  age targets identical. Larger changes: 2022 MSM diagnoses were incomplete in the earlier download
  (reporting delay; e.g. NM 71 -> 113, MS 321 -> 379, MT 9 -> 12); TX MSM prevalence +5-8% (CDC
  re-attribution of transmission category). New Hampshire MSM counts are suppressed in every year of
  the new release, so NH is excluded from the epi fit (not a primary state).
  Epi fit: window 2017-2024 for 48 states (fit window and state exclusion are derived from the
  data); diagnosis periods 2017-2019, 2021-2022, 2023-2024; incidence likelihood unchanged (183
  annual observations, 2017-2022, 2020 excluded). Bundle: CDC stocks 2017-2024, PrEP stocks
  (through 2023) carried forward, all held flat after 2024.
- CDC stocks in projections (2026-10-22, decided): PrEP-eligible and not-virally-suppressed stocks
  (estimated prevalence - viral suppression) are now held at the last observed CDC year
  (`models/shared/alignment.py`, `build_cdc_inputs_from_sem`). Previously the simulator read
  `cdc_raw_native` and linearly extrapolated the final one-year change to 2036 (CA not virally
  suppressed: 53.6k in 2022 -> 59.7k in 2036; PrEP eligible 290.8k -> 296.4k). Changes baseline
  incidence levels; relative scenario effects expected to move little. AMIS trajectories are
  unaffected (interpolated between periods; projected by the transition model).

- Estimator (2026-10-22): `JacobianEstimator.fit()` ran three optimizations per unit (periods
  1-2, 1-3, 1-4) and only the last was used; with re-initialization each fit started from the
  same values, so the first two could not affect it. It now fits once on all periods: identical
  results (0.0 difference in J, drift, success and loss on two draws), about 6.5x faster (500
  draws: ~8 min). Unused options (warm starts, J_init, x whitening, baseline component scaling)
  and `Predictor.predict_rolling` removed.
- Hierarchy: state shrinkage weight = 5 x 300 / n (n = state respondents across periods): 0.36 (CA)
  to 2.76 (AL); the small-state boosts (n < 350) do not bind for any primary state. Divisions are
  shrunk to the nation with weight 10. State in S2.5.
- Sigma_eta (covariance-target innovation variance) = 0.16^2 x diag(SigmaX); now listed in Table S4.

- Code structure (2026-10-22): model logic moved from scripts into the package. The hierarchical
  fit over input draws and the model specifications are in `pipeline/state_fit.py` (replacing
  the archived `models/sbm/runner.py`); the research bundle is built by
  `input/research_bundle.py`. Scripts 09 and 10 now only parse arguments and call them. Verified
  on two draws: identical fits (every J, drift, success flag and loss component) for the primary
  and sign-only specifications, and identical scenario projections. Specifications reduced to the
  paper's ten (primary + S7: trajectory only, covariance weight 0.10 and 0.40, sign only, common
  weak prior, no hierarchy, half and zero PrEP drift, stability weight 10); the unused variants and
  the response-profile classification were removed. The supplement label "covariance weight 0.20"
  was wrong (code: 0.10). Default draws 500 (was 100).

- Covariance target was hard-coded (2026-10-22). `DataPrep` took the covariance target from a fixed
  8x8 matrix typed into `data/loader.py` (commit dd6439d, April 2026; built from the old "ever"
  stigma definitions, binary testing and earlier weights); the workbook `cov` sheet from scripts/01
  was never read. All fits to date used it (e.g. AHS variance 0.217 vs 0.104 in the current data;
  PrEP-testing correlation 0.32 vs 0.48). `DataLoader.load_covariance()` now reads the `cov` sheet
  and checks its variable order; the hard-coded covariance and sign-mask fallback are removed (the
  sign mask always comes from `config/sem_structural_priors_v3.csv`).

- State raking did not converge (2026-10-21). `rake()` was called with `control = list(partial = TRUE)`;
  `partial` is not a `rake` option, and the default tolerance (`epsilon = 1`, absolute, on margins
  that sum to 1) stopped every state-period after the second pass. PrEP (the last margin) matched
  exactly; age, race and education did not (primary states: age margin median error 1.0 pp, max
  9.8 pp). Now `maxit = 100, epsilon = 1e-7` (as in the national raking), with demographic targets
  renormalized over the categories present in each state-period. All primary-state periods
  converge; 10 small non-primary state-periods (n 31-85) do not and are flagged in
  `state_raking_diagnostics.csv`. Matching the margins lowers effective sample sizes (median
  state-period Kish n_eff 117 -> 108); under the unchanged precision screen MA and TN no longer
  qualify (expected 15 primary states; strict screen 11). All downstream results are regenerated.
- The workbook's derived CDC `prep_used` and `hivtest12` rows (the latter was diagnoses / incidence,
  mislabelled) were never read by the model and are removed. The PrEP raking plot is removed; the
  diagnostics CSV reports weighted PrEP against target for every state-period.
- National age x PrEP targets are now built in `scripts/00_build_cdc_stocks.py` for all periods
  (previously hand-entered "user-provided" male figures for 2017-2022, no 2023-2024 targets, and
  a 35+ target that omitted ages 55+). target_age = national men's target (PrEP / PrEP Eligible
  from the state stocks) x coverage_age / coverage_all, where coverage = CDC persons prescribed /
  persons with indications by age (AtlasPlus, both sexes, ages 16-54; `inputs/cdc_raw/
  atlasplus_prep_by_age.csv`) and age groups are combined by AIDSVu users. Assumption: ages 55+
  have the coverage of ages 45-54. 2023 uses AIDSVu 2023 users over the 2017-2022 mean
  indications. Versus the former targets (2017-2022): <=24 within 0.1-1.6 pp, 25-34 within 1.3
  pp, 35+ 1-2.7 pp lower (55+ now included). The 2023-2024 overall-share stopgap is removed.
- CDC's AtlasPlus PrEP footnote: CDC has paused PrEP coverage reporting and advises against citing
  specific coverage data points. Describe targets as derived from CDC estimates of persons with
  PrEP indications, without quoting CDC coverage values.
- National weights exclude respondents of other/multiple race (3.7%, n = 1,477): the national
  race target has only Black, White and Hispanic/Latino. Report in S1.4.
- HIV testing measure: never-testers (`evertest == 0`) skip the test-count item, so `test2yrs` was
  missing for them and they were dropped from the testing measure (6,080 of 6,872 missing values).
  They are now coded as zero tests. Primary-state median testing falls from about 57% to 46% (never
  tested: 16-18% weighted); the cross-state ordering of 2018-2024 changes is similar (rho 0.91).
  The measure is tests in the past two years (capped at 4, scaled 0-1), not past-12-month testing;
  the code keeps the name `hivtest12`.
- Respondents aged 15-17 (n = 1,493) are excluded: the youngest raking targets are adult 18-24
  shares, so including them would rake 15-24 respondents to an 18-24 population share. The age
  rule is explicit: 18-99 (previously implicit through the missing age category). Including them
  changed no primary-screen decision.
- Capped state weights are rescaled to sum to 1 within each state-period (capping had reduced sums to
  0.83-1.00, under-weighting heavily capped cells in the pooled covariance).
- Inputs live in the repository and scripts use relative paths (run from the repository root):
  `inputs/` holds the CDC stocks, state MSM margins, state MSM population and national raking
  targets (tracked); `inputs/restricted/combined_amis_nocovid.csv` holds the AMIS respondent data
  (gitignored). `config/paths.*` and `scripts/R/paths.R` are removed. Copies are frozen at the
  versions used for the paper.
- Respondent definition and the testing recode live in `scripts/R/amis.R`, shared by 01 and 02;
  03, 05 and 06 read the recoded measure from the saved weights.
- After these changes: primary screen unchanged (15 states; TN median n_eff 53, 25th percentile 35);
  median-60 screen 13 states; strict screen 13 (adds IN, MD vs the earlier 11), because testing
  cells now include never-testers.
- Demographic analyses kept only HIV-negative respondents (`_hivstat == 1`, n = 30,295) although the
  national weights were raked on all non-positive respondents (`_hivstat != 2`, n = 36,541; codes
  3-5 = unknown/untested, 17%). They now use all non-positive respondents, matching the weighting
  and the state pipeline.
- R stage rewritten as plain scripts (no pandoc/knitr): `01_state_weights.R`, `03_state_bootstrap.R`
  (bit-identical draws to the former Rmd), `05_group_associations.R` (regressions only; per-group
  dynamic-model inputs removed with the subgroup models), `06_group_bootstrap.R` (six national
  groups; CA Hispanic/Latino bridge removed). Raking helpers inlined into `01`.
- Division inputs average their states weighted by respondent count (states-sheet n); the
  national input uses MSM population. Consistent in scripts 03, 04 and the model's unit builder;
  kept (divisions serve as shrinkage targets). State in S1.
- Bootstrap replicates hold raking weights fixed (no re-raking), so intervals omit weighting
  uncertainty; state as a limitation.
- The pooled covariance sheet deliberately uses all states (every state contributes valid respondent
  data), weighting states by sample size.

- The state bootstrap Rmd cached its resampling chunk (`cache=TRUE`), so reruns silently reused old replicates when the weights changed. Caching is removed.
- State raking wrote weights back by `rowID` matching; replaced with positional indexing (no collisions occurred in the data).

## Model specification and projection

| Decision | Previous | Final | Supplement |
|---|---|---|---|
| Stability penalty weight | 10 (not binding: median spectral radius 1.027, 12-13 of 17 states > 1) | 10,000 (soft penalty: state-median spectral radius 0.999 in every state; 7% of state draws slightly above 1, maximum 1.016; trajectory-fit loss +1%) | S2.5.3 |
| Healthcare-stigma persistence target | None | None, by design: anticipated healthcare stigma is the proximal, healthcare-facing stigma and the most susceptible to change, whereas general social and family stigma are upstream and slowly changing (persistence 0.85). Tested 0.85 for AHS: barely changed TN, shifted every other state, worsened fit 3%. | S2.4 (state the rationale) |
| Population-model convergence | - | 3,498 of 3,500 refits converged; the two non-converged draws (Hispanic/Latino draw 327, ages 25-34 draw 434) are excluded, leaving 499 refits for those groups | S6.4 |
| Projection starting point | Fitted system propagated from 2018 point estimates | Observed values held through 2024; each draw projects from its own resampled 2024 values | S4.1, S4.3, Table S4 |
| Starting-point sensitivity | - | Projections from the 2018 reconstruction (`--projection-start reconstructed`) | S7 (new) |

TN healthcare stigma rises toward ~90% by 2036 in the baseline. Its 2024 stigma values jump (family stigma +21 pp) in its lowest-precision period (n_eff ~31). Across the 17 states, 4.6% of period-to-period stigma changes exceed two standard errors (about the 5% expected from sampling noise alone). No state-specific adjustment was made; TN is excluded in the stricter-screen sensitivity.

## Scenario definitions

`config/interventions.py` is current; Table S4 needs updating. Primary scenarios
are reducing anticipated healthcare stigma, reducing upstream social stigma
(general social and family stigma), and reducing all three stigma levels. The
all-levels scenario contains no pathway attenuation (Table S4 currently adds a
50% AHS -> PrEP attenuation). Pathway scenarios run as additional results.

## Outcomes and framing (2026-10-20)

Decided: state structural sensitivity (coupling strength and route, below) is the primary
result and replaces the response profiles. Incidence outcomes are secondary translations of
that structure; cumulative 2025-2036 and 2036 single-year changes are both reported, neither
as the headline.


- Single-year 2036 relative incidence change mixes structure with PrEP saturation
  and drift; its cross-state ranking is unstable across S7 specifications (rho 0.16-0.88).
  Absolute infections averted rank robustly (rho 0.82-0.98) but mostly reflect incidence level.
- Cumulative 2025-2036 incidence reduction (AHS scenario median -2.5%, range -0.6% to
  -4.3%) is consistent with the Jenness/Onwubiko model estimates (JS31).
- Proposed lead result: structural sensitivity of PrEP to anticipated healthcare stigma,
  i.e. the cumulative impulse response (sum of J^k[PrEP, AHS], k = 1..6), decomposed exactly
  by the first channel through which the effect leaves AHS (direct to PrEP, or via
  disclosure, healthcare contact, HIV testing); `scripts/13_structural_sensitivity.py`.
  Total-sensitivity ranking is stable across specifications (rho 0.75-0.98; 0.65 under zero
  PrEP drift, 0.51 under covariance weight 0.40).
- Findings: coupling strength varies about fivefold (strong: CA, IN, TX, MD; weak: GA, FL,
  PA, NY), but the route varies little: the direct AHS -> PrEP channel carries 57-94% of the
  effect in 16 of 17 states. Indirect channels are notable in NC (43%; disclosure and
  testing), NJ (41%; testing), MA (36%; disclosure); FL is 73% indirect but very weak.
  An earlier approximate split (direct = 6 x J[PrEP, AHS]) misassigned carry-over of the
  direct effect to "indirect" and overstated engagement mediation (IN, TN); superseded.
- Discrete route profiles are threshold- and specification-sensitive; present the
  continuous structural map instead of response profiles.
- GSS/family -> PrEP direct pathways not added: national cross-sectional SEM found positive
  associations (probit +0.06 to +0.09), consistent with Onwubiko (aPR 1.14) and NHBS (JS59).

## Population analyses (S6)

- Verified: bootstrap (500), national age-PrEP weights for the six national groups (annual
  waves with within-replicate ESS-weighted smoothing), California state weights for CA
  Hispanic/Latino (two-year periods), new references, drift 0.433, diagonal persistence
  targets, stability 10,000. 3,498 of 3,500 refits converged.
- The draft's population numbers do not come from this method: national groups came from
  the old Beta-sampling Monte Carlo with zero drift (`priority_population_sem_mc_annual_smoothed_n500`),
  and CA Hispanic/Latino -0.497 from a zero-drift Monte Carlo on national race x PrEP weights.
  Those weights put CA Hispanic/Latino PrEP at 2-5% (2018-2024); state weights give 13-26%,
  national age-PrEP weights 19-43%. The CA contrast was a weighting artifact.
- Current results: AHS -> PrEP coupling is clearest for ages <=24 and White non-Hispanic MSM,
  weak and often zero for Hispanic/Latino (38% of draws) and Black non-Hispanic (61%); CA
  Hispanic/Latino coupling is weaker than the CA aggregate (structural -0.46 vs -1.60).
  Zero-drift population sensitivity (`results/05_sensitivity/population_zero_drift`): group
  orderings do not survive (rank rho -0.03 for structural sensitivity, -0.26 for AHS -> PrEP).
- Decided: remove the demographic-group dynamic models (S6, CA Hispanic/Latino bridge) from
  the paper. Add one Limitations sentence stating that they were explored but annual subgroup
  data were too imprecise to identify group-specific pathway structure robustly.
- Decided: report demographic groups descriptively instead: weighted 2017-2024 stigma and
  PrEP trajectories with bootstrap intervals (Black MSM were the only group with significant
  increases in all three stigma domains: AHS +9.6, family +14.2, general social +12.2 pp,
  while PrEP rose as in other groups), and survey-weighted cross-sectional associations
  (AHS associated with lower provider disclosure in every group, OR 0.43-0.59, p <= 0.001;
  AHS-PrEP associations inconsistent and positive where significant). Endpoints come from
  the smoothed annual series; state this.

## Route robustness (states)

- Within the primary fit, direct and indirect contributions are not traded off across draws
  (median correlation +0.15; indirect is smaller, not larger, when the direct path is at zero).
- Across specifications, the direct AHS -> PrEP route depends on the PrEP drift assumption
  (state-median direct -0.50 primary, -0.08 half drift, 0.00 zero drift; rank rho 0.17 under
  zero drift), while engagement-mediated contributions keep their ordering (rank rho
  0.80-0.96) but shrink to about zero under a common weak prior (state median -0.25 -> -0.02),
  because the AHS -> disclosure pathway carries a strong-tier prior. Each route's size
  therefore rests on a different assumption (direct: PrEP drift; engagement: pathway priors).
- Decided: the user regards the secular PrEP drift and the theory-informed pathway priors as
  credible. Report the primary-specification route decomposition as the result (direct
  barrier carries most coupling in most states; engagement routes about a third in a typical
  state, mainly disclosure and testing; larger in NC, NJ, MA), justify both assumptions, and
  report the drift and prior sensitivity analyses as showing how the route balance shifts.
- Lead quantity: total coupling strength (finite-horizon structural response), defined as
  S = sum_{h=1..6} e_PrEP' J^h e_AHS: the cumulative centered-logit PrEP response to a
  one-time unit centered-logit AHS impulse in 2024, through all routes, with drift cancelling.

## Full rerun (2026-10-22)

Run 16:36-20:45 (stage 1 data/weights/bootstrap/screen/constants; epi refit 2 h 46 min; 10 state-model
specifications with multi-start in parallel; scenarios, fixed dose, specification scenarios,
coupling). Previous results archived in `../EEE-SD-Model-2026-archive/results_pre_rerun_20261022/`.
- 24 primary states; constants: PrEP drift 0.420, persistence 0.85, AHS reference 0.061.
- All 5,000 state-model draws converged (primary + 9 specifications). Multi-start: the three starts
  differ by > 1% of the loss in 41% of primary state fits (> 14% in a quarter); FL and GA most often.
- Epi refit (48 states, 2017-2024): 0 divergences, max R-hat 1.012, min bulk ESS 586; alpha 0.053
  [0.030, 0.085] (prior 0.060 [0.034, 0.100]); post-2021 diagnosis multiplier 1.06 [0.99, 1.13]
  (prior 1.40); 2023-2024 diagnoses median error -0.8%, 69% of states within the 90% interval
  (under-fits IN, NV, IL, NE by 16-19%).
- Headline (median across states, 2036): AHS reduction lowers incidence 4.9% (range 0.2-10.1; was
  10.3% in the 17-state run) and cumulative 2025-2036 incidence 1.0% (was 2.5%); all stigma levels
  5.0% / 1.0%; upstream social stigma 1.1% / 0.15%. State rank agreement with the previous run on
  the 17 common states: rho 0.3-0.5.
- AHS -> PrEP coupling: state median -0.58 (was -0.90), CA strongest (-1.20), TN and VA near zero;
  engagement ~26% of the median. Robust to covariance weight up to 0.10, hierarchy and stability
  weight (rank rho 0.84-0.98); weaker under sign_only and common_weak priors (median -0.31, -0.17);
  the direct AHS -> PrEP edge goes to 0 when the PrEP drift prior is halved or zeroed.
- Fixed doses (matched): AHS 7.7, general social 9.7, family 9.2 pp. Rank agreement with the
  relative scenarios is low for 2036 incidence (rho 0.27-0.44) and moderate for cumulative incidence
  (0.54-0.69).

## AHS history counterfactual (2026-10-22, added)

`scripts/13b_ahs_history_counterfactual.py` (core: `models/sbm/structural.py`,
`source_history_counterfactual`; tests in tests/test_structural.py). For each state and draw, the
fitted system runs from the draw's observed 2018 values with AHS on its observed path vs held at its
2018 level; other measures evolve through the fitted pathways, drift cancels. Continued two steps
(2026, 2028) with AHS at its 2024 vs 2018 level to show the effect already set in motion (PrEP
responds one step after AHS). Frames the question as how AHS has shaped PrEP, rather than how PrEP
would respond to a future change.
- Median AHS path across primary states: -2.7 pp (2018-2020), +1.6 (2020-2022), +4.0 (2022-2024);
  runs reproduce observed 2024 PrEP closely.
- Effect on PrEP is small next to the observed PrEP rise (median +13 pp, mostly secular): 2024
  median 0.0 pp (range -0.4 to +1.6); by 2028, rising AHS lowers PrEP in WI (-1.6), CA (-1.4),
  MI (-1.2), IN (-0.8), TX (-0.5), while early AHS declines raise it in CO, FL, MA, MO, NC
  (+0.5 to +0.9); IQRs include 0 in most states. Rank rho with the AHS change -0.50, with coupling
  strength 0.45.
- The 2022-2024 AHS rise spans the AMIS 2022 wording change; the 2026-2028 effects depend on it, so
  the AHS wording sensitivity analysis applies here as well.
- Decided: no ceiling on the PrEP gain from centered logits for now (user: not yet relevant); the
  comparison with the Atlanta healthcare-stigma model (Georgia-specific, cascade among people with
  HIV, no AHS feedback) is within range for cumulative incidence.

## Loose PrEP drift specification (2026-10-23)

`loose_prep_drift` (PrEP drift prior scale 1.0 instead of 0.35; pipeline/state_fit.py), added after
finding that coupling tracked slow PrEP growth (states' drifts pooled toward one value let access-
related shortfalls be credited to AHS). All 500 draws converged.
- Coupling survives and strengthens: median -0.72 (primary -0.58); direct -0.40 (-0.33); engagement
  -0.13 (-0.14); rank rho with primary 0.82 (direct 0.81, engagement 0.99).
- Fitted PrEP drifts vary somewhat more (0.34-0.84 vs 0.37-0.70).
- Coupling's rank correlation with 2018-2024 PrEP growth falls from 0.55 to 0.36.
- Largest increases in AL, UT, SC, OH, TX, IN (AL, UT and SC are high/rising-AHS states affected
  by the runaway AHS dynamics; interpret with the state-anchoring revision).
- AHS-scenario 2036 incidence change: median -6.4% (primary -5.1%).
Conclusion: coupling is not only an artifact of the pooled drift; a state-specific PrEP trend is
adopted in the planned state-anchored revision (decision to confirm).

## State-anchored revision (2026-10-23, implemented; full refit pending)

Motivation: in the globally centred model each state could hold its level only with persistence near
1 (no intercepts), giving near-unit roots, runaway projections (AL AHS 36% -> 97%) and implausible
equilibria; coupling was identified mainly across states through the pooled PrEP drift (tracking slow
PrEP growth, confounded with access); the model's total AHS -> PrEP effect was about 40x the
individual-level association.
- AHS wording: shift estimated as the age-adjusted log-odds difference between AMIS 2022 and 2021
  respondents in the same state-period (2021-2022): 0.452 (SE 0.105); placebo contrasts between
  adjacent cycles -0.15 (2018 vs 2017) and -0.23 (2024 vs 2023). State-period AHS moved back by the
  shift x the share of AMIS 2022+ respondents (scripts/01, 03; AHS_WORDING=unadjusted for the
  sensitivity). Adjusted AHS is flat from 2020 (primary-state median 12.2, 10.2, 10.1, 10.0%).
- State anchoring: every measure centred at the unit's own mean logit over the observed periods
  (per draw), i.e. a state-specific intercept; pathways act on departures from a state's usual level.
  Persistence constrained to be non-negative (no oscillation; the short panel biases it downward).
- Drift: PrEP drift prior scale 1.0 (state-specific expansion; loose-drift result); weak zero-centred
  AHS drift (scale 0.35); risk drift unchanged.
- Structure: healthcare contact -> AHS and PrEP -> AHS estimated without a sign constraint (prior 0,
  scale 0.7). The nine PrEP-cluster pathways (PrEP <-> risk, PrEP <-> testing, risk -> disclosure,
  disclosure -> PrEP and testing, risk -> testing, contact -> testing) lowered from 0.15 to 0.05
  (scale 0.3): at 0.15 the cluster was self-amplifying (spectral radius 1.05) and was being stabilised
  only by AHS feedback, which made long-run AHS effects sign-unstable.
- Evidence target (models/sbm/calibration.py; loss term L_total_effect): the model's total AHS -> PrEP
  response over 6 steps under a sustained AHS set-point shift (all routes and feedback acting) is
  asked to match the unit's access-adjusted individual association (scripts/08b: survey-weighted
  Poisson adjusted for insurance, age, race/ethnicity, education, risk and cycle, not for mediators;
  random-effects pooled aPR 0.69, 95% CI 0.58-0.82; state ratios pooled by empirical Bayes), converted
  to a centred-logit slope over the move from the unit's AHS level to the benchmark (AHS reference
  probability); tolerance 0.01. Theory priors decide the split across routes. A fixed-structure
  rescaling of the AHS priors was tried first and failed (the structure becomes unstable when the
  AHS pathways weaken).
- Scenarios: pinned (as before) and shifted set point (``*_shift``: feedback onto the stigma measure
  acts). New outputs: AHS history and AHS-free PrEP trajectories (scripts/13b).
- Specifications: theory_prior (no evidence target), pooled_prep_drift (scale 0.35), placebo_ahs and
  placebo_engagement (periods permuted within unit and draw); loose_prep_drift retired (now primary).
- Smoke check (20 draws): total response matches the evidence in every state; coupling median -0.06
  (direct -0.02, engagement -0.03; theory-only -0.49); spectral radius 0.92; baseline 2036 AHS median
  13% (4.7-33%), PrEP 47%; AHS-free PrEP gap median 0.2 pp in 2024 and 0.9 pp in 2036 (max 3.0);
  AHS-halving 2036 incidence -0.7% (pinned) / -0.4% (shifted); cumulative -0.15% / -0.07%. Placebo
  coupling agrees with primary (rank rho 0.91): magnitudes are evidence-set. PrEP -> AHS is +0.06
  in the primary and -0.04 in the placebo (possible data signal for the feedback).

## Contextual (climate) stigma associations (2026-10-23, exploratory)

Survey-weighted Poisson models of PrEP use (primary states, state weights; confounders insurance, age,
race/ethnicity, education, risk, cycle) with own AHS plus the leave-one-out state-period share of
other respondents reporting AHS (wording-adjusted), GSS and family stigma ("climate", per 10 points):
- Own AHS: PR 0.63-0.65 in every version.
- AHS climate: PR 0.70 (0.63-0.78) across and within states, but 1.06 (0.88-1.28) within states
  only (state fixed effects); with GSS/family climate 0.80 vs 1.07.
- GSS climate 1.35 (1.21-1.52) and family climate 0.77 (0.70-0.85) across states; 0.98 and 1.00
  within states.
Conclusion: contextual associations are between-state only (confounded with access, region, policy);
no within-state evidence that changing a state's climate moves PrEP for everyone. The primary
evidence target stays individual (compositional); a cross-state contextual effect is a candidate
upper-bound sensitivity, clearly labelled. Added scenario s11_eliminate_ahs (AHS to the benchmark,
same ramp).

## Pathway and persistence priors: basis and scale (2026-10-23)

Off-diagonal pathways: signs and existence come from the participatory causal loop diagram and the
literature; evidence on dynamic magnitudes is limited, so pathways are regularised toward modest
effects. The PrEP/risk/testing/disclosure/contact pathways are centred at 0.05 (scale 0.3) so the
network does not encode endogenous amplification unsupported by the short series (at 0.15 the block
had spectral radius 1.05 on its own); stability is imposed separately (penalty), not used to choose
every centre. Sensitivity: cluster_moderate (0.10), theory_prior, sign_only, common_weak.
Probability-scale meaning (a source moving from 10% to 20%, target at 40%):

| Pathway | Target persistence | Next-period target change | Long-run change if the source stays at 20% |
|---|---|---|---|
| 0.05 | 0.80 | +1.0 pp | +4.9 pp |
| 0.05 | 0.90 | +1.0 pp | +10.0 pp |
| 0.10 | 0.80 | +2.0 pp | +10.0 pp |
| 0.10 | 0.90 | +2.0 pp | +20.0 pp |
| 0.15 | 0.80 | +3.0 pp | +15.1 pp |
| 0.15 | 0.90 | +3.0 pp | +29.2 pp |

Persistence (diagonal) priors, scale 0.2; untargeted diagonals are estimated within [0, 3] under the
stability penalty:

| Measure | Centre | Basis |
|---|---|---|
| PrEP | 0.85 | Empirical: pooled AR(1) slope of state centred-logit PrEP, 24 primary states, 2018-2024 (0.846; scripts/08) |
| Healthcare contact | 0.95 | Assumption: stable population-level prevalence |
| General social stigma | 0.85 | Assumption (upstream, slow); between-state differences are mostly sampling noise (reliability 0.04) |
| Family stigma | 0.85 | Assumption (upstream, slow); noise-corrected pooled AR(1) about 0.9 |
| AHS, disclosure, risk, testing | untargeted | Estimated; non-negative (no oscillation) |

Sensitivity: persistence_low / persistence_high (targeted centres -/+ 0.10), varied separately
from pathway strength. Stability: soft penalty above spectral radius 0.999; the share of draws above
1 is reported and such draws flagged in projections.
Contextual upper bound (post-fit, pipeline/joint_simulation.py `contextual_prep_pr`; scripts/11
--contextual-prep-pr): scenario PrEP multiplied by PR ** (AHS change / 0.10), PR = 0.80 (adjusted for
GSS/family climate; conservative bound) and 0.70 (outer bound), per state and draw, then through the
epi model; reported as a deliberately broad bound, not an identified effect (within-state PR 1.06).

## Optimizer iteration cap (2026-10-23)

The full state-anchored refit stopped at scripts/10: 11 of 17,000 primary unit-draw fits (USA in 9
draws, PA draw 147, CA draw 451) ended with "TOTAL NO. of ITERATIONS REACHED LIMIT" (maxiter 500); the
sensitivity fits had 0-131 such fits per specification. maxiter was raised to 5000 (config/optimization.py)
and the optimizer status and message are now saved in the loss-components CSV. L-BFGS-B is
deterministic, so fits that converged within 500 iterations are unchanged; only draws containing a
capped fit were refitted (whole draws, since the national and division fits are the hierarchy
references for the states in the same draw) and spliced into each fit file, which equals a full refit
at 5000. Original fits: ../EEE-SD-Model-2026-archive/fits_maxiter500_20261023/.
Primary check: all 11 converge; the capped fits' losses fall by at most 5e-4; USA stays at the stability
limit (spectral radius 0.999); median state total effect unchanged (-0.027). The small change in the
national reference moves some state J entries in the same draws by up to 0.5 (CA, OH) at nearly equal
loss: individual off-diagonal entries are weakly identified, the total effect is not.
Stability limit: 31% of USA draws and 9.5% of state draws have spectral radius > 0.998 (CO 35%, SC 27%,
CA 25%, PA 16%), i.e. the penalty, not the data, keeps these systems from being non-decaying.

## State-anchored revision: full results (2026-10-23)

24 states, 500 draws; all fits converged (maxiter 5000). Spectral radius median 0.91, share > 1 0.02%.
Total AHS -> PrEP response matches the evidence target (median -0.0275 vs -0.0274).
24-state pooled change in 2036 incidence, median (95% interval):
- Halve AHS, pinned (s1): -0.89% (-1.08, -0.70); cumulative 2024-2036 -0.20%.
- Halve AHS, shifted set point (s1_shift): -0.43%; combined package -0.91% (pinned) / -0.49% (shifted).
- Eliminate AHS to benchmark 5.3% (s11): -1.12% (-1.66, -0.59); cumulative -0.24%.
- GSS / family halving: -0.11% / -0.10%; pathway scenarios s4-s9 within +/-0.1%.
- Contextual bound (PrEP x PR ** (AHS change / 0.10), post-fit): halve AHS -24% (PR 0.80; -36, -15)
  and -36% (PR 0.70); eliminate -31% / -42%; cumulative -10% / -16%. The median state: AHS 13.2% ->
  6.6% in 2036, PrEP 47% -> 57%, incidence -16% (PR 0.80). Incidence moves about 1:1 with PrEP coverage
  in both primary and bound, so the gap is entirely the size of the assumed PrEP response.
- Robust to start year (2018 reconstruction -0.88%), unadjusted AHS wording (-0.95%), drift, persistence,
  cluster, stability and hierarchy specifications (-0.77% to -1.13%); covariance_balanced -1.36%,
  trajectory_only -0.63%. Without the evidence target: common_weak -6.6%, theory_prior -8.1%,
  sign_only -13.1%.
- Placebos (periods permuted) give the same incidence effect (-0.89%, -0.92%): with the total effect
  pinned by evidence, scenario incidence is set by the evidence, not by the within-state dynamics.
- AHS-free PrEP gap in 2036 (pp, median): AL 2.9, UT 2.4, CA 2.3, SC 2.2, NC 2.0; IN, MN ~0; MA, WA ~0.
- History counterfactual to 2028: PrEP effect of the observed AHS change within +/-0.4 pp in all states.

## Evidence target on the old AHS wording, with its uncertainty (2026-10-23)

Question: does the AHS-PrEP association change as PrEP use grows? Pooled median-55 states, state fixed
effects, 08b confounders (scripts/08b outputs ahs_prep_evidence_by_year.csv, _wording_step.csv):
- Year-specific prevalence ratios: 0.55, 0.42, 0.60, 0.56 (2017-2021, old wording); 0.82, 0.85, 0.69
  (2022-2024, new wording). AHS prevalence rises from about 10% to 16% at the wording change.
- With a separate wording step, the time trend is null (ratio x1.02 per year, p = 0.78); the step is
  x1.42 (p = 0.13). Within the old wording the ratio is stable at about 0.5 while PrEP among men
  without AHS doubles (10% -> 20%) and the absolute gap widens (-5 -> -8.6 pp): the ratio, not the
  difference, is the stable scale. The apparent attenuation is the wording change: the new item counts
  more men, presumably with milder stigma, diluting the association.
Decision: the evidence target uses the old-wording cycles (2017-2021), the scale of the
wording-adjusted AHS the model uses (scripts/01). Made for measurement consistency, and it would be
made if it lowered the effect; the all-years target stays as a sensitivity (evidence_all_years) and
is the target of the unadjusted-wording fit (09 --evidence-path ..._all_years.csv).
Old wording: random-effects mean aPR 0.55 (0.45-0.67), between-state SD of log aPR 0.27 (all years:
0.69, 0.58-0.82, SD 0.31). Median state target slope -0.039 vs -0.027 (x1.42). Indiana 0.74 (all
years 1.18).
aPR uncertainty: each draw takes each unit's ratio from N(log pooled aPR, posterior SE) (empirical-Bayes
posterior including the mean's uncertainty; seeded by the source draw index, so specifications are
paired); 1.6% of state draws have aPR > 1. evidence_fixed_apr isolates this change.
Static evidence translation (scripts/11 --constant-pr-prep): scenario PrEP = baseline PrEP x
(1 - A_s (1 - r)) / (1 - A_b (1 - r)) with the draw's r, in place of the fitted PrEP response: what the
individual evidence implies at the projected levels without the dynamics; the comparator showing what
the dynamic model adds.
Full results (500 draws, all fits converged; spectral radius median 0.91, > 1 in 0.02%; total effect
matches the per-draw target, median -0.039). 24-state pooled change in 2036 incidence, median (95%):
- Halve AHS -1.17% (-1.49, -0.92); shifted set point -0.61%; eliminate -1.47% (-2.15, -0.86);
  combined package -1.19%; GSS / family halving -0.15% / -0.13%; upstream social stigma -0.27%.
  Cumulative 2024-2036: -0.27% (halve), -0.32% (eliminate).
- Specifications: evidence_all_years -0.87% (the previous primary), evidence_fixed_apr -1.20% (the
  uncertainty barely moves the median); drift, hierarchy, persistence, cluster, stability, placebos
  -1.02% to -1.34%; covariance_moderate / balanced -1.51% / -1.80%; trajectory_only -0.85%. Without
  the evidence target: common_weak -6.6%, theory_prior -8.1%, sign_only -13.1%. Start 2018 -1.16%;
  unadjusted wording (all-years target) -0.94%.
- Static evidence translation (constant ratio): halve -5.4% (-10.2, -3.0), eliminate -7.7%.
- Contextual bound: halve -24% (PR 0.80) / -36% (PR 0.70); eliminate -32% / -43%.
Check before interpreting the static and contextual tiers: baseline AHS is projected to rise in several
states from 2024 to 2036 (SC 18 -> 36%, FL 10 -> 23%, MA 4 -> 17%, NJ 9 -> 18%, GA 10 -> 19%,
CO 16 -> 24%), while it falls or is flat in others (WA, VA, IN, PA, OH). Both tiers scale with the
projected AHS level; the pathway driving the rise (persistence, drift, PrEP/contact -> AHS feedback)
is to be decomposed.

## System decomposition and the PrEP -> AHS pattern (2026-10-24)

scripts/13d_system_decomposition.py splits each measure's projected 2024-2036 change exactly into own
carryover, carryover from other measures, own drift, feedback on own drift, and drift arriving from
each group of measures (models/sbm/structural.py, trajectory_decomposition).
- PrEP rises 15-40 pp, almost all its own drift (20-52 pp, less own carryover); the rest of the system
  adds 0.5-3.5 pp. PrEP growth is essentially exogenous in the model.
- The projected AHS rise is PrEP feeding into AHS: AHS change from PrEP SC +17.9 pp, CO +12.4, FL +11.9,
  MA +9.9, UT +8.5, GA +6.7, TN +6.1; negative in VA, IN, MI, WA, PA (-1.5 to -3.6). PrEP -> AHS median
  0.075 (positive in 60% of state draws).
- Placebos: permuting AHS periods removes it (median -0.4 pp, state ranking reversed, rho -0.45;
  PrEP -> AHS -0.04); permuting engagement leaves it unchanged. It is learned from the joint timing of
  AHS and PrEP within states.
- loose_ahs_drift (AHS drift prior scale 1.0, so a state's AHS can trend on its own): unchanged (AHS change
  from PrEP rank rho 0.99 with the primary, median 2.0 vs 1.9 pp; PrEP -> AHS 0.072; AHS own drift stays
  near 0). The pattern is not a shared upward trend loaded onto the PrEP route for want of AHS drift.
  Incidence effects unchanged (halve AHS -1.19%).
- Interpretation: a within-state, time-ordered association consistent with healthcare-generated stigma
  (PrEP bringing men into clinical contact), positive in some states and negative in others. Not causal:
  common causes (programme expansion, AMIS recruitment), residual state-specific wording effects after
  the 2022 change, and three transitions per state. A cross-sectional individual check cannot separate
  the directions (associations are symmetric; past-vs-never PrEP users are equally explained by AHS
  causing discontinuation), so none was run. Its projected magnitude scales with unbounded PrEP growth.
- What drives it: the state AHS change from PrEP is rank-correlated -0.90 with the observed AHS change
  2017-18 -> 2019-20 and +0.92 with 2019-20 -> 2023-24, but only 0.13 with 2017-18 -> 2023-24. The
  "positive" states dipped in the 2019-20 period and rebounded while PrEP kept rising (SC 17 -> 7 -> 17
  -> 18, CO 17 -> 5 -> 10 -> 16, FL 13 -> 5 -> 10 -> 10); the "negative" states spiked then fell (VA
  7 -> 16 -> 7 -> 10, MI 10 -> 17 -> 11 -> 11). The 2019-20 period is AMIS 2019 alone; its swings (10-12
  pp) are about four times the bootstrap SD of a state-period estimate (about 2.6 pp). A V-shape is not a
  trend, so loose_ahs_drift cannot absorb it, and permuting periods breaks it. The PrEP -> AHS pattern is
  therefore driven by one cycle's state-specific swings and is reported as exploratory at most; the
  projected AHS rises in SC, CO, FL and MA are partly that cycle extrapolated. Test: add AMIS 2020
  (excluded for COVID-19 disruption) to the 2019-20 period, in a scratch copy with the states and
  evidence target held fixed.
- AMIS 2020 test (2026-10-24; same 6-month items and category codes as 2019; wording shift unchanged,
  0.452): the 2019-20 dips in the "positive" states are reproduced (SC 7.1 -> 7.8%, CO 5.0 -> 4.6%,
  FL 5.4 -> 6.2%, MA 2.7 -> 3.8%), while some "negative" spikes shrink (VA 15.6 -> 11.4%, MI 16.8 ->
  12.8%, MN 12.0 -> 6.7%). The swings are not smaller overall (median deviation of the 2019-20 period
  from its neighbours 2.4 -> 2.9 pp). The PrEP -> AHS pattern persists and strengthens: coefficient
  median 0.075 -> 0.148 (positive in 60% -> 69% of state draws); AHS change from PrEP median 1.9 -> 3.7
  pp, state ranking rho 0.94. Incidence effects unchanged (halve AHS -1.16%, eliminate -1.52%).
  Reading (set before the test): not a quirk of one cycle, since two independent cycles show the same
  state-specific low AHS in 2019-20. It still rests on that period's contrast with the others (the
  2019-20 era may share a recruitment or composition change), remains a within-state association, and
  is reported as an observed pattern consistent with healthcare-generated stigma, not a causal effect.
  The primary analysis keeps AMIS 2020 excluded (decided before the results, for COVID-19 disruption);
  including it is reported as a sensitivity. Restricted copies used for the test were deleted.

## The AHS wording shift may be a post-COVID rebound (2026-10-24, decision pending)

Per-cycle analytic means (unweighted, all states): every stigma measure dips in 2020-2021 (AHS 13.1 ->
10.3 -> 8.7%, general social 28.8 -> 26.3 -> 19.5%, family 27.9 -> 28.6 -> 19.8%) and rebounds in 2022,
plausibly fewer recent (6-month) experiences while contact fell during COVID-19. The general social
and family items did not change wording in 2022 (confirmed by the author), so they are controls for
the AHS item change.
Difference-in-differences (same data, weights, state and age adjustment as scripts/01; respondents
clustered): 2022 vs 2021 log-odds shift AHS +0.443 (the 0.452 used), general social +0.635, family
+0.241; AHS minus control: vs general social -0.19, vs family +0.20, vs both +0.002 (-0.21 to 0.21).
Placebo pairs (no wording change): 2018 vs 2017 -0.19 (both controls), 2024 vs 2023 -0.16. So the AHS
jump in 2022 is not distinguishable from the general rebound; relative to the usual AHS-vs-control
difference, a wording effect of at most about 0.2. The 0.452 adjustment mostly removes the rebound.
Association: year-specific PrEP prevalence ratios step up in 2022 for AHS (x1.42, p = 0.13) and general
social stigma (x1.22, p = 0.11; from 1.0-1.3), not family (x1.03): partly general, partly AHS-specific.
Unadjusted fit: PrEP -> AHS median 0.22 (positive in 76% of state draws; adjusted 0.075, 60%), AHS change
from PrEP median 7.3 pp (adjusted 1.9), same state ranking (rho 0.97). In the unadjusted series the
post-COVID rebound, common to all stigma measures, coincides with PrEP growth, a common cause.
Scripts: exploratory (scratchpad wording_did.R, stigma_prep_by_year.R).
By PrEP status (2021 -> 2022, log-odds): AHS rose +0.82 among PrEP users vs +0.38 among non-users
(difference +0.43, p = 0.06); no such difference for general social (+0.07) or family stigma (-0.09), and
not larger with a recent healthcare visit (-0.29). Most plausibly the item change: the old item (a
provider may learn you have sex with men) rarely applies to PrEP users, who have disclosed; the new item
(a provider may treat you poorly) does. A PrEP -> stigma change cannot be excluded.
Decision (author, 2026-10-24): keep the original item as the construct: old-wording evidence target and
AHS adjusted to the original item's level (primary unchanged); unadjusted AHS stays a sensitivity, and
the difference-in-differences result is reported as a limitation of the uniform level shift.

## Direction of the PrEP-AHS relationship among PrEP users (2026-10-24, exploratory)

Raw AMIS PrEP duration (months in a row; bands <2, 2-6, 7-12, 12+) and stopping reasons, linked to the
analytic file by rowID (scratchpad prep_ahs_direction.R; aggregate output only).
- Duration: among current users AHS falls with time on PrEP: old item 17.1% (<2 months) -> 8.6 -> 5.6 ->
  4.8% (12+), adjusted ratio 0.72 per band (0.60-0.87); new item adjusted 0.79 (0.67-0.94). AHS -> stopping
  (high-AHS men drop out) predicts this; PrEP care generating stigma predicts the opposite. (PrEP care
  reducing AHS is also consistent.)
- Stopping: among men who stopped, those with AHS cited a stigma-related reason (people will know they
  have sex with men / think they are very sexually active / think they have HIV) far more often: old item
  39.7% vs 4.6% (ratio 6.8, 3.4-13.6), new item 14.7% vs 1.6% (8.4, 3.0-23.7).
- Reading (revised with the author): neither check identifies direction. Duration also reflects patterns
  of use (short, episodic use during periods of risk is appropriate use), and lower AHS among long-term
  users is equally explained by selection (men with less AHS stay on longer, "resilience"). The stopping
  association is largely shared construct: the AHS item and the stigma-related stopping reasons measure
  overlapping worries in the same respondent at the same time. Individual cross-sections among PrEP users
  also do not speak to how population AHS changes as PrEP expands within a state (different level of
  analysis). These results are not used to interpret the state-level co-movement in either direction and
  are not reported as evidence of direction.
- Durability (shifted / held AHS effect, 0.19 MA to 0.98 CO) survives both placebos (rank rho 0.92,
  0.99): it comes from the persistence priors, not the data, and is not reported as a finding.
- State ranking of incidence effects survives both placebos (0.97, 0.98); the evidence target and the
  projected AHS level explain about 60% of the between-state variance.

## Paper scenarios and the PrEP -> AHS feedback check (2026-10-24)

Scenario set (config/interventions.py PAPER_SCENARIOS; decided with the author): halve AHS from its
2024 level; AHS to the benchmark (5th percentile of observed state-period levels, 5.3%); stigma-
resilient PrEP delivery (half of the benchmark scenario's PrEP change, AHS and all else at baseline; a
derived policy comparison); halve general social and family stigma from 2024. Supplement: AHS held at
or below its 2024 level. Cuts relative to 2024 are held and never above the baseline path
(probability_relative_to_last_observed). Dropped from the paper: combined package (equal to AHS alone),
single-pathway and healthcare-generated-stigma scenarios (routes not identified), one-time vs
maintained (not translatable; durability prior-driven), dose-response.
Results (500 samples, 2036 incidence, 24-state pooled median, 95% interval; cumulative 2025-2036):
- Halve AHS -1.48% (-2.15, -0.93); cumulative -0.32%. AHS to benchmark -1.47% (-2.15, -0.86); -0.32%.
  (They coincide on average: half the median 2024 level is near the benchmark.)
- Resilient PrEP delivery -0.74% (-1.08, -0.43): half the benchmark scenario by construction.
- Halve upstream stigma -0.27%. AHS no worsening -0.55%.
- Specifications (halve AHS): anchored -0.95% (trajectory only) to -2.05% (covariance balanced);
  all-years target -1.09%; placebo AHS -1.26%; theory/sign/common-weak priors -8.5% to -18%.
PrEP -> AHS off in projections (11 --zero-projection-edge prep_used:stigma_ahs; baseline and
scenarios): halve AHS -1.21% (-1.61, -0.87), benchmark -1.16%, resilient -0.58%, no worsening -0.22%,
upstream unchanged. Projected AHS no longer rises (2036 baseline SC 36 -> 15%, MA 17 -> 4%, FL 23 ->
9%, CO 24 -> 10%). State ranking with vs without feedback: rho 0.71 (halve AHS), 0.89 (benchmark).
State drivers (benchmark scenario, rank correlation with the incidence effect): without feedback,
observed 2024 AHS 0.87, the state association 0.65, leverage 0.42; with feedback, projected 2036 AHS
0.93, observed 2024 AHS 0.78. Robust positions: UT, AL, SC, CA, TN, GA, NC at the top; WA, IN, OH, MO,
MN at the bottom. Positions that depend on projected AHS: MA (rank 14 -> 24 without feedback), NY
(15 -> 21), CO (9 -> 14), FL (7 -> 10) gain from projected AHS rises; VA (22 -> 16), MI (16 -> 12),
PA (20 -> 17) lose from projected falls.
Proposed presentation (pending the author): state comparisons use observed 2024 AHS as the burden driver; the effect is reported as
-1.2% to -1.5% (without / with the fitted feedback); states whose positions depend on projected AHS
are flagged.

## Final design: static evidence translation (2026-10-25, frozen)

Decision (author, with two external reviews): the dynamic model describes co-evolution and projects each
state's baseline; the scenario PrEP change comes directly from the respondent-level prevalence ratio
(per state and draw, at the projected levels: P_s = P_b (1 - A_s (1 - r)) / (1 - A_b (1 - r)); 11
--constant-pr-prep); the epidemiological model translates it. The in-fit evidence constraint is kept
(fits unchanged) and described as constraining the fitted AHS-PrEP relationships to be consistent in
magnitude with the association; it does not set any scenario result. The open-loop anchor, 3-step
anchors, PrEP-persistence 0.6 refit, durability and dose-response were dropped. Resilient delivery
under the static translation: half of the benchmark scenario's PrEP change, AHS and all else at
baseline. Feedback decision rule (fixed before results): feedback-on stays primary only if, under the
AHS placebo, the feedback increment falls below half of its primary value and the placebo PrEP -> AHS
coefficient is centred near or below zero; otherwise feedback-off is primary.
Results (500 samples; 24 states pooled; 2036 incidence change, 95% interval; cumulative 2025-2036
infections averted of about 173,000):
- Halve AHS from 2024: feedback on -7.6% (-14.6, -3.6), cumulative -2.95%, about 5,000 averted
  (2,700-8,700); feedback off -3.3% (-5.3, -2.1), cumulative -1.47%, about 2,500 (1,700-3,700).
- AHS to benchmark: -7.7% / -3.4% (on / off). Resilient PrEP delivery: -4.3% / -1.7%, cumulative about
  2,800 / 1,300 averted. Halve upstream stigma: -2.7% / -1.8%. No worsening: -5.2% / -0.8%.
- Unadjusted wording (all-years ratio): halve AHS -9.4%. Specifications (halve AHS, feedback on):
  all-years ratio -5.3%, persistence low -4.7%, persistence high -12.5%, placebo AHS -4.7%.
- Feedback rule: increment (on minus off, halve AHS, 2036) -4.30 points in the primary vs -1.43 under the
  AHS placebo (33%); placebo PrEP -> AHS coefficient centred below zero (-0.04). Rule met: feedback-on
  primary, feedback-off reported in parallel. More than half of the primary effect depends on projected
  PrEP-AHS co-movement.
- Observed AHS (80% of bootstrap draws agree on direction): 2018-2024 falling in 2 states (PA, OH),
  rising in 1 (WI), no clear change in 21; 2018-2020 falling in 7; 2020-2022 rising in 4 (FL, CO, SC,
  UT). PrEP rose clearly in all 24.
Validation: paired baselines identical across scenarios; cumulative averted consistent with the
percentage changes; resilient delivery exactly half of the benchmark PrEP change with AHS unchanged.
Architecture frozen; next: reporting scripts (13, 13c, 14-16) and text.

## AtlasPlus inputs consolidated (2026-10-25)

All AtlasPlus indicators now come from one tidy file, inputs/cdc_raw/atlasplus_combined.csv (9,843 rows),
built from the October 22 exports plus October 25 re-exports of estimated incidence, estimated
prevalence and knowledge of status (both sexes, 2017-2022), PrEP coverage for men (2017-2023) and PrEP
coverage by age group (both sexes; 13-24 to 45-54). The earlier atlasplus_hiv.csv and
atlasplus_prep_by_age.csv are retired. scripts/00_build_cdc_stocks.py reads only the combined file
(the base-plus-update merge and its duplicate-diagnoses fix are gone). Check: rebuilt cdc_stocks.csv and
national_prep_age_targets.csv identical to the previous ones in every value and missing pattern, apart
from 30 empty USA placeholder rows the old download carried; nothing downstream changes. Estimates
flagged unstable by AtlasPlus are kept (cases_unstable). Raw exports and the combine script are in
../EEE-SD-Model-2026-archive/inputs_cdc_raw_atlasplus_20261025/ (the script reproduces the combined
file byte for byte there); sources in inputs/cdc_raw/SOURCES.md. The superseded epidemic posterior
inputs/epi_v7d/ moved to ../EEE-SD-Model-2026-archive/inputs_epi_v7d_20261025/.
2023 PrEP coverage: the exports cover 2023 but every male and age-group value is "not available" (male
coverage has values for 35-46 states in 2017-2022 and none in 2023; all 24 primary states 2017-2022). The
2023 PrEP-eligible imputation (2017-2022 mean) therefore stays. Only the unused both-sexes, all-ages
series has some 2023 values.
Found for the code review: the all-population "New diagnoses (not AIDS)" stock subtracts an
all-population AIDS series that no download provides, so it equals total diagnoses; unused downstream.

## Open items

- AHS rose sharply between 2021 and 2023 in every demographic group simultaneously (e.g. White
  MSM 25% -> 31%, Black MSM 21% -> 31%). The AHS items did not change; the user attributes it
  to lower recent survey volume. Revisit before submission: it drives much of the 2022 -> 2024
  transition in the state models.

## Consequences for reported numbers

- Table S2: 2017-2018 eligible N decreases (AMIS 2016 removed); calibration remains exact.
- S1.4: national weight CVs 1.28-1.51; state weighting (capping) needs its own description.
- Table S3, S2.3 PrEP drift target (0.468), S2.4 PrEP persistence (0.746), S6.5 California n_eff, and all fitted results are regenerated.
- Population regularization tiers: two swaps for ages <=24 (disclosure -> testing now strong; healthcare visit -> testing now weak).
