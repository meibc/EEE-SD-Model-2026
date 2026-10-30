# Code review guide (from 2026-10-26)

Order is by how much each part can move the paper's numbers, not by file number. Scripts 00-02 are
reviewed. For each file: (1) what it reads and writes, (2) does the docstring match the code, (3) are the
key decisions those in docs/analysis_decisions.md, (4) hard-coded paths or numbers, (5) leftovers from
earlier designs. Line counts in brackets.

## Stage 1: what sets the scenario effect (highest impact)

The headline effect is: projected baseline AHS and PrEP (from the fits) x the AHS-PrEP prevalence ratio
(from 2_evidence/01) -> scenario PrEP -> epidemic model. These files decide it.

- [x] `scripts/2_evidence/01_ahs_prep_evidence.R` (REML via metafor, 2026-10-27; docstring updated; refit needed) - the prevalence ratio (0.55) that drives every scenario.
  - Confounders: insurance, age, race/ethnicity, education, risk, cycle; mediators (visit, disclosure,
    testing) deliberately not adjusted. Agree?
  - Old-wording cycles only (2017-2021) for the primary; state pooling (DerSimonian-Laird, empirical Bayes);
    `se_pooled` includes the mean's uncertainty.
  - Uses the median-55 states (a superset of the primary 24): intended?
- [x] `pipeline/projection.py` (reviewed and cleaned 2026-10-27) - the translation P_s = P_b (1 - A_s (1 - r)) / (1 - A_b (1 - r))
  (`constant_pr_prep`), the contextual bound, `prep_only_fraction` (resilient delivery), the PrEP -> AHS
  switch (`zero_projection_edges`), and the draw pairing (now shared across states).
- [x] `models/scenarios/intervention.py` (cleaned 2026-10-27) - how "halve from the 2024 level" is applied: target =
  min(2024 value x 0.5, baseline), ramp 2028-2032, held after.
- [x] `config/interventions.py` (reduced to the five scenarios 2026-10-27) - the five paper scenarios plus 39 older scenario codes no longer used
  (candidates to remove).

## Stage 2: what sets the baselines

- [x] `scripts/1_data/04_state_bootstrap.R` - reviewed by the author (restructured; model inputs verified identical).
- [x] `scripts/3_model/01_build_state_inputs.py` [44] - turns the bootstrap into the NPZ the fits read. (first pass 2026-10-28; see Package code A)
- [x] `scripts/1_data/05_state_precision_screen.R` (ported from Python 2026-10-27, identical outputs) - the 24-state rule (all 32 cells, median n_eff >= 60,
  25th percentile >= 40); the n_cells fix.
- [x] `scripts/1_data/06_reported_constants.R` - references (5th percentiles), PrEP drift 0.420, persistence 0.846.
  Reviewed 2026-10-27; ported from Python (identical outputs), docstring updated, within-state persistence
  (0.769) added as a reported, unused constant.
- [x] `pipeline/dynamic_fit.py` + `models/dynamic/estimation/calibration.py` - state anchoring and the evidence target (first pass 2026-10-28; see Package code A)
  constraint (6 steps, tolerance 0.01), per-draw prevalence ratio, specifications.
- [x] `scripts/3_model/03_fit_dynamic_model.py` [80], `scripts/3_model/04_build_bundle.py` [28] - wiring. (first pass 2026-10-28; see Package code A)
- [x] `config/sem_*_v3.csv`, `config/optimization.py`, `config/shrinkage.py` - check against Table S4 (first pass 2026-10-28; see Package code A)
  (results/sections/supplement/tableS_model_specification.csv is generated from them).

## Stage 3: epidemic model and scenario runner

- [ ] `scripts/3_model/02_fit_epi_model.py` [61] (+ `pipeline/epi_fit.py`) - calibration years,
  priors, the post-2021 multiplier; matches supplement S3?
- [ ] `scripts/3_model/05_run_scenarios.py` [76] + `pipeline/scenarios.py` [380] - report years, draw-level
  output (.csv.gz). Known: the default output folder is still `results/04_scenarios/primary` (old
  design); every paper run passes --output-dir. Point the default at `paper_static`.

## Stage 4: secondary analyses

- [x] `scripts/2_evidence/02_wording_items.R` - trimmed 2026-10-27 to the Fig. 1B series and the 2022 shift table.
- [x] `scripts/2_evidence/03_climate_associations.R` - reviewed 2026-10-27; standard errors clustered by state.
- [ ] `scripts/5_demographic/01_group_associations.R` [96], `02_group_bootstrap.R` [89], `03_demographic_descriptives.py`
  [155] - demographic descriptives only (group dynamic models removed).
- [x] `scripts/13d_system_decomposition.py` - retired to deprecated/ (2026-10-28; not in the paper outputs)

## Stage 5: reporting (lowest risk; read-only on model outputs)

- [ ] `reporting/sources.py`, `summaries.py`, `style.py` - paths, interval definitions (2.5/25/50/75/97.5),
  pooling (sum over states within draws).
- [ ] `scripts/4_results/`: `fig1_trajectories.py`, `fig2_projections.py`,
  `table1_fig3_ahs_reduction.py`, `tableS_sensitivity.py`, `figS_coevolution.py`, `supplement.py`.

## Package code (the model itself; scripts above mostly call into these)

All modules below are used by the pipeline (import graph from scripts/, 2026-10-28). Line counts in brackets.
Suggested order: the fit first (it is what the refit runs), then the bundle, the epidemic model, scenarios.

### A. Dynamic model fit (refit depends on it)
- [x] `config/dynamic.py` [125], `config/optimization.py` [32], `config/shrinkage.py` [10] (first pass 2026-10-28) - settings, priors,
  mask; compare with Table S4 (results/sections/supplement/tableS_model_specification.csv)
- [x] `models/dynamic/estimation/transforms.py` [66] (first pass 2026-10-28) - centring (state anchor), covariance transform
- [x] `models/dynamic/estimation/loss.py` [246] (first pass 2026-10-28; stale default weights removed) - the objective: trajectory fit, covariance, priors, shrinkage,
  stability, evidence term
- [x] `models/dynamic/estimation/jacobian.py` (first pass 2026-10-28) - the optimizer (L-BFGS-B, starts, gradients);
  get_bounds (sign constraints and bounds; merged from constraints.py 2026-10-28)
- [x] `models/dynamic/estimation/calibration.py` (walked through 2026-10-28) - the total AHS -> PrEP response and its gradient (evidence constraint)
- [ ] `models/dynamic/estimation/hierarchy.py` - one draw through the hierarchy (moved from state_fit 2026-10-28) and
  ShrinkageCalculator (state -> division -> nation weights; merged from shrinkage.py 2026-10-28)
- [x] `pipeline/dynamic_fit.py` (first pass and cleanup 2026-10-28; author to read in detail) - fits every unit and draw; evidence slopes; specifications
- [x] `data/state_inputs.py` [99] (reviewed 2026-10-28; author restructured earlier) - reads the bootstrap draws for the fit

### B. Bundle and data structures
- [ ] `artifacts/research_bundle.py` [107], `artifacts/standardized_runtime.py` [222] - the fit + epidemic bundle the
  scenarios read (dashboard-compatible format)
- [ ] `data/state_inputs.py` - the model-input file (written once by scripts/3_model/01; read by 3_model/02, 03, 04)
- [ ] `data/unit.py`, `data/workbook.py` (workbook reading and unit construction, scripts/3_model/01 only; merged from
  loader.py, data_prep.py and unit_builder.py), `core/transforms.py`, `core/years.py`

### C. Epidemic model
- [ ] `pipeline/epi_fit.py` - priors, likelihood, sampling and bundle (scripts/3_model/02); uses the shared probability transforms
- [ ] `models/epi/prediction/predictor.py` [54], `models/epi/params.py` [18] - the incidence and diagnosis equations
- [ ] `core/years.py` - shared year alignment and extension for fitting and projection

### D. Scenarios
- [x] `pipeline/projection.py` [285], `models/scenarios/intervention.py` [81], `config/interventions.py` [42],
  `models/dynamic/prediction/predictor.py` [57] - reviewed and cleaned 2026-10-27 (author still to read)
- [ ] `models/scenarios/prep_translation.py` - the three scenario PrEP translations (extracted 2026-10-28; tested)
- [ ] `pipeline/scenarios.py` [185] - runs the baseline and requested scenarios and writes their output tables
- [ ] `pipeline/scenario_results.py` [130] - paired endpoint extraction, cumulative incidence and summary rows

### E. Other
- [x] `models/dynamic/structural.py` and its tests - retired to deprecated/ with 13d (2026-10-28)
- [ ] `reporting/` [186] - see Stage 5

## Known items for the cleanup pass

- scripts/3_model/01 docstring points to results/logs/revision_run.sh (untracked) for the unadjusted-wording inputs;
  point it to the pipeline run script written in the reorganization.

- [x] Naming pass done 2026-10-28 ("dynamic" / "epi"; table in docs/analysis_decisions.md).

- [x] Folder pass done 2026-10-28: scripts in stage folders (1_data, 2_evidence, 3_model, 4_results, 5_demographic);
  input/ split into data/state_inputs.py and artifacts/ (bundle writer and reader); section headers in
  pipeline/epi_fit.py (kept as one file); results/07_structural (retired outputs) moved to the Trash.

- If the epidemic model is refitted: hold the 2017-18 survey value for 2017 instead of extrapolating (see the
  decision log); consider splitting pipeline/epi_fit.py (inputs, PyMC model, bundle) and moving
  STATE_POPS to a documented CSV.

- [x] Settings consolidation done 2026-10-28: config/specifications.py and config/projection.py (decision log).

- [x] Package code used only by retired scripts: models/dynamic/structural.py and its tests moved to deprecated/ (2026-10-28).
- [x] `scripts/1_data/02_build_cdc_stocks.R`: reviewed by the author 2026-10-27; output pruned to the six indicators the
  pipeline reads (resolves the unused "New diagnoses (not AIDS)" stock).
- Specification fits in results/05_sensitivity/model_specification/fits/ that no output uses.
