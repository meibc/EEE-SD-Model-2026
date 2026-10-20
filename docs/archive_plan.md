# Archive plan (draft for approval)

Goal: the repository keeps only the code and frozen inputs needed to regenerate
every result in the paper and supplement under the final specification:

- AMIS inputs: 500 joint respondent-bootstrap replicates, fixed raking weights,
  HIV testing = min(test2yrs, 4) / 4 (`output/amis_joint_bootstrap_direct_v2_n500`).
- Centered-logit references: empirical 5th percentiles (now in
  `config/sem_variable_config_v3.csv`), for states, demographic groups, and CA Hispanic/Latino.
- Epidemiological posterior: retained v7d posterior (2,000 draws), not recalibrated.

Nothing below is deleted unless listed under "Delete". Everything else is moved,
with its relative path preserved, to the archive folder
(proposed: `../EEE-SD-Model-2026-archive/`, outside the repo).

All existing results are produced under the old references, so they are archived
and regenerated, not kept.

---

## 1. Keep: code

| Path | Role | Supplement |
|---|---|---|
| `config/` (all `.py` + 3 CSVs) | Model specification, references, priors | S2, Table S4 |
| `core/`, `data/`, `input/`, `models/` | Model implementation | S2, S3 |
| `pipeline/hierarchical_epi_fit.py` | Epi calibration | S3 |
| `pipeline/joint_simulation.py`, `pipeline/results.py`, `pipeline/sem_scenarios.py` | Scenario projection | S4, S5 |
| `pipeline/priority_population_sem.py`, `pipeline/priority_population_sem_mc.py` | Demographic-group fits | S6 |
| `scripts/amis_joint_respondent_bootstrap_v1.Rmd` | Respondent bootstrap | S1.5, S5.1 |
| `scripts/16_build_joint_bootstrap_sem_inputs.py` | Bootstrap → model input draws | S5.1 |
| `scripts/12_final_v9_information_source_ablation.py` | Primary 500 refits + S7 ablations | S2, S5.2, S7.1–S7.4 |
| `scripts/export_current_research.py` | Builds standardized fit + epi bundle | S5.4 |
| `scripts/04_run_sem_scenarios.py` | Scenarios, paired draws | S4, S5.4–S5.5 |
| `scripts/11_finalize_state_screen_and_response_categories.py` | Precision screen, response profiles | S1.6, S4.5, S7.6 |
| `scripts/14_absolute_stigma_reduction_sensitivity.py` | Fixed 5/10-pp scenarios | S7.5 |
| `scripts/05_fit_hierarchical_epi_model.py` | Epi calibration (optional rerun) | S3 |
| `scripts/17_build_population_joint_bootstrap.R`, `scripts/18_fit_population_joint_bootstrap.py` | Demographic groups + CA Hispanic/Latino | S6 |
| `scripts/99_build_population_regression_prior_fit_table.py` | Regularization-tier table | S6.2 |
| `scripts/prepare_supplementary_results_s5_s10.py` | Data Tables 5–10 | Results summary |
| `scripts/plot_supplementary_fit_diagnostics.py` | Supp. Fig. 2–3 | S2, S3.8 |
| `scripts/plot_supplementary_sensitivity_summary_v2.py` | Supp. Fig. 4 | S7 |
| `scripts/build_expanded_si_tables_s3_s4.py` | Tables S3/S4 (see Q2) | S1.6, S4 |
| `paper_figure_scripts/98_*.py`, `99_*.py` | Main figures 1–5 | Main text |
| `tests/test_current_export.py` | Export checks | |
| `requirements*.txt`, `README.md`, `.gitignore` | Environment, docs (README rewritten later) | |

## 2. Keep: frozen inputs (not regenerated in the default run)

| Path | Why frozen |
|---|---|
| `output/amis_joint_bootstrap_direct_v2_n500/amis_joint_model_draws_direct_v2.{rds,csv.gz}` + `amis_*_v1.csv`, `amis_production_reproduction_audit_v2.csv`, `amis_direct_division_bootstrap_diagnostic_v1.rds` | Rmd output; needs restricted raw AMIS data |
| `output/amis_joint_bootstrap_direct_v2_n500/sem_input_draws_direct_bootstrap_n500.{npz,*.audit.csv}` | Model input draws (script 16) |
| `output/standardized_input_v9.npz` | Holds the 2,000-draw v7d epi posterior and CDC inputs |
| `output/population_joint_bootstrap_n500/population_joint_bootstrap_*` (draws, point trajectories, centering audit, metadata) | R output (script 17); needs raw AMIS data |
| `outputs/ca_hispanic_latino_sem_inputs_corrected/` | CA Hispanic/Latino inputs used by script 18 |

These will move into one `inputs/` folder during restructuring, so all regenerated
results can live under a clean `results/` folder.

## 3. Archive: code

| Path | Reason |
|---|---|
| `scripts/01_generate_sem_input_draws.py`, `02_fit_sem_draws.py`, `03_export_dashboard_standardized_v8.py` | Old Beta-sampling (v5) pipeline and dashboard export |
| `scripts/06_run_priority_population_sem.py`, `07_run_priority_population_sem_mc.py` | Pre-bootstrap population analysis; replaced by 17/18 |
| `scripts/08_*`, `09_*`, `10_*` | Exploratory sensitivities not in S7 |
| `scripts/13_summarize_prior_regularization_comparison.py` | Old-input n500 comparison; S7.1 uses the n100 bootstrap subset |
| `scripts/15_ahs_pathway_family_ablation.py`, `21_stigma_drift_sensitivity.py`, `22_gss_persistence_sensitivity.py` | Diagnostics not in S7 |
| `scripts/20_compare_population_uncertainty_versions.py`, `21_decompose_ga_ahs_prep_response.py` | One-off comparisons and diagnostics |
| `scripts/23_national_cross_sectional_sem.R` | Cross-sectional benchmark; not reported in the paper |
| `scripts/build_compact_si_tables_s3_s4.py` | Expanded version kept |
| `scripts/plot_supplementary_sensitivity_summary.py` | Superseded by `_v2` |
| `pipeline/dashboard_export.py`, `sem_mc_fits.py`, `sem_mc_inputs.py` | Used only by archived scripts 01–03 |
| `pipeline/export.py`, `loaders.py`, `intervention_compare.py`, `visualization/` | Not imported by any kept script |
| `main.py` | Lists the retired 01–05 workflow; replaced by a new run entrypoint |
| `notebooks/` (`division_plots.ipynb` + 28 MB of overlays) | Exploratory |
| `test.ipynb`, `test2.ipynb`, `test3.ipynb`, `testcalib.ipynb` | Scratch |
| `work/` | Rollback copies from the earlier consolidation |
| `docs/consolidation_checks.md`, `docs/consolidation_manifest.json` | History of the migration; superseded by this cleanup |

## 4. Archive: data and results

| Path | Reason |
|---|---|
| Root `cdc_posteriors*.nc`, `sem_mc_results.npz`, `sem_mc_samples_v2–v4.npz`, `standardized_input.npz`, `trans_results.npz` | Old-version artifacts (`.nc` and two `.npz` are already staged for deletion in git) |
| `Factor Analysis v3.xlsx`, `Factor Analysis Final.xlsx` | Old data versions; scripts use `EEE/Factor Analysis v5.xlsx` (see Q5) |
| `output/` — every directory and file not listed in section 2 | Old references, old inputs, superseded figures, or exploratory runs (2.9 GB) |
| `output/amis_joint_bootstrap_direct_v2_n500/` — fits, scenarios, `paper_figures/`, `diagnostics/`, `state_screen_*`, `absolute_*`, `standardized_input_v9_direct_bootstrap.*` | Old references; regenerated |
| `output/population_joint_bootstrap_n500/fits_*`, comparison CSVs | Old references; regenerated |
| `output/amis_joint_bootstrap_v1*` | Superseded by direct_v2 bootstrap |
| `output/empirical_p05_reference_sensitivity_*`, `output/current/` | New references but old v5 inputs |
| `outputs/` — everything except `ca_hispanic_latino_sem_inputs_corrected/` | Pre-bootstrap population runs, superseded figure 5 versions, diagnostics |

## 5. Delete (no information content)

- All `__pycache__/` directories and `.DS_Store` files
- `~$Factor Analysis Final.xlsx` (Excel lock file, currently tracked in git)
- `README_files/` (Quarto render assets for a README that is Markdown)
- `*.xlsx.inspect.ndjson` (tool scratch output)

---

## Gaps to fix during restructuring (not archive decisions)

1. **External scratch paths.** Scripts 12, 17, 18 and 99 read from
   `~/Documents/Codex/2026-08-13/d/outputs/`: the regional covariance workflow,
   the raked population `.rds`, and the population structural priors. Script 12
   loads the regional covariance even when that variant isn't run. The needed
   files will be copied into `inputs/`, or the dependency removed.
2. **Old-repo paths.** Data Table 9 epi diagnostics and the Figure 5 source come
   from `EEE-SD-Model/`. Copy the inputs in and regenerate.
3. **Missing generator.** No script builds
   `standardized_input_v9_direct_bootstrap.npz`. `export_current_research.py` does
   the same job for the new fit; generalize it.
4. **Values cited in the supplement with no generating script:**
   - the PrEP drift target 0.468 (46 states, S2.3)
   - the PrEP AR(1) persistence φ = 0.746 (S2.4; output exists in
     `output/prep_persistence_diagnostics/`, no script)
   - weighting CV/design-effect values (S1.4)

   Add a small `derive_constants` script.
5. **Silent defaults.** Scripts 11, 12, 14 and 16 default to old inputs: v5 draws,
   v1 bootstrap, old scenario draws. This caused the input mix-up in the current
   fit. Make the inputs required arguments or point the defaults at the final inputs.
6. **`config/sem.py`** defaults `data_path` to `Factor Analysis v3.xlsx`; point it at v5.
