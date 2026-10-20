# EEE-SD-Model-2026

Canonical research repository for the EEE stigma–prevention model and paper.
Data preparation lives in `AMIS-Variable-Trends` and upstream `EEE` files.
Dashboard integration lives in `EEE-Model-Dashboard-Integration`.

## Current accepted workflow

The primary state analysis uses joint respondent bootstrapping with fixed final
raking weights, 500 refits, empirical fifth-percentile references across eight
constructs, and HIV-testing frequency (`min(test2yrs, 4) / 4`). The retained
v7d epidemiological posterior is reused; routine scenario runs do not recalibrate it.

Primary scenarios: AHS reduction, upstream social/family stigma reduction, and
all three stigma levels reduced. Reductions are 50% relative to baseline, ramped
over three two-year steps. The combined scenario contains no pathway attenuation.

## Run from this repository

```bash
python -m pip install -r requirements.txt
python scripts/export_current_research.py
python scripts/04_run_sem_scenarios.py --save-draw-level-endpoints
```

The exporter uses the completed 500-refit empirical-reference fit and retained
CDC posterior bundle already on disk. It validates state/variable/year alignment
and exports **only the 17 refitted primary states**. It deliberately does not
apply new references to states whose parameters were fitted under older references.

Required cached inputs:

- `output/standardized_input_v9.npz`: retained epidemiological parameters and observations.
- `output/empirical_p05_reference_sensitivity_n500/fits/empirical_p05_references.npz`: current fitted state matrices, drift, and reference vector.

Output: `output/current/standardized_research.npz`, with a checksum manifest.
Scenario results go to `output/current/scenarios/`. Generated artifacts are local
and ignored by Git. They must be supplied separately when cloning the repository.

## Source layout

- `config/`, `core/`, `data/`, `input/`, `models/`, `pipeline/`: model implementation.
- `scripts/amis_joint_respondent_bootstrap_v1.Rmd`: respondent bootstrap preparation.
- `scripts/16_build_joint_bootstrap_sem_inputs.py`: assembly of model input draws.
- `scripts/12_final_v9_information_source_ablation.py`: state fitting and sensitivity implementation.
- `scripts/17_build_population_joint_bootstrap.R`, `18_fit_population_joint_bootstrap.py`: demographic analyses.
- `paper_figure_scripts/`: paper figure preparation and rendering.
- `scripts/05_fit_hierarchical_epi_model.py`: optional long-running calibration; dependencies in `requirements-calibration.txt`.

## Consolidation status

Current analysis scripts and local results were transferred from `EEE-SD-Model`.
Its output directories point here so existing chats can still find their results.
`docs/consolidation_manifest.json` records transfers; ignored `work/` holds rollback
copies of overwritten generated figures. Neither repository history was rewritten.

The numerical runtime is consolidated. The complete from-source submission run is
still being assembled: some upstream data paths are machine-specific, older
analysis/figure defaults need updating, and the all-state dashboard artifact needs
its own consistency check. Existing figures are not automatically relabeled as
current; regenerate them from the current scenarios before submission.
