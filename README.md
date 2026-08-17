# EEE-SD-Model-2026

Staged SEM + epidemiologic scenario workflow for the centered-logit drift SEM model and dashboard-ready standardized input export.

## Current staged workflow

Run stages explicitly. Stage 02 and Stage 05 are long/optional and should only be rerun when their inputs or model assumptions change.

```bash
python scripts/01_generate_sem_input_draws.py
python scripts/02_fit_sem_draws.py
python scripts/03_export_dashboard_standardized_v8.py
python scripts/04_run_sem_scenarios.py
python scripts/05_fit_hierarchical_epi_model.py
```

Recommended routine use after `output/standardized_input_v8.npz` already exists:

```bash
python scripts/04_run_sem_scenarios.py
```

## Stage responsibilities

- `01_generate_sem_input_draws.py`: builds SEM Monte Carlo input draws.
- `02_fit_sem_draws.py`: fits SEM `J` and drift draws. This is optional/cached once `output/sem_fits_centered_drift_v5.npz` exists.
- `03_export_dashboard_standardized_v8.py`: merges the Stage 02 SEM fit into a dashboard-ready standardized NPZ.
- `04_run_sem_scenarios.py`: runs scenarios from `output/standardized_input_v8.npz`; does not refit SEM.
- `05_fit_hierarchical_epi_model.py`: optional long-running PyMC hierarchical epidemiologic calibration. Do not run casually.

## Key artifacts

- `output/sem_fits_centered_drift_v5.npz`: internal Stage 02 SEM fit artifact.
- `output/standardized_input_v8.npz`: canonical dashboard/runtime artifact containing CDC parameters plus SEM `J`, drift, and reference probabilities.
- `output/sem_scenarios_centered_drift_v5/`: scenario endpoint summaries.

## Configuration

- `config/sem_variable_config_v3.csv`: SEM variable reference probabilities and drift priors.
- `config/sem_structural_priors_v3.csv`: SEM structural sign constraints and soft prior settings.
- `config/interventions.py`: scenario/intervention codebook.

## Runtime loaders

The active scenario workflow reads SEM and CDC parameters from `standardized_input_v8.npz` via `input/standardized_runtime.py`. The old monolithic pipeline and legacy CDC/SEM parameter loaders have been retired.
