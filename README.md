# EEE-SD-Model-2026

SEM + CDC joint modeling pipeline with deterministic and uncertainty simulation, interventions, export, and plotting.

## Core Structure

- `main.py`: entrypoint (`run_pipeline` wrapper).
- `pipeline/pipeline.py`: top-level orchestration.
- `pipeline/joint_simulation.py`: SEM -> CDC deterministic and uncertainty runners.
- `pipeline/results.py`: shared result dataclasses.
- `config/`: run/model/intervention/plotting/data configs.
- `data/`: data loading and parameter loaders.
- `models/sbm/`: SEM estimation + prediction.
- `models/epi/`: EPI prediction.
- `visualization/plotter.py`: interactive plotting.

## Environment

Use `.venv313`:

```bash
source .venv313/bin/activate
python -V
pip install -r requirements.txt
```

## Required Input Files (repo root)

- `Factor Analysis Final.xlsx`
- `cdc_posteriors.nc`
- `trans_results.npz`
- `sem_mc_samples_v2.npz` (default in `RunConfig.sem_params_path`)

## How To Run

```bash
python main.py
```

Runtime behavior is controlled by [`config/run.py`](/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/config/run.py).

Key switches:

- `execution_mode`: `"run"` or `"plot_only"`
- `sem_fit_mode`: `"fit_and_save"`, `"fit_no_save"`, `"load"`
- `joint_mode`: `"none"`, `"deterministic"`, `"uncertainty"`
- `scenario_mode`: `"baseline"`, `"intervention"`, `"compare"`

Intervention codebooks are defined in [`config/interventions.py`](/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/config/interventions.py).

## Outputs

Saved under `output/`:

- `output.pkl` (SEM output)
- `joint_output.pkl` or `joint_baseline.pkl`/`joint_intervention.pkl`
- `uncertainty_output.pkl` or `uncertainty_baseline.pkl`/`uncertainty_intervention.pkl`
- `unified_outputs.csv`

`run_pipeline(...)` also returns a structured container:

- `result["simulation"].deterministic.output/baseline/intervention`
- `result["simulation"].uncertainty.output/baseline/intervention`

## Plotting

Configured in `RunConfig` and `config/plotting.py`:

- `show_state_plots`
- `show_sem_j_violin_plots`
- `states_to_plot`
- `n_states_to_plot`

Plots are shown interactively (not auto-saved).

## Notes

- If `execution_mode="plot_only"`, existing pickle outputs are loaded; SEM fitting is not run.
- If you update SEM constraints (for example the sign matrix), refit/regenerate SEM artifacts before comparing J plots.
