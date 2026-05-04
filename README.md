# EEE-SD-Model-2026

SEM + CDC joint modeling pipeline with deterministic and uncertainty simulation, intervention scenarios, export, and interactive plotting.

## Core Structure

- `main.py`: entrypoint (`run_pipeline` wrapper)
- `pipeline/pipeline.py`: top-level orchestration
- `pipeline/joint_simulation.py`: deterministic + uncertainty SEM -> CDC runners
- `pipeline/results.py`: shared result dataclasses (`RunOutput`, `SimulationOutputs`, etc.)
- `config/`: run/model/intervention/plotting/data configs
- `data/`: raw-data prep and parameter loaders
- `models/sbm/`: SEM estimation + prediction
- `models/epi/`: EPI prediction
- `visualization/plotter.py`: interactive plotting

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

## Run

```bash
python main.py
```

Runtime behavior is controlled by `config/run.py` (`RunConfig`).

Primary switches:

- `execution_mode`: `"run" | "plot_only"`
- `sem_fit_mode`: `"fit_and_save" | "fit_no_save" | "load"`
- `joint_mode`: `"none" | "deterministic" | "uncertainty"`
- `scenario_mode`: `"baseline" | "intervention" | "compare"`

Intervention codebooks are in `config/interventions.py`.

## Output Artifacts

Saved under `output/`:

- `output.pkl` (SEM output)
- `joint_output.pkl` or `joint_baseline.pkl`/`joint_intervention.pkl`
- `uncertainty_output.pkl` or `uncertainty_baseline.pkl`/`uncertainty_intervention.pkl`
- `unified_outputs.csv`

## Returned Object Structure

`run_pipeline(...)` returns a dict with a structured simulation container:

- `result["simulation"].deterministic.output`
- `result["simulation"].deterministic.baseline`
- `result["simulation"].deterministic.intervention`
- `result["simulation"].uncertainty.output`
- `result["simulation"].uncertainty.baseline`
- `result["simulation"].uncertainty.intervention`

Legacy top-level keys are still included for convenience (`joint_output`, `uncertainty`, etc.).

## Plotting Controls

From `RunConfig`:

- `show_state_plots`
- `show_sem_j_violin_plots`
- `states_to_plot`
- `n_states_to_plot`

Plots are displayed interactively (not auto-saved).

## Notebook/API Migration Notes

- `config.base.BaseConfig` was replaced by `config.sem.SEMConfig`.
- `engine.*` modules were renamed to `pipeline.*`.
- Uncertainty outputs now use `UncertaintyOutput`:
  - old: `u[state_id]`
  - new: `u.results[state_id]`

## Notes

- If `execution_mode="plot_only"`, pipeline loads existing pickle outputs and does not refit SEM.
- If SEM constraints change (for example sign matrix), regenerate SEM artifacts before comparing J distributions/trajectories.
