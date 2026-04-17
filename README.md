# EEE-SD-Model-2026

SEM + CDC joint modeling pipeline with deterministic and uncertainty runs, optional interventions, and plotting.

## Project Layout

- `main.py`: thin entrypoint that calls pipeline runner.
- `pipeline/pipeline.py`: run orchestration and mode branching.
- `config/`: model and optimization config dataclasses.
- `data/`: data loaders, unit builders, SEM/CDC parameter loaders.
- `models/sbm/`: SEM estimation and prediction.
- `models/epi/`: CDC/EPI prediction.
- `pipeline/joint_simulation.py`: SEM -> CDC deterministic + uncertainty connectors.
- `visualization/plotter.py`: plotting helpers.
- `output/`: saved run artifacts.

## Requirements

- Python 3.9+ (you are currently using `.venv313`, which is fine).
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Files Expected At Repo Root

- `Factor Analysis Final.xlsx`
- `cdc_posteriors.nc`
- `trans_results.npz`
- `sem_mc_samples_v2.npz` (or `sem_mc_results.npz` if you switch `sem_params_path`)

## Run

Default run:

```bash
python main.py
```

This uses `RunConfig` defaults defined in `config/run.py`.

## Common Run Modes (via `RunConfig` in `config/run.py`)

- Deterministic only:
  - `execution_mode="run"`
  - `joint_mode="deterministic"`

- Uncertainty only:
  - `execution_mode="run"`
  - `joint_mode="uncertainty"`

- Baseline only:
  - `scenario_mode="baseline"`

- Intervention only:
  - `scenario_mode="intervention"`
  - set one or both:
    - `state_intervention_codes`
    - `relationship_intervention_codes`

- Baseline vs intervention comparison:
  - `scenario_mode="compare"`
  - set one or both:
    - `state_intervention_codes`
    - `relationship_intervention_codes`

- Plot-only (reuse saved outputs):
  - `execution_mode="plot_only"`

## Intervention Controls

Configured in `RunConfig` (`config/run.py`):

- `execution_mode: "run" | "plot_only"`
- `sem_fit_mode: "fit_and_save" | "fit_no_save" | "load"`
- `joint_mode: "none" | "deterministic" | "uncertainty"`
- `scenario_mode: "baseline" | "intervention" | "compare"`
- `state_intervention_codes: list[str]`
- `relationship_intervention_codes: list[str]`
- `intervention_duration_steps: int`

Intervention definitions live in:

- `models/shared/intervention.py`

## Outputs

Saved under `output/`:

- `output.pkl` (SEM output object)
- `joint_output.pkl` (single deterministic run)
- `joint_baseline.pkl`, `joint_intervention.pkl` (deterministic comparison)
- `uncertainty_output.pkl` (single uncertainty run)
- `uncertainty_baseline.pkl`, `uncertainty_intervention.pkl` (uncertainty comparison)
- `unified_outputs.csv` (if export enabled)

## Plotting

Controlled by:

- `show_state_plots`
- `show_sem_j_violin_plots`
- `states_to_plot`
- `n_states_to_plot`

Plots are shown interactively (not automatically saved).

## Notes

- If you change SEM constraints (for example sign matrix), refit/regenerate SEM artifacts before expecting changed `J` plots.
- If plots do not appear, check your matplotlib backend in your IDE/terminal.
