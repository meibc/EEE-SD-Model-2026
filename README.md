# EEE-SD-Model-2026

Code for the paper's state-level model of stigma and HIV prevention among men who have sex with men (MSM)
in the United States. Survey data from the American Men's Internet Survey (AMIS, 2017-2024) and CDC
surveillance are combined in two linked models:

- **Dynamic model:** how anticipated healthcare stigma (AHS), general social and family stigma, outness,
  healthcare contact, risk behaviour, PrEP use and HIV testing evolve together in each state. It is fitted
  hierarchically (nation, Census division, state) to 500 respondent-bootstrap draws.
- **Epidemic model:** a hierarchical Bayesian model of HIV incidence and diagnoses, fitted to CDC
  estimates, that turns projected PrEP use, risk behaviour and testing into infections.

The scenarios project both models to 2036 and compare stigma-reduction scenarios with the baseline
(for example, halving AHS from its 2024 level).

## Repository layout

| Folder | Contents |
|---|---|
| `scripts/` | The analysis, in run order: `1_data/` (R), `2_evidence/` (R), `3_model/` (Python), `4_results/` (Python), `5_demographic/` (R + Python) |
| `config/` | Model structure and settings: variables, pathways and priors (CSV), fitting, projection and scenario settings, sensitivity specifications |
| `core/` | Shared helpers (logit transforms, year alignment) |
| `data/` | The model-input file (`state_inputs.py`) and per-state input containers |
| `models/` | Model equations: `dynamic/` (estimation and prediction), `epi/`, `scenarios/` (interventions, PrEP translation) |
| `pipeline/` | Fitting and projection runners, scenario runs and their summaries |
| `artifacts/` | Writer and reader of the research bundle (fitted parameters joined with the epidemic posterior) |
| `reporting/` | Paths, summaries and plot style shared by the results scripts |
| `inputs/` | Public inputs (CDC and AIDSVu files, MSM population margins); `inputs/restricted/` for AMIS (not in the repository) |

`eee-model-workflow.html` (open in a browser) is a diagram of how the scripts, modules and outputs connect.

## Requirements

- **R** 4.5 with `tidyverse`, `survey`, `srvyr`, `haven`, `readxl`, `openxlsx2`, `metafor`.
- **Python** 3.9 or later:

  ```bash
  python -m pip install -r requirements-calibration.txt   # requirements.txt plus PyMC for the epidemic model
  ```

  Tested with Python 3.9.13, NumPy 1.23, pandas 2.3, SciPy 1.12, PyMC 5.12, ArviZ 0.16. The epidemic fit
  needs a C compiler (on macOS, the Xcode command-line tools); PyTensor uses it to compile the model.

## Data

- **AMIS (restricted).** The annual restricted-use AMIS files (`sitq<year>_*.dta`, 2017-2019 and
  2021-2024) are obtained under a data use agreement and are not distributed. Place them in
  `inputs/restricted/amis_raw/`; the folder is ignored by Git. `results/` is also ignored, because
  several outputs contain respondent-level data.
- **Public inputs (included).** CDC AtlasPlus and AIDSVu files in `inputs/cdc_raw/` (sources and download
  dates in `inputs/cdc_raw/SOURCES.md`), and state and national MSM population margins in `inputs/`
  (from the MSM-Population-Estimates repository).

## Running the analysis

All commands run from the repository root.

```bash
./run_pipeline.sh data evidence model results demographic   # everything, from the raw AMIS files
./run_pipeline.sh                                           # model + results only (R outputs already current)
./run_pipeline.sh results                                   # regenerate figures and tables from existing fits
```

| Stage | What it does | Time* |
|---|---|---|
| `data` | Clean AMIS; build CDC stocks; state weights and the 500-draw respondent bootstrap (wording-adjusted and unadjusted); precision screen; reported constants; national weights | 3 min |
| `evidence` | AHS-PrEP association (the evidence target), the 2022 AHS wording change, climate associations | 20 s |
| `model` | Model-input files; epidemic fit; 10 dynamic-model fits in parallel; bundles; 13 scenario runs in parallel | 2 h |
| `results` | Paper figures and tables | 1.5 min |
| `demographic` | Weighted trajectories by demographic group | 1 min |

\*Measured on a 12-core Mac with 16 GB of memory; the dynamic-model fits take nearly all the time.

The epidemic model is refitted (about 5 minutes) whenever the `data` stage runs; otherwise its stored
posterior is reused unless `EPI_FIT=1` is set. Logs go to `results/logs/run_<date>_<time>/`.
For a quick end-to-end check, `FIT_DRAWS=2 SCENARIO_SAMPLES=20 ./run_pipeline.sh` shrinks the run; use it
only in a copy of the repository, since it overwrites `results/`.

## Outputs

Paper figures and tables are written to `results/sections/`:

| Paper item | File |
|---|---|
| Every number quoted in the text, in paper order, with its source | `paper_numbers.md` |
| Fig. 1 (observed trajectories) | `trajectories/fig1_trajectories` |
| Fig. 2 (projections) | `projections/fig2_projections` |
| Table 1, Fig. 3 (effects of reducing AHS) | `ahs_reduction/table1_pooled.csv`, `ahs_reduction/fig3_state_breakdown` |
| Sensitivity table | `sensitivity/table_sensitivity.csv` |
| Supplement: model specification, dynamic-model fit, epidemic calibration, pathways | `supplement/` |
| Supplement: state effects | `ahs_reduction/figS_state_effects`, `ahs_reduction/tableS_state_effects.csv` |
| Supplement: PrEP-AHS co-evolution | `coevolution/figS_coevolution` |
| Supplement: demographic group trajectories | `results/08_demographic/group_trajectories` |
| Supplement: reported constants (Table S4) | `results/02_inputs/reported_constants.csv` |

Intermediate outputs, numbered by stage: `00_amis/`, `00_cdc/` (cleaned data), `01_weights/`,
`02_inputs/` (bootstrap draws, evidence, model-input files, epidemic posterior), `03_fits/` (primary fit
and bundle), `04_scenarios/`, `05_sensitivity/`.

## Dashboard

The research bundle (`results/03_fits/primary/standardized_research.npz`) is also read by the
EEE-Model-Dashboard-Integration repository, which keeps its own copy of the reader. Its `sem_*` and
`cdc_*` array names are that dashboard's schema and are kept unchanged.
