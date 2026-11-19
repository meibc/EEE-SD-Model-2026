#!/bin/bash
# Run the analysis from the repository root, stage by stage.
#
#   ./run_pipeline.sh                   model + results (default: the R outputs are current)
#   ./run_pipeline.sh data evidence model results demographic
#   EPI_FIT=1 ./run_pipeline.sh model   also refit the epidemic model (about 5 minutes). Without it the stored
#                                       posterior in results/02_inputs/epi_fit is reused, unless the data
#                                       stage runs: new survey inputs always refit the epidemic model.
#
# Stages (scripts/<stage>/, numbered in run order):
#   data         R: AMIS file, CDC stocks, state weights and bootstrap (wording-adjusted and unadjusted),
#                precision screen, reported constants, national weights   needs inputs/restricted/ (AMIS)
#   evidence     R: AHS-PrEP evidence, 2022 wording items, climate associations, PrEP diffusion inputs (ceilings, growth prior)
#   model        Python: model-input files, dynamic-model fits (in parallel), bundles, scenario runs (in parallel)
#   results      Python: paper figures and tables, and every number quoted in the text (results/paper/)
#   demographic  R + Python: demographic-group trajectories (results/08_demographic/; figure in results/paper/supplement/)
#
# Logs go to results/logs/run_<date>_<time>/. Settings: config/ (specifications.py, projection.py, ...).
# FIT_DRAWS and SCENARIO_SAMPLES (defaults in config/fit.py and config/projection.py) shrink the run for a quick
# end-to-end check; use them only in a copy of the repository, since the outputs overwrite results/.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STAGES=${*:-model results}
for s in $STAGES; do
  case $s in data|evidence|model|results|demographic) ;;
    *) echo "unknown stage: $s (choose from data evidence model results demographic)"; exit 1 ;;
  esac
done
LOG=results/logs/run_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"

IN=results/02_inputs
PRIMARY=results/03_fits/primary
PRIMARY_SPECS="primary_fb_neg075 primary_fb_0 primary_fb_pos15 primary_fb_pos30"   # primary model, one fit per PrEP -> AHS feedback value (main set -0.075, 0, +0.15; supplement +0.30)
SENS=results/05_sensitivity/specifications
SENS_SPECS="evidence_all_years persistence_low persistence_high no_hierarchy"         # primary_fb_0 with one change
UNADJ_SPEC=unadjusted_fb_pos45                                                        # supplement: no AHS wording adjustment
UNADJ_INPUTS=$IN/state_inputs_ahs_unadjusted.npz

has() { [[ " $STAGES " == *" $1 "* ]]; }
step() { echo "[$(date +%H:%M:%S)] $1"; }
# Run commands given as "logname|command" lines in parallel; stop if any fails.
run_parallel() {
  local pids=() names=() status=0
  while IFS='|' read -r name cmd; do
    [ -z "$name" ] && continue
    bash -c "$cmd" > "$LOG/$name.log" 2>&1 & pids+=($!); names+=("$name")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "  FAILED: ${names[$i]} (see $LOG/${names[$i]}.log)"; status=1; }
  done
  return $status
}

step "stages: $STAGES; logs in $LOG"

if has data; then
  step "data"
  Rscript scripts/1_data/01_clean_amis.R > "$LOG/data_01_clean_amis.log" 2>&1
  Rscript scripts/1_data/02_build_cdc_stocks.R > "$LOG/data_02_cdc_stocks.log" 2>&1
  Rscript scripts/1_data/03_state_weights.R > "$LOG/data_03_state_weights.log" 2>&1
  AHS_WORDING=unadjusted Rscript scripts/1_data/03_state_weights.R > "$LOG/data_03_state_weights_unadjusted.log" 2>&1
  Rscript scripts/1_data/04_state_bootstrap.R > "$LOG/data_04_bootstrap.log" 2>&1
  AHS_WORDING=unadjusted Rscript scripts/1_data/04_state_bootstrap.R > "$LOG/data_04_bootstrap_unadjusted.log" 2>&1
  Rscript scripts/1_data/05_state_precision_screen.R > "$LOG/data_05_precision_screen.log" 2>&1
  # Without --write-config: config/ priors are not overwritten (pass it by hand after reviewing the constants).
  Rscript scripts/1_data/06_reported_constants.R > "$LOG/data_06_reported_constants.log" 2>&1
  Rscript scripts/1_data/07_national_weights.R > "$LOG/data_07_national_weights.log" 2>&1
fi

if has evidence; then
  step "evidence"
  Rscript scripts/2_evidence/01_ahs_prep_evidence.R > "$LOG/evidence_01_ahs_prep.log" 2>&1
  Rscript scripts/2_evidence/02_wording_items.R > "$LOG/evidence_02_wording.log" 2>&1
  Rscript scripts/2_evidence/03_climate_associations.R > "$LOG/evidence_03_climate.log" 2>&1
  Rscript scripts/2_evidence/04_prep_diffusion.R > "$LOG/evidence_04_prep_diffusion.log" 2>&1
fi

if has model; then
  step "model: model-input files (wording-adjusted and unadjusted)"
  $PY scripts/3_model/01_build_state_inputs.py > "$LOG/model_01_inputs.log" 2>&1
  $PY scripts/3_model/01_build_state_inputs.py \
    --bootstrap-path $IN/state_bootstrap_ahs_unadjusted/state_input_draws.csv.gz \
    --data-path results/01_weights/state_model_inputs_ahs_unadjusted.xlsx \
    --output-path $UNADJ_INPUTS > "$LOG/model_01_inputs_unadjusted.log" 2>&1

  if has data && [ "${EPI_FIT:-0}" != 1 ]; then
    EPI_FIT=1
    step "model: epidemic fit runs because the data stage changed its survey inputs"
  fi
  if [ "${EPI_FIT:-0}" = 1 ]; then
    step "model: epidemic fit (about 5 minutes)"
    $PY scripts/3_model/02_fit_epi_model.py > "$LOG/model_02_epi_fit.log" 2>&1
  else
    step "model: epidemic fit skipped (stored posterior reused; EPI_FIT=1 to refit)"
  fi

  step "model: dynamic-model fits, all specifications in parallel (about 2 hours)"
  FIT="$PY scripts/3_model/03_fit_dynamic_model.py --force ${FIT_DRAWS:+--n-draws $FIT_DRAWS}"
  {
    for s in $PRIMARY_SPECS; do echo "fit_$s|$FIT --specifications $s --output-dir $PRIMARY"; done
    for s in $SENS_SPECS; do echo "fit_$s|$FIT --specifications $s --output-dir $SENS"; done
    echo "fit_$UNADJ_SPEC|$FIT --specifications $UNADJ_SPEC --output-dir $SENS --input-draws $UNADJ_INPUTS"
  } | run_parallel
  $PY - <<'EOF2'
import glob, numpy as np
bad = {f: int((~np.load(f)["success_stack"]).sum()) for f in sorted(glob.glob("results/0[35]_*/**/fits/*.npz", recursive=True))}
print("optimizer failures:", {k: v for k, v in bad.items() if v} or "none")
EOF2

  step "model: PrEP -> AHS feedback profile (chose the feedback values; config/specifications.py)"
  $PY scripts/3_model/06_profile_feedback.py > "$LOG/model_06_profile.log" 2>&1

  step "model: bundles"
  BUNDLE="$PY scripts/3_model/04_build_bundle.py"
  mkdir -p $PRIMARY/bundles $SENS/bundles
  for s in $PRIMARY_SPECS; do
    $BUNDLE --fit $PRIMARY/fits/$s.npz --output $PRIMARY/bundles/$s.npz >> "$LOG/model_04_bundle.log" 2>&1
  done
  for s in $SENS_SPECS; do
    $BUNDLE --fit $SENS/fits/$s.npz --output $SENS/bundles/$s.npz >> "$LOG/model_04_bundle.log" 2>&1
  done
  $BUNDLE --fit $SENS/fits/$UNADJ_SPEC.npz --input-draws $UNADJ_INPUTS --output $SENS/bundles/$UNADJ_SPEC.npz \
    >> "$LOG/model_04_bundle.log" 2>&1

  step "model: scenario runs in parallel"
  # Report years, samples and seed come from config/projection.py and the script's defaults.
  RUN="$PY scripts/3_model/05_run_scenarios.py --save-draw-level-endpoints ${SCENARIO_SAMPLES:+--n-samples $SCENARIO_SAMPLES}"
  SCEN="--scenario halve_ahs --scenario resilient_services --scenario prep_stall --scenario ahs_no_worsening --scenario worsening_climate --intervention-duration-steps 1"
  {
    for s in $PRIMARY_SPECS; do
      echo "scen_$s|$RUN $SCEN --standardized-input-path $PRIMARY/bundles/$s.npz --output-dir results/04_scenarios/primary/$s"
    done
    for s in $SENS_SPECS $UNADJ_SPEC; do
      echo "scen_$s|$RUN $SCEN --standardized-input-path $SENS/bundles/$s.npz --output-dir $SENS/scenarios/$s"
    done
  } | run_parallel
fi

if has results; then
  step "results"
  for f in fig1_trajectories fig2_projections table1_fig3_ahs_reduction tableS_sensitivity figS_coevolution supplement; do
    $PY scripts/4_results/$f.py > "$LOG/results_$f.log" 2>&1
  done
fi

if has demographic; then
  step "demographic"
  Rscript scripts/5_demographic/01_group_bootstrap.R > "$LOG/demographic_01_bootstrap.log" 2>&1
  $PY scripts/5_demographic/02_demographic_descriptives.py > "$LOG/demographic_02_descriptives.log" 2>&1
fi

if has results || has demographic; then
  # Last, because it reads the demographic outputs as well as the results.
  step "paper numbers"
  $PY scripts/4_results/paper_numbers.py > "$LOG/results_paper_numbers.log" 2>&1
fi

step "done"
