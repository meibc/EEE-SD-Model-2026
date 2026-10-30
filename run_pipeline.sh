#!/bin/bash
# Run the analysis from the repository root, stage by stage.
#
#   ./run_pipeline.sh                   model + results (default: the R outputs are current)
#   ./run_pipeline.sh data evidence model results demographic
#   EPI_FIT=1 ./run_pipeline.sh model   also refit the epidemic model (about 3 hours). Without it the stored
#                                       posterior in results/02_inputs/epi_fit is reused, unless the data
#                                       stage runs: new survey inputs always refit the epidemic model.
#
# Stages (scripts/<stage>/, numbered in run order):
#   data         R: AMIS file, CDC stocks, state weights and bootstrap (wording-adjusted and unadjusted),
#                precision screen, reported constants, national weights   needs inputs/restricted/ (AMIS)
#   evidence     R: AHS-PrEP evidence, 2022 wording items, climate associations
#   model        Python: model-input files, dynamic-model fits (in parallel), bundles, scenario runs (in parallel)
#   results      Python: paper figures and tables (results/sections/)
#   demographic  R + Python: demographic-group associations and descriptives (results/08_demographic/)
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
SPEC=results/05_sensitivity/model_specification
UNADJ=results/05_sensitivity/ahs_wording_unadjusted
SENS=results/05_sensitivity
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
    step "model: epidemic fit (about 3 hours)"
    $PY scripts/3_model/02_fit_epi_model.py > "$LOG/model_02_epi_fit.log" 2>&1
  else
    step "model: epidemic fit skipped (stored posterior reused; EPI_FIT=1 to refit)"
  fi

  step "model: dynamic-model fits, all specifications in parallel (about 3-4 hours)"
  FIT="$PY scripts/3_model/03_fit_dynamic_model.py --force ${FIT_DRAWS:+--n-draws $FIT_DRAWS}"
  {
    echo "fit_primary|$FIT --specifications primary"
    for s in evidence_all_years persistence_low persistence_high placebo_ahs \
             trajectory_only covariance_balanced no_hierarchy zero_prep_drift; do
      echo "fit_$s|$FIT --specifications $s --output-dir $SPEC"
    done
    echo "fit_unadjusted|$FIT --specifications primary --output-dir $UNADJ --input-draws $UNADJ_INPUTS --evidence-path $IN/ahs_prep_evidence_all_years.csv"
  } | run_parallel
  $PY - <<'EOF'
import glob, numpy as np
bad = {f: int((~np.load(f)["success_stack"]).sum()) for f in sorted(glob.glob("results/0[35]_*/**/fits/*.npz", recursive=True))}
print("optimizer failures:", {k: v for k, v in bad.items() if v} or "none")
EOF

  step "model: bundles"
  BUNDLE="$PY scripts/3_model/04_build_bundle.py"
  $BUNDLE > "$LOG/model_04_bundle.log" 2>&1
  mkdir -p $SPEC/bundles
  for s in evidence_all_years persistence_low persistence_high placebo_ahs; do
    $BUNDLE --fit $SPEC/fits/$s.npz --output $SPEC/bundles/$s.npz >> "$LOG/model_04_bundle.log" 2>&1
  done
  $BUNDLE --fit $UNADJ/fits/primary.npz --input-draws $UNADJ_INPUTS --output $UNADJ/standardized_research.npz \
    >> "$LOG/model_04_bundle.log" 2>&1

  step "model: scenario runs in parallel"
  # Report years, samples, seed and the primary PrEP translation come from config/projection.py and the script's defaults.
  RUN="$PY scripts/3_model/05_run_scenarios.py --save-draw-level-endpoints ${SCENARIO_SAMPLES:+--n-samples $SCENARIO_SAMPLES}"
  ALL="--scenario halve_ahs --scenario ahs_to_benchmark --scenario resilient_prep_delivery --scenario halve_upstream_stigma --scenario ahs_no_worsening"
  OFF="--zero-projection-edge prep_used:stigma_ahs"
  {
    echo "scen_continued|$RUN $ALL --output-dir results/04_scenarios/paper_static"
    echo "scen_not_continued|$RUN $ALL $OFF --output-dir $SENS/static_no_prep_ahs_feedback"
    echo "scen_placebo_continued|$RUN --scenario halve_ahs --standardized-input-path $SPEC/bundles/placebo_ahs.npz --output-dir $SENS/static_placebo_ahs/feedback_on"
    echo "scen_placebo_not_continued|$RUN --scenario halve_ahs $OFF --standardized-input-path $SPEC/bundles/placebo_ahs.npz --output-dir $SENS/static_placebo_ahs/feedback_off"
    echo "scen_unadjusted|$RUN $ALL --standardized-input-path $UNADJ/standardized_research.npz --output-dir $UNADJ/scenarios_static"
    # placebo_ahs runs above (static_placebo_ahs), with and without the feedback.
    for s in evidence_all_years persistence_low persistence_high; do
      echo "scen_spec_$s|$RUN --scenario halve_ahs --standardized-input-path $SPEC/bundles/$s.npz --output-dir $SENS/static_specifications/$s"
    done
    for pr in 0.80 0.70; do
      echo "scen_contextual_${pr}_continued|$RUN --contextual-prep-pr $pr --scenario halve_ahs --output-dir $SENS/contextual_static_pr${pr/./}/continued"
      echo "scen_contextual_${pr}_not_continued|$RUN --contextual-prep-pr $pr --scenario halve_ahs $OFF --output-dir $SENS/contextual_static_pr${pr/./}/not_continued"
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
  Rscript scripts/5_demographic/01_group_associations.R > "$LOG/demographic_01_associations.log" 2>&1
  Rscript scripts/5_demographic/02_group_bootstrap.R > "$LOG/demographic_02_bootstrap.log" 2>&1
  $PY scripts/5_demographic/03_demographic_descriptives.py > "$LOG/demographic_03_descriptives.log" 2>&1
fi

step "done"
