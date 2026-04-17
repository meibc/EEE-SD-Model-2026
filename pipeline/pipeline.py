"""Pipeline orchestration."""

from config.sem import SEMConfig
from config.optimization import OptimConfig
from config.shrinkage import ShrinkageConfig
from config.run import RunConfig
from models.sbm.runner import SBRunner
from pipeline.results import RunOutput
from data.unit import Unit
from data.params_cdc import CDCParamsLoader
from data.params_sem import SEMParamsLoader
from pipeline.loaders import save, load, load_fit_results
from pipeline.export import export_unified_table
from pipeline.joint_simulation import run_joint, run_uncertainty
from models.shared.alignment import build_model_years
from visualization.plotter import (
    plot_deterministic_comparison,
    plot_state_outputs,
    plot_uncertainty_comparison,
    plot_state_uncertainty_outputs,
    plot_sem_j_violin,
)


def run_pipeline(options: RunConfig | None = None) -> dict:
    """Run full pipeline."""
    if options is None:
        options = RunConfig()
    options.validate()

    options.output_dir.mkdir(exist_ok=True)

    sem_pickle_path = options.output_dir / options.sem_pickle_name
    joint_pickle_path = options.output_dir / "joint_output.pkl"
    joint_baseline_pickle_path = options.output_dir / "joint_baseline.pkl"
    joint_intervention_pickle_path = options.output_dir / "joint_intervention.pkl"
    uncertainty_pickle_path = options.output_dir / "uncertainty_output.pkl"
    uncertainty_baseline_pickle_path = options.output_dir / "uncertainty_baseline.pkl"
    uncertainty_intervention_pickle_path = options.output_dir / "uncertainty_intervention.pkl"

    if options.execution_mode == "plot_only":
        output = load(sem_pickle_path)
        joint_output = load(joint_pickle_path) if joint_pickle_path.exists() else None
        joint_baseline = load(joint_baseline_pickle_path) if joint_baseline_pickle_path.exists() else None
        joint_intervention = load(joint_intervention_pickle_path) if joint_intervention_pickle_path.exists() else None
        uncertainty = load(uncertainty_pickle_path) if uncertainty_pickle_path.exists() else None
        sem_loader = SEMParamsLoader(options.sem_params_path) if options.show_sem_j_violin_plots else None
        uncertainty_baseline = (
            load(uncertainty_baseline_pickle_path)
            if uncertainty_baseline_pickle_path.exists()
            else None
        )
        uncertainty_intervention = (
            load(uncertainty_intervention_pickle_path)
            if uncertainty_intervention_pickle_path.exists()
            else None
        )

        if options.show_state_plots or options.show_sem_j_violin_plots:
            _show_plots(
                output,
                joint_output,
                uncertainty,
                uncertainty_baseline,
                uncertainty_intervention,
                joint_baseline,
                joint_intervention,
                sem_loader,
                options,
            )

        return {
            "sem_output": output,
            "joint_output": joint_output,
            "uncertainty": uncertainty,
            "uncertainty_baseline": uncertainty_baseline,
            "uncertainty_intervention": uncertainty_intervention,
            "joint_baseline": joint_baseline,
            "joint_intervention": joint_intervention,
        }

    base = SEMConfig()
    opt = OptimConfig()
    shrink = ShrinkageConfig()
    runner = SBRunner(base, opt, shrink)

    if options.sem_fit_mode == "load":
        fit_results = load_fit_results(sem_pickle_path)
        output = runner.run(fit=False, predict=options.run_predict, fit_results=fit_results)
    else:
        output = runner.run(fit=True, predict=options.run_predict)

    if options.sem_fit_mode == "fit_and_save":
        save(output, sem_pickle_path)
        print(f"Saved SEM output to {sem_pickle_path}")

    units = Unit.to_dict(output.inputs.units)
    cdc_loader = CDCParamsLoader(options.cdc_posterior_path, options.cdc_trans_path)
    all_unit_ids = sorted(set(units.keys()) & set(cdc_loader.geo_names))
    model_years = build_model_years(cdc_loader.years, options.target_end_year)

    joint_output = None
    joint_baseline = None
    joint_intervention = None
    uncertainty = None
    uncertainty_baseline = None
    uncertainty_intervention = None
    sem_loader = None

    if options.scenario_mode == "baseline":
        state_codes = []
        rel_codes = []
        run_compare = False
    elif options.scenario_mode == "intervention":
        state_codes = options.state_intervention_codes
        rel_codes = options.relationship_intervention_codes
        run_compare = False
    else:
        state_codes = options.state_intervention_codes
        rel_codes = options.relationship_intervention_codes
        run_compare = True

    if options.joint_mode == "deterministic":
        has_interventions = bool(state_codes or rel_codes)
        if run_compare and has_interventions:
            joint_baseline = run_joint(
                output,
                cdc_loader,
                units,
                unit_ids=all_unit_ids,
                model_years=model_years,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=[],
                relationship_intervention_codes=[],
                intervention_duration_steps=options.intervention_duration_steps,
            )
            joint_intervention = run_joint(
                output,
                cdc_loader,
                units,
                unit_ids=all_unit_ids,
                model_years=model_years,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=state_codes,
                relationship_intervention_codes=rel_codes,
                intervention_duration_steps=options.intervention_duration_steps,
            )
            save(joint_baseline, joint_baseline_pickle_path)
            save(joint_intervention, joint_intervention_pickle_path)
            joint_output = joint_intervention
            print(f"Saved baseline and intervention deterministic outputs for {len(joint_output.results)} units")
        else:
            joint_output = run_joint(
                output,
                cdc_loader,
                units,
                unit_ids=all_unit_ids,
                model_years=model_years,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=state_codes,
                relationship_intervention_codes=rel_codes,
                intervention_duration_steps=options.intervention_duration_steps,
            )
            save(joint_output, joint_pickle_path)
            print(f"Saved deterministic joint output for {len(joint_output.results)} units")

    if options.joint_mode == "uncertainty":
        sem_loader = SEMParamsLoader(options.sem_params_path)
        if run_compare:
            uncertainty_baseline = run_uncertainty(
                sem_loader,
                cdc_loader,
                units,
                unit_ids=all_unit_ids,
                n_samples=options.n_uncertainty_samples,
                seed=options.seed,
                show_progress=options.show_progress,
                model_years=model_years,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=[],
                relationship_intervention_codes=[],
                intervention_duration_steps=options.intervention_duration_steps,
            )
            uncertainty_intervention = run_uncertainty(
                sem_loader,
                cdc_loader,
                units,
                unit_ids=all_unit_ids,
                n_samples=options.n_uncertainty_samples,
                seed=options.seed,
                show_progress=options.show_progress,
                model_years=model_years,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=state_codes,
                relationship_intervention_codes=rel_codes,
                intervention_duration_steps=options.intervention_duration_steps,
            )
            save(uncertainty_baseline, uncertainty_baseline_pickle_path)
            save(uncertainty_intervention, uncertainty_intervention_pickle_path)
            uncertainty = uncertainty_intervention
            print(f"Saved baseline and intervention uncertainty outputs for {len(uncertainty)} units")
        else:
            uncertainty = run_uncertainty(
                sem_loader,
                cdc_loader,
                units,
                unit_ids=all_unit_ids,
                n_samples=options.n_uncertainty_samples,
                seed=options.seed,
                show_progress=options.show_progress,
                model_years=model_years,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=state_codes,
                relationship_intervention_codes=rel_codes,
                intervention_duration_steps=options.intervention_duration_steps,
            )
            save(uncertainty, uncertainty_pickle_path)
            print(f"Saved uncertainty output for {len(uncertainty)} units")

    if options.export_unified_csv:
        export_path = options.output_dir / options.unified_csv_name
        df = export_unified_table(
            export_path,
            sem_output=output,
            joint_output=joint_output,
            uncertainty=uncertainty,
        )
        print(f"Exported unified table: {export_path} ({len(df)} rows)")

    if options.show_state_plots:
        _show_plots(
            output,
            joint_output,
            uncertainty,
            uncertainty_baseline,
            uncertainty_intervention,
            joint_baseline,
            joint_intervention,
            sem_loader,
            options,
        )
    elif options.show_sem_j_violin_plots:
        if sem_loader is None:
            sem_loader = SEMParamsLoader(options.sem_params_path)
        plotted_j = plot_sem_j_violin(
            sem_loader=sem_loader,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
        )
        print(f"Displayed SEM J violin plots for states: {plotted_j}")

    return {
        "sem_output": output,
        "joint_output": joint_output,
        "uncertainty": uncertainty,
        "uncertainty_baseline": uncertainty_baseline,
        "uncertainty_intervention": uncertainty_intervention,
        "joint_baseline": joint_baseline,
        "joint_intervention": joint_intervention,
    }


def _show_plots(
    output: RunOutput,
    joint_output,
    uncertainty,
    uncertainty_baseline,
    uncertainty_intervention,
    joint_baseline,
    joint_intervention,
    sem_loader,
    options: RunConfig,
) -> None:
    """Display plots."""
    if uncertainty_baseline is not None and uncertainty_intervention is not None:
        plotted = plot_uncertainty_comparison(
            sem_output=output,
            baseline=uncertainty_baseline,
            intervention=uncertainty_intervention,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
        )
        print(f"Displayed baseline vs intervention plots for states: {plotted}")
    elif uncertainty is not None:
        plotted = plot_state_uncertainty_outputs(
            sem_output=output,
            uncertainty=uncertainty,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
            hivtest_var=options.joint.hivtest_var,
            prep_var=options.joint.prep_var,
        )
        print(f"Displayed uncertainty plots for states: {plotted}")
    elif joint_baseline is not None and joint_intervention is not None:
        plotted = plot_deterministic_comparison(
            sem_output=output,
            baseline=joint_baseline,
            intervention=joint_intervention,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
        )
        print(f"Displayed deterministic baseline vs intervention plots for states: {plotted}")
    elif joint_output is not None:
        plotted = plot_state_outputs(
            sem_output=output,
            joint_output=joint_output,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
            hivtest_var=options.joint.hivtest_var,
            prep_var=options.joint.prep_var,
        )
        print(f"Displayed plots for states: {plotted}")
    else:
        print("Skipping plots: set joint_mode to 'deterministic' or 'uncertainty'.")

    if options.show_sem_j_violin_plots:
        if sem_loader is None:
            sem_loader = SEMParamsLoader(options.sem_params_path)
        plotted_j = plot_sem_j_violin(
            sem_loader=sem_loader,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
        )
        print(f"Displayed SEM J violin plots for states: {plotted_j}")
