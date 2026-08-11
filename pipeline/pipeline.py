"""Pipeline orchestration."""

from config.sem import SEMConfig
from config.optimization import OptimConfig
from config.shrinkage import ShrinkageConfig
from config.run import RunConfig
from models.sbm.runner import SBRunner
from pipeline.results import (
    RunOutput,
    DeterministicScenarios,
    UncertaintyScenarios,
    SimulationOutputs,
)
from data.unit import Unit
from data.params_cdc import CDCParamsLoader
from data.params_sem import SEMParamsLoader
from pipeline.loaders import save, load, load_fit_results
from pipeline.export import export_unified_table
from pipeline.joint_simulation import run_joint, run_uncertainty
from pipeline.intervention_compare import (
    build_deterministic_year10_tables,
    build_uncertainty_year10_tables,
    save_year10_tables,
)
from models.shared.alignment import build_model_years
from config.interventions import SCENARIO_CODEBOOK
from visualization.plotter import (
    plot_deterministic_comparison,
    plot_sem_loss_history,
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
    joint_interventions_pickle_path = options.output_dir / "joint_interventions.pkl"
    uncertainty_interventions_pickle_path = options.output_dir / "uncertainty_interventions.pkl"

    if options.execution_mode == "plot_only":
        output = load(sem_pickle_path)
        deterministic = DeterministicScenarios(
            output=load(joint_pickle_path) if joint_pickle_path.exists() else None,
            baseline=load(joint_baseline_pickle_path) if joint_baseline_pickle_path.exists() else None,
            intervention=load(joint_intervention_pickle_path) if joint_intervention_pickle_path.exists() else None,
            interventions=(
                load(joint_interventions_pickle_path)
                if joint_interventions_pickle_path.exists()
                else None
            ),
        )
        uncertainty = UncertaintyScenarios(
            output=load(uncertainty_pickle_path) if uncertainty_pickle_path.exists() else None,
            baseline=(
                load(uncertainty_baseline_pickle_path)
                if uncertainty_baseline_pickle_path.exists()
                else None
            ),
            intervention=(
                load(uncertainty_intervention_pickle_path)
                if uncertainty_intervention_pickle_path.exists()
                else None
            ),
            interventions=(
                load(uncertainty_interventions_pickle_path)
                if uncertainty_interventions_pickle_path.exists()
                else None
            ),
        )
        simulation = SimulationOutputs(
            deterministic=deterministic,
            uncertainty=uncertainty,
        )
        sem_loader = SEMParamsLoader(options.sem_params_path) if options.show_sem_j_violin_plots else None

        if options.show_state_plots or options.show_sem_j_violin_plots or options.show_sem_loss_plots:
            _show_plots(
                output,
                simulation,
                sem_loader,
                options,
            )

        return {
            "sem_output": output,
            "simulation": simulation,
            "joint_output": simulation.deterministic.output,
            "uncertainty": simulation.uncertainty.output,
            "uncertainty_baseline": simulation.uncertainty.baseline,
            "uncertainty_intervention": simulation.uncertainty.intervention,
            "uncertainty_interventions": simulation.uncertainty.interventions,
            "joint_baseline": simulation.deterministic.baseline,
            "joint_intervention": simulation.deterministic.intervention,
            "joint_interventions": simulation.deterministic.interventions,
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
    target_end_year = (
        int(options.target_end_year)
        if options.target_end_year is not None
        else int(cdc_loader.years[-1]) + int(options.forecast_years_ahead)
    )
    model_years = build_model_years(cdc_loader.years, target_end_year)

    simulation = SimulationOutputs(
        deterministic=DeterministicScenarios(),
        uncertainty=UncertaintyScenarios(),
    )
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
        state_codes = []
        rel_codes = []
        run_compare = True

    if options.joint_mode == "deterministic":
        if run_compare:
            simulation.deterministic.baseline = run_joint(
                output,
                cdc_loader,
                units,
                unit_ids=all_unit_ids,
                model_years=model_years,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
                risk_var=options.joint.risk_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=[],
                relationship_intervention_codes=[],
                intervention_duration_steps=options.intervention_duration_steps,
            )
            scenario_outputs: dict[str, object] = {}
            for scenario_code in options.intervention_scenario_codes:
                spec = SCENARIO_CODEBOOK[scenario_code]
                scenario_outputs[scenario_code] = run_joint(
                    output,
                    cdc_loader,
                    units,
                    unit_ids=all_unit_ids,
                    model_years=model_years,
                    hivtest_var=options.joint.hivtest_var,
                    prep_var=options.joint.prep_var,
                    risk_var=options.joint.risk_var,
                    n_elig_var=options.joint.n_elig_var,
                    state_intervention_codes=spec.get("state_codes", []),
                    relationship_intervention_codes=spec.get("relationship_codes", []),
                    intervention_duration_steps=options.intervention_duration_steps,
                )
            simulation.deterministic.interventions = scenario_outputs
            first_key = options.intervention_scenario_codes[0]
            simulation.deterministic.intervention = scenario_outputs[first_key]
            simulation.deterministic.output = simulation.deterministic.intervention
            save(simulation.deterministic.baseline, joint_baseline_pickle_path)
            save(simulation.deterministic.intervention, joint_intervention_pickle_path)
            save(simulation.deterministic.interventions, joint_interventions_pickle_path)
            print(
                "Saved baseline and multi-scenario deterministic outputs: "
                f"{len(simulation.deterministic.interventions)} scenarios"
            )
        else:
            simulation.deterministic.output = run_joint(
                output,
                cdc_loader,
                units,
                unit_ids=all_unit_ids,
                model_years=model_years,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
                risk_var=options.joint.risk_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=state_codes,
                relationship_intervention_codes=rel_codes,
                intervention_duration_steps=options.intervention_duration_steps,
            )
            save(simulation.deterministic.output, joint_pickle_path)
            print(f"Saved deterministic joint output for {len(simulation.deterministic.output.results)} units")

    if options.joint_mode == "uncertainty":
        sem_loader = SEMParamsLoader(options.sem_params_path)
        if run_compare:
            simulation.uncertainty.baseline = run_uncertainty(
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
                risk_var=options.joint.risk_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=[],
                relationship_intervention_codes=[],
                intervention_duration_steps=options.intervention_duration_steps,
            )
            if options.intervention_scenario_codes:
                scenario_unc_outputs: dict[str, object] = {}
                for scenario_code in options.intervention_scenario_codes:
                    spec = SCENARIO_CODEBOOK[scenario_code]
                    scenario_unc_outputs[scenario_code] = run_uncertainty(
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
                        risk_var=options.joint.risk_var,
                        n_elig_var=options.joint.n_elig_var,
                        state_intervention_codes=spec.get("state_codes", []),
                        relationship_intervention_codes=spec.get("relationship_codes", []),
                        intervention_duration_steps=options.intervention_duration_steps,
                    )
                simulation.uncertainty.interventions = scenario_unc_outputs
                first_key = options.intervention_scenario_codes[0]
                simulation.uncertainty.intervention = scenario_unc_outputs[first_key]
            else:
                simulation.uncertainty.intervention = run_uncertainty(
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
                    risk_var=options.joint.risk_var,
                    n_elig_var=options.joint.n_elig_var,
                    state_intervention_codes=state_codes,
                    relationship_intervention_codes=rel_codes,
                    intervention_duration_steps=options.intervention_duration_steps,
                )
            save(simulation.uncertainty.baseline, uncertainty_baseline_pickle_path)
            save(simulation.uncertainty.intervention, uncertainty_intervention_pickle_path)
            if simulation.uncertainty.interventions is not None:
                save(simulation.uncertainty.interventions, uncertainty_interventions_pickle_path)
            simulation.uncertainty.output = simulation.uncertainty.intervention
            print(
                "Saved baseline and intervention uncertainty outputs for "
                f"{len(simulation.uncertainty.output.results)} units"
            )
        else:
            simulation.uncertainty.output = run_uncertainty(
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
                risk_var=options.joint.risk_var,
                n_elig_var=options.joint.n_elig_var,
                state_intervention_codes=state_codes,
                relationship_intervention_codes=rel_codes,
                intervention_duration_steps=options.intervention_duration_steps,
            )
            save(simulation.uncertainty.output, uncertainty_pickle_path)
            print(f"Saved uncertainty output for {len(simulation.uncertainty.output.results)} units")

    if options.export_unified_csv:
        export_path = options.output_dir / options.unified_csv_name
        df = export_unified_table(
            export_path,
            sem_output=output,
            joint_output=simulation.deterministic.output,
            uncertainty=simulation.uncertainty.output,
        )
        print(f"Exported unified table: {export_path} ({len(df)} rows)")

    if options.export_intervention_year10_csv and run_compare:
        if (
            simulation.deterministic.baseline is not None
            and simulation.deterministic.interventions is not None
        ):
            sem_df, epi_df = build_deterministic_year10_tables(
                sem_output=output,
                baseline=simulation.deterministic.baseline,
                interventions=simulation.deterministic.interventions,
                target_year=int(model_years[-1]),
            )
            sem_out = options.output_dir / options.intervention_year10_sem_csv
            epi_out = options.output_dir / options.intervention_year10_epi_csv
            save_year10_tables(sem_df, epi_df, sem_out, epi_out)
            print(f"Exported deterministic year-10 intervention tables: {sem_out}, {epi_out}")

        if simulation.uncertainty.baseline is not None and simulation.uncertainty.interventions is not None:
            sem_df_u, epi_df_u = build_uncertainty_year10_tables(
                sem_output=output,
                baseline=simulation.uncertainty.baseline,
                interventions=simulation.uncertainty.interventions,
                target_year=int(model_years[-1]),
            )
            sem_out_u = options.output_dir / f"uncertainty_{options.intervention_year10_sem_csv}"
            epi_out_u = options.output_dir / f"uncertainty_{options.intervention_year10_epi_csv}"
            save_year10_tables(sem_df_u, epi_df_u, sem_out_u, epi_out_u)
            print(f"Exported uncertainty median year-10 intervention tables: {sem_out_u}, {epi_out_u}")

    if options.show_state_plots:
        _show_plots(
            output,
            simulation,
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
    elif options.show_sem_loss_plots:
        plotted_loss = plot_sem_loss_history(
            sem_output=output,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
        )
        print(f"Displayed SEM loss plots for states: {plotted_loss}")

    return {
        "sem_output": output,
        "simulation": simulation,
        "joint_output": simulation.deterministic.output,
        "uncertainty": simulation.uncertainty.output,
        "uncertainty_baseline": simulation.uncertainty.baseline,
        "uncertainty_intervention": simulation.uncertainty.intervention,
        "uncertainty_interventions": simulation.uncertainty.interventions,
        "joint_baseline": simulation.deterministic.baseline,
        "joint_intervention": simulation.deterministic.intervention,
        "joint_interventions": simulation.deterministic.interventions,
    }


def _show_plots(
    output: RunOutput,
    simulation: SimulationOutputs,
    sem_loader,
    options: RunConfig,
) -> None:
    """Display plots."""
    if simulation.uncertainty.baseline is not None and simulation.uncertainty.intervention is not None:
        plotted = plot_uncertainty_comparison(
            sem_output=output,
            baseline=simulation.uncertainty.baseline,
            intervention=simulation.uncertainty.intervention,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
        )
        print(f"Displayed baseline vs intervention plots for states: {plotted}")
    elif simulation.uncertainty.output is not None:
        try:
            plotted = plot_state_uncertainty_outputs(
                sem_output=output,
                uncertainty=simulation.uncertainty.output,
                state_ids=options.states_to_plot,
                max_states=options.n_states_to_plot,
                hivtest_var=options.joint.hivtest_var,
                prep_var=options.joint.prep_var,
            )
            print(f"Displayed uncertainty plots for states: {plotted}")
        except ValueError as exc:
            print(f"Skipping uncertainty state plots: {exc}")
    elif simulation.deterministic.baseline is not None and simulation.deterministic.intervention is not None:
        plotted = plot_deterministic_comparison(
            sem_output=output,
            baseline=simulation.deterministic.baseline,
            intervention=simulation.deterministic.intervention,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
        )
        print(f"Displayed deterministic baseline vs intervention plots for states: {plotted}")
    elif simulation.deterministic.output is not None:
        plotted = plot_state_outputs(
            sem_output=output,
            joint_output=simulation.deterministic.output,
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

    if options.show_sem_loss_plots:
        plotted_loss = plot_sem_loss_history(
            sem_output=output,
            state_ids=options.states_to_plot,
            max_states=options.n_states_to_plot,
        )
        print(f"Displayed SEM loss plots for states: {plotted_loss}")
