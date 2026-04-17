"""Top-level run/pipeline options."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from config.joint import JointConfig


@dataclass
class RunConfig:
    execution_mode: Literal["run", "plot_only"] = "run"

    # SEM
    sem_fit_mode: Literal["fit_and_save", "fit_no_save", "load"] = "fit_and_save"
    run_predict: bool = True
    output_dir: Path = Path("output")
    sem_pickle_name: str = "output.pkl"

    # Joint / CDC
    joint: JointConfig = field(default_factory=JointConfig)
    joint_mode: Literal["none", "deterministic", "uncertainty"] = "uncertainty"
    scenario_mode: Literal["baseline", "intervention", "compare"] = "compare"
    cdc_posterior_path: Path = Path("cdc_posteriors.nc")
    cdc_trans_path: Path = Path("trans_results.npz")
    sem_params_path: Path = Path("sem_mc_samples_v2.npz")
    n_uncertainty_samples: int = 1000
    seed: int = 123
    show_progress: bool = True
    state_intervention_codes: list[str] = field(default_factory=lambda: ["reduce_ahs"])
    relationship_intervention_codes: list[str] = field(default_factory=list)
    intervention_duration_steps: int = 1

    # Forecasting
    target_end_year: int = 2036

    # Export
    export_unified_csv: bool = True
    unified_csv_name: str = "unified_outputs.csv"

    # Visualization
    show_state_plots: bool = True
    show_sem_j_violin_plots: bool = True
    n_states_to_plot: int = 2
    states_to_plot: list[str] = field(
        default_factory=lambda: ["CA", "TX"]
    )

    def validate(self) -> None:
        if self.execution_mode not in {"run", "plot_only"}:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}")
        if self.sem_fit_mode not in {"fit_and_save", "fit_no_save", "load"}:
            raise ValueError(f"Invalid sem_fit_mode: {self.sem_fit_mode}")
        if self.joint_mode not in {"none", "deterministic", "uncertainty"}:
            raise ValueError(f"Invalid joint_mode: {self.joint_mode}")
        if self.scenario_mode not in {"baseline", "intervention", "compare"}:
            raise ValueError(f"Invalid scenario_mode: {self.scenario_mode}")
        if self.execution_mode == "run" and self.joint_mode == "none":
            raise ValueError("joint_mode='none' is only useful with execution_mode='plot_only'.")
        if self.scenario_mode in {"intervention", "compare"}:
            if not (self.state_intervention_codes or self.relationship_intervention_codes):
                raise ValueError(
                    "scenario_mode requires at least one intervention code "
                    "(state_intervention_codes or relationship_intervention_codes)."
                )
