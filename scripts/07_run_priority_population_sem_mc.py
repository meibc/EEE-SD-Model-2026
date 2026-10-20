#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.interventions import SCENARIO_CODEBOOK
from pipeline.priority_population_sem_mc import PriorityPopulationMCConfig, run_priority_population_mc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MC SEM-only uncertainty for national priority populations")
    parser.add_argument("--summary-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/priority_population_amis_summary_stats_v4.csv"))
    parser.add_argument("--input-base-dir", type=Path, default=Path("/Users/meibinchen/Documents/Codex/2026-08-13/d/outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/priority_population_sem_mc"))
    parser.add_argument("--variable-config-path", type=Path, default=Path("config/sem_variable_config_v3.csv"))
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-end-year", type=int, default=2036)
    parser.add_argument("--intervention-duration-steps", type=int, default=3)
    parser.add_argument("--drift-mode", choices=["zero", "none", "current"], default="zero")
    parser.add_argument("--drift-tau", type=float, default=0.20)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--scenario", dest="scenarios", action="append", choices=sorted(SCENARIO_CODEBOOK), help="Scenario code to run; repeatable. Default runs all.")
    args = parser.parse_args()

    paths = run_priority_population_mc(PriorityPopulationMCConfig(
        summary_path=args.summary_path,
        input_base_dir=args.input_base_dir,
        output_dir=args.output_dir,
        variable_config_path=args.variable_config_path,
        n_draws=args.n_draws,
        seed=args.seed,
        target_end_year=args.target_end_year,
        intervention_duration_steps=args.intervention_duration_steps,
        drift_mode=args.drift_mode,
        drift_tau=args.drift_tau,
        maxiter=args.maxiter,
        scenario_codes=args.scenarios,
    ))
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
