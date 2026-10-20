#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.priority_population_sem import (
    PriorityPopulationSEMConfig,
    run_priority_population_drift_sensitivity,
    run_priority_population_sem,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEM-only fit/scenarios for a national priority population")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing *_trends/cov_matrix/structural_priors CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("output/priority_population_sem"))
    parser.add_argument("--variable-config-path", type=Path, default=Path("config/sem_variable_config_v3.csv"))
    parser.add_argument("--target-end-year", type=int, default=2036)
    parser.add_argument("--intervention-duration-steps", type=int, default=3)
    parser.add_argument("--drift-mode", choices=["zero", "none", "current"], default="zero")
    parser.add_argument("--drift-tau", type=float, default=0.20, help="Tau for zero-centered enabled drift terms")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--init-j-path", type=Path, default=None, help="Optional NPZ or CSV pathway matrix used only as optimizer initialization")
    parser.add_argument("--init-j-mode", choices=["none", "state_median", "national"], default="none")
    parser.add_argument("--sensitivity", action="store_true", help="Run none, zero-tau sweep, and current drift comparison")
    parser.add_argument("--sensitivity-taus", type=float, nargs="*", default=[0.10, 0.20, 0.35])
    args = parser.parse_args()

    config = PriorityPopulationSEMConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        variable_config_path=args.variable_config_path,
        target_end_year=args.target_end_year,
        intervention_duration_steps=args.intervention_duration_steps,
        drift_mode=args.drift_mode,
        drift_tau=args.drift_tau,
        maxiter=args.maxiter,
        make_plots=not args.no_plots,
        init_j_path=args.init_j_path,
        init_j_mode=args.init_j_mode,
    )
    if args.sensitivity:
        path = run_priority_population_drift_sensitivity(config, zero_taus=args.sensitivity_taus)
        print(f"Wrote drift sensitivity summary: {path}")
    else:
        paths = run_priority_population_sem(config)
        for label, path in paths.items():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
