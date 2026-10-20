#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.interventions import SCENARIO_CODEBOOK, PRIMARY_SCENARIOS
from pipeline.sem_scenarios import SEMScenarioRunConfig, run_sem_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SEM/CDC scenarios from the current research SEM/CDC bundle")
    parser.add_argument("--standardized-input-path", type=Path, default=REPO_ROOT / "output/current/standardized_research.npz")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/current/scenarios")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-end-year", type=int, default=2036)
    parser.add_argument(
        "--intervention-duration-steps",
        type=int,
        default=3,
        help="Number of 2-year SEM steps over which interventions ramp to full strength; 3 = 6 years.",
    )
    parser.add_argument("--scenario", dest="scenarios", action="append", choices=sorted(SCENARIO_CODEBOOK), help="Scenario code to run; repeatable. Default runs the three primary paper scenarios.")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--save-outputs-pickle", action="store_true")
    parser.add_argument("--save-draw-level-endpoints", action="store_true")
    args = parser.parse_args()

    paths = run_sem_scenarios(
        SEMScenarioRunConfig(
            standardized_input_path=args.standardized_input_path,
            output_dir=args.output_dir,
            scenario_codes=args.scenarios or list(PRIMARY_SCENARIOS),
            n_samples=args.n_samples,
            seed=args.seed,
            target_end_year=args.target_end_year,
            intervention_duration_steps=args.intervention_duration_steps,
            show_progress=args.show_progress,
            save_outputs_pickle=args.save_outputs_pickle,
            save_draw_level_endpoints=args.save_draw_level_endpoints,
        )
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
