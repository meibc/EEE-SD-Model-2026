#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.interventions import PAPER_SCENARIOS, SCENARIO_CODEBOOK, SUPPLEMENT_SCENARIOS
from config.projection import INTERIM_REPORT_YEARS, INTERVENTION_DURATION_STEPS, N_SAMPLES, SEED, TARGET_END_YEAR
from pipeline.scenarios import ScenarioRunConfig, run_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline and scenarios from the research bundle (dynamic + epidemic models)")
    parser.add_argument("--standardized-input-path", type=Path, default=REPO_ROOT / "results/03_fits/primary/standardized_research.npz")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/04_scenarios/paper_static")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-end-year", type=int, default=TARGET_END_YEAR)
    parser.add_argument(
        "--intervention-duration-steps",
        type=int,
        default=INTERVENTION_DURATION_STEPS,
        help="Number of 2-year dynamic-model steps over which interventions ramp to full strength; 3 = 6 years.",
    )
    parser.add_argument("--scenario", dest="scenarios", action="append", choices=sorted(SCENARIO_CODEBOOK), help="Scenario code to run; repeatable. Default: the paper and supplement scenarios.")
    parser.add_argument(
        "--report-year", dest="report_years", type=int, action="append",
        help="Endpoint year to summarize; repeatable. Default: config/projection.py INTERIM_REPORT_YEARS and the target end year.",
    )
    parser.add_argument("--contextual-prep-pr", type=float, default=None,
                        help="Contextual upper-bound sensitivity: multiply scenario PrEP by PR ** (AHS change / 0.10).")
    parser.add_argument("--zero-projection-edge", action="append", default=[], metavar="SOURCE:TARGET",
                        help="Set a pathway to zero in projections (baseline and scenarios), e.g. prep_used:stigma_ahs; repeatable.")
    parser.add_argument("--constant-pr-prep", action="store_true",
                        help="Primary analysis: scenario PrEP from the respondent-level prevalence ratio at the projected levels.")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--save-outputs-pickle", action="store_true")
    parser.add_argument("--save-draw-level-endpoints", action="store_true")
    args = parser.parse_args()

    paths = run_scenarios(
        ScenarioRunConfig(
            standardized_input_path=args.standardized_input_path,
            output_dir=args.output_dir,
            scenario_codes=args.scenarios or list(PAPER_SCENARIOS + SUPPLEMENT_SCENARIOS),
            n_samples=args.n_samples,
            seed=args.seed,
            target_end_year=args.target_end_year,
            intervention_duration_steps=args.intervention_duration_steps,
            report_years=tuple(args.report_years or sorted({*INTERIM_REPORT_YEARS, args.target_end_year})),
            show_progress=args.show_progress,
            save_outputs_pickle=args.save_outputs_pickle,
            save_draw_level_endpoints=args.save_draw_level_endpoints,
            contextual_prep_pr=args.contextual_prep_pr,
            constant_pr_prep=args.constant_pr_prep,
            zero_projection_edges=tuple(tuple(e.split(":")) for e in args.zero_projection_edge),
        )
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
