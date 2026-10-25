#!/usr/bin/env python3
"""Fit the hierarchical state model to every input draw, for each model specification.

For each specification (pipeline/state_fit.py) this writes
  <output-dir>/fits/<specification>.npz                      fitted transition matrices and drifts
  <output-dir>/fits/<specification>.loss_components.csv       loss components by draw and unit
  <output-dir>/scenarios/<specification>.csv                  incidence change in the target year
                                                              (with --scenarios; supplement S7)
The primary fit goes to results/03_fits/primary; the S7 specifications to
results/05_sensitivity/model_specification (--output-dir).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.state_fit import EVIDENCE_PATH, SPECIFICATIONS, fit_specification, specification_scenarios  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--specifications", default="primary",
                        help=f"Comma-separated, from: {', '.join(SPECIFICATIONS)}")
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--draw-seed", type=int, default=20260921,
                        help="Seed for choosing input draws when --n-draws is below the number available.")
    parser.add_argument("--scenarios", action="store_true", help="Also project the paper scenarios.")
    parser.add_argument("--constant-pr-prep", action="store_true",
                        help="Scenario PrEP change from the respondent-level prevalence ratio (primary analysis); "
                             "outputs go to <output-dir>/scenarios_static/.")
    parser.add_argument("--scenario-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-year", type=int, default=2036)
    parser.add_argument("--selected-states", type=Path, default=RESULTS / "02_inputs/state_screen/primary_states.csv")
    parser.add_argument("--input-draws", type=Path, default=RESULTS / "02_inputs/state_input_draws.npz")
    parser.add_argument("--workbook", type=Path, default=RESULTS / "01_weights/state_model_inputs.xlsx")
    parser.add_argument("--epi-base", type=Path, default=ROOT / "results/02_inputs/epi_fit/epi_bundle.npz")
    parser.add_argument("--evidence-path", type=Path, default=EVIDENCE_PATH,
                        help="AHS-PrEP evidence (scripts/08b); the all-years file for the unadjusted-wording fit.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "03_fits/primary")
    parser.add_argument("--force", action="store_true", help="Refit specifications whose outputs already exist.")
    args = parser.parse_args()

    specifications = [s.strip() for s in args.specifications.split(",") if s.strip()]
    unknown = sorted(set(specifications) - set(SPECIFICATIONS))
    if unknown:
        raise ValueError(f"Unknown specifications: {unknown}")
    states = pd.read_csv(args.selected_states)["state"].astype(str).tolist()
    n_available = int(np.load(args.input_draws, allow_pickle=True)["Xbars_samples"].shape[0])
    if args.n_draws > n_available:
        raise ValueError(f"Requested {args.n_draws} draws but only {n_available} are available")
    draw_indices = np.sort(np.random.default_rng(args.draw_seed).choice(n_available, args.n_draws, replace=False))
    (args.output_dir / "fits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source_draw_index": draw_indices}).to_csv(args.output_dir / "paired_source_draw_indices.csv", index=False)

    for name in specifications:
        fit_path = args.output_dir / f"fits/{name}.npz"
        if args.force or not fit_path.exists():
            fit_specification(name, draw_indices, states, args.input_draws, args.workbook, fit_path,
                              evidence_path=args.evidence_path)
        else:
            print(f"using existing fit: {fit_path}")
        if args.scenarios:
            subdir = args.output_dir / ("scenarios_static" if args.constant_pr_prep else "scenarios")
            subdir.mkdir(exist_ok=True)
            specification_scenarios(name, fit_path, args.epi_base, args.workbook, args.input_draws,
                                    args.scenario_samples, args.seed, args.target_year,
                                    subdir / f"{name}.csv", constant_pr_prep=args.constant_pr_prep)


if __name__ == "__main__":
    main()
