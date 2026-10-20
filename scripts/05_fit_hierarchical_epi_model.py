#!/usr/bin/env python3
from __future__ import annotations

"""Stage 04: fit the long-running hierarchical Bayesian epi model.

This stage is intentionally separate from SEM fitting, scenario generation, and
standardized dashboard export. It can take a long time, especially with
`--run-mode full`; do not run it as part of ordinary smoke tests.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.hierarchical_epi_fit import HierarchicalEpiFitConfig, fit_hierarchical_epi_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit hierarchical Bayesian CDC incidence/diagnosis parameters. This is long-running."
    )
    parser.add_argument("--standardized-input-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE-Model-Dashboard-Integration/standardized_input_v7.npz"))
    parser.add_argument("--data-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/hierarchical_epi_fit_v7d"))
    parser.add_argument("--output-nc-path", type=Path, default=None)
    parser.add_argument("--model-scope", choices=["all_states", "selected_states"], default="all_states")
    parser.add_argument("--states", nargs="+", default=["CA", "TX", "NY", "MA", "MS"])
    parser.add_argument("--run-mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-diagnosis", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--quick-draws", type=int, default=100)
    parser.add_argument("--quick-tune", type=int, default=300)
    parser.add_argument("--full-draws", type=int, default=500)
    parser.add_argument("--full-tune", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.99)
    args = parser.parse_args()

    output_path = fit_hierarchical_epi_model(
        HierarchicalEpiFitConfig(
            standardized_input_path=args.standardized_input_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            output_nc_path=args.output_nc_path,
            model_scope=args.model_scope,
            run_mode=args.run_mode,
            fit_diagnosis=not args.skip_diagnosis,
            states=args.states,
            random_seed=args.random_seed,
            quick_draws=args.quick_draws,
            quick_tune=args.quick_tune,
            full_draws=args.full_draws,
            full_tune=args.full_tune,
            target_accept=args.target_accept,
        )
    )
    print(f"Wrote hierarchical epi posterior: {output_path}")


if __name__ == "__main__":
    main()
