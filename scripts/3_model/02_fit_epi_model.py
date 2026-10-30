#!/usr/bin/env python3
"""Fit the hierarchical Bayesian epidemiological model and write its bundle.

Calibrates incidence and diagnosis parameters for every state to CDC MSM
estimated incidence and diagnoses, using the AMIS PrEP, risk and testing
estimates (pipeline/epi_fit.py). Needs scripts/1_data/02_build_cdc_stocks.R,
01 and 04. The full run (4 chains x 500 draws, 1,000 tuning steps) takes about
3 hours; --run-mode quick (2 x 100, 300 tuning) checks the wiring.

Outputs (results/02_inputs/epi_fit/):
  posterior.nc              posterior draws
  period_targets.csv        observed diagnosis and incidence period means
  period_fit_summary.csv    posterior period means against the targets
  epi_bundle.npz            parameter draws and CDC stocks read by scripts/3_model/04 (bundle)

--bundle-only rebuilds epi_bundle.npz from an existing posterior.nc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.epi_fit import (  # noqa: E402
    HierarchicalEpiFitConfig, build_hierarchical_epi_inputs, fit_hierarchical_epi_model, write_epi_bundle,
)


def main() -> None:
    defaults = HierarchicalEpiFitConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--run-mode", choices=["quick", "full"], default="full")
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument("--quick-draws", type=int, default=defaults.quick_draws)
    parser.add_argument("--quick-tune", type=int, default=defaults.quick_tune)
    parser.add_argument("--bundle-only", action="store_true")
    args = parser.parse_args()

    config = HierarchicalEpiFitConfig(
        output_dir=args.output_dir, run_mode=args.run_mode, random_seed=args.random_seed,
        quick_draws=args.quick_draws, quick_tune=args.quick_tune,
    )
    if args.bundle_only:
        import arviz as az
        posterior = az.from_netcdf(args.output_dir / "posterior.nc").posterior
        write_epi_bundle(posterior, build_hierarchical_epi_inputs(config), args.output_dir / "epi_bundle.npz")
        print(f"Wrote {args.output_dir / 'epi_bundle.npz'}")
    else:
        fit_hierarchical_epi_model(config)


if __name__ == "__main__":
    main()
