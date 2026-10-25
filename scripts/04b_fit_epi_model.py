#!/usr/bin/env python3
from __future__ import annotations

"""Fit the hierarchical Bayesian epidemiological model and write its bundle.

Calibrates incidence and diagnosis parameters for every state to CDC MSM
estimated incidence and diagnoses, using the AMIS PrEP, risk and testing
estimates (pipeline/hierarchical_epi_fit.py). Needs scripts/00_build_cdc_stocks.py,
01 and 04. The full run (4 chains x 500 draws, 1,000 tuning steps) takes about
3 hours; --run-mode quick (2 x 100, 300 tuning) checks the wiring.

Outputs (results/02_inputs/epi_fit/):
  posterior.nc              posterior draws
  period_targets.csv        observed diagnosis and incidence period means
  period_fit_summary.csv    posterior period means against the targets
  epi_bundle.npz            parameter draws and CDC stocks read by scripts/09-12

--bundle-only rebuilds epi_bundle.npz from an existing posterior.nc.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.hierarchical_epi_fit import (  # noqa: E402
    HierarchicalEpiFitConfig, build_hierarchical_epi_inputs, fit_hierarchical_epi_model, write_epi_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/02_inputs/epi_fit")
    parser.add_argument("--run-mode", choices=["quick", "full"], default="full")
    parser.add_argument("--model-scope", choices=["all_states", "selected_states"], default="all_states")
    parser.add_argument("--states", nargs="+", default=["CA", "TX", "NY", "MA", "MS"])
    parser.add_argument("--skip-diagnosis", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--quick-draws", type=int, default=100)
    parser.add_argument("--quick-tune", type=int, default=300)
    parser.add_argument("--bundle-only", action="store_true")
    args = parser.parse_args()

    config = HierarchicalEpiFitConfig(
        output_dir=args.output_dir, run_mode=args.run_mode, model_scope=args.model_scope, states=args.states,
        fit_diagnosis=not args.skip_diagnosis, random_seed=args.random_seed,
        quick_draws=args.quick_draws, quick_tune=args.quick_tune,
    )
    if args.bundle_only:
        import arviz as az
        posterior = az.from_netcdf(args.output_dir / "posterior.nc").posterior
        write_epi_bundle(posterior, build_hierarchical_epi_inputs(config), config.cdc_stocks_path,
                         args.output_dir / "epi_bundle.npz")
        print(f"Wrote {args.output_dir / 'epi_bundle.npz'}")
    else:
        fit_hierarchical_epi_model(config)


if __name__ == "__main__":
    main()
