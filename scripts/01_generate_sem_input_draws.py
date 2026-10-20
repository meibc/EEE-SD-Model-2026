#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.sem_mc_inputs import SEMInputDrawConfig, generate_sem_input_draws


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached SEM input draws from Factor Analysis workbook")
    parser.add_argument("--data-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx"))
    parser.add_argument("--summary-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE-SD-Model-2026/amis_summary_stats_v4.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("output/sem_input_draws_v5.npz"))
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--value-source", choices=["factor_analysis", "weighted_prop"], default="factor_analysis")
    parser.add_argument("--smooth-draws", action="store_true", help="Lightly smooth sampled trajectories after Beta sampling")
    parser.add_argument("--smooth-strength", type=float, default=0.5, help="Convex smoothing strength in [0, 1]")
    args = parser.parse_args()

    path = generate_sem_input_draws(
        SEMInputDrawConfig(
            data_path=args.data_path,
            summary_path=args.summary_path,
            output_path=args.output_path,
            n_draws=args.n_draws,
            seed=args.seed,
            value_source=args.value_source,
            smooth_draws=args.smooth_draws,
            smooth_strength=args.smooth_strength,
        )
    )
    print(f"Wrote SEM input draws: {path}")
    print(f"Wrote audit: {path.with_suffix('.audit.csv')}")


if __name__ == "__main__":
    main()
