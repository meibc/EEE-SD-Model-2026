#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.sem_mc_fits import SEMDrawFitConfig, fit_sem_draws


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit centered-logit drift SEM to cached input draws")
    parser.add_argument("--input-path", type=Path, default=Path("output/sem_input_draws_v5.npz"))
    parser.add_argument("--output-path", type=Path, default=Path("output/sem_fits_centered_drift_v5.npz"))
    parser.add_argument("--data-path", type=Path, default=Path("/Users/meibinchen/Documents/GitHub/EEE/Factor Analysis v5.xlsx"))
    parser.add_argument("--n-draws", type=int, default=None, help="Number of draws to fit; default fits all draws")
    parser.add_argument("--start-draw", type=int, default=0)
    parser.add_argument("--save-rolling", action="store_true", help="Also save full rolling J sequence")
    args = parser.parse_args()

    path = fit_sem_draws(
        SEMDrawFitConfig(
            input_path=args.input_path,
            output_path=args.output_path,
            data_path=args.data_path,
            n_draws=args.n_draws,
            start_draw=args.start_draw,
            save_rolling=args.save_rolling,
        )
    )
    print(f"Wrote fitted SEM draws: {path}")
    print(f"Wrote audit: {path.with_suffix('.audit.csv')}")
    print(f"Wrote summary: {path.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
