#!/usr/bin/env python3
"""Arrange the respondent-bootstrap estimates (scripts/03) as model-input draws.

Places the state, division and national bootstrap estimates in the model's unit,
measure and period order, requires the R-prepared estimates to be complete,
clips probability boundaries for finite logit transformation, and writes the
``Xbars_samples`` NPZ read by scripts/09 (input/state_input_draws.py). Division
fallback is completed and audited upstream in scripts/03_state_bootstrap.R.

Outputs: results/02_inputs/state_input_draws.npz plus .audit.csv. For the
AHS-unadjusted sensitivity, pass the unadjusted bootstrap, workbook and output
paths (results/logs/revision_run.sh).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from input.state_input_draws import build_state_input_draws  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-path", type=Path, default=RESULTS / "02_inputs/state_bootstrap/state_input_draws.csv.gz")
    parser.add_argument("--data-path", type=Path, default=RESULTS / "01_weights/state_model_inputs.xlsx")
    parser.add_argument("--output-path", type=Path, default=RESULTS / "02_inputs/state_input_draws.npz")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--expected-draws", type=int, default=500)
    args = parser.parse_args()

    metrics = build_state_input_draws(args.bootstrap_path, args.data_path, args.output_path, args.eps, args.expected_draws)
    print(f"Wrote {args.output_path}: {metrics['n_draws']} draws x {metrics['n_variables']} measures x "
          f"{metrics['n_years']} periods x {metrics['n_units']} units; "
          f"{metrics['n_boundary']} at the bounds")


if __name__ == "__main__":
    main()
