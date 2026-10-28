#!/usr/bin/env python3
"""Write the model-input file: the one step that reads the R outputs for the model.

Reads the respondent-bootstrap estimates (scripts/1_data/04) and the model-input workbook (scripts/1_data/03), and
writes the observed draws (draw x unit x measure x period), point estimates, units with their
divisions and sample sizes, and the covariance, in the model's order (data/state_inputs.py).
The epidemic fit (3_model/02), the dynamic-model fits (03) and the bundle (04) read only this file. Requires the
R-prepared estimates to be complete and clips probability boundaries for finite logit transformation.
Division fallback is completed and audited upstream in scripts/1_data/04_state_bootstrap.R.

Outputs: results/02_inputs/state_inputs.npz plus .audit.csv. For the
AHS-unadjusted sensitivity, pass the unadjusted bootstrap, workbook and output
paths (run_pipeline.sh does both).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.math.transforms import PROB_EPS  # noqa: E402
from data.state_inputs import build_state_inputs  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-path", type=Path, default=RESULTS / "02_inputs/state_bootstrap/state_input_draws.csv.gz")
    parser.add_argument("--data-path", type=Path, default=RESULTS / "01_weights/state_model_inputs.xlsx")
    parser.add_argument("--output-path", type=Path, default=RESULTS / "02_inputs/state_inputs.npz")
    parser.add_argument("--eps", type=float, default=PROB_EPS)
    parser.add_argument("--expected-draws", type=int, default=500)
    args = parser.parse_args()

    metrics = build_state_inputs(args.bootstrap_path, args.data_path, args.output_path, args.eps, args.expected_draws)
    print(f"Wrote {args.output_path}: {metrics['n_draws']} draws x {metrics['n_units']} units x "
          f"{metrics['n_variables']} measures x {metrics['n_years']} periods; "
          f"{metrics['n_clipped_or_boundary']} clipped to or at the bounds")


if __name__ == "__main__":
    main()
