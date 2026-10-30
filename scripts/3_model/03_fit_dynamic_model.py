#!/usr/bin/env python3
"""Fit the hierarchical dynamic model (nation, divisions, states) to every input draw, for each model specification.

For each specification (pipeline/dynamic_fit.py) this writes
  <output-dir>/fits/<specification>.npz                      fitted transition matrices and drifts
  <output-dir>/fits/<specification>.loss_components.csv       loss components by draw and unit
The primary fit goes to results/03_fits/primary; the other specifications to
results/05_sensitivity/model_specification (--output-dir). Scenario projections use scripts/3_model/04 (bundle)
and scripts/3_model/05.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.state_inputs import StateInputs  # noqa: E402
from config.fit import DRAW_SEED, FIT_DRAWS  # noqa: E402
from config.specifications import SPECIFICATIONS  # noqa: E402
from pipeline.dynamic_fit import EVIDENCE_PATH, fit_specification  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--specifications", default="primary",
                        help=f"Comma-separated, from: {', '.join(SPECIFICATIONS)}")
    parser.add_argument("--n-draws", type=int, default=FIT_DRAWS)
    parser.add_argument("--draw-seed", type=int, default=DRAW_SEED,
                        help="Seed for choosing input draws when --n-draws is below the number available.")
    parser.add_argument("--selected-states", type=Path, default=RESULTS / "02_inputs/state_screen/primary_states.csv")
    parser.add_argument("--input-draws", type=Path, default=RESULTS / "02_inputs/state_inputs.npz")
    parser.add_argument("--evidence-path", type=Path, default=EVIDENCE_PATH,
                        help="AHS-PrEP evidence (scripts/2_evidence/01); the all-years file for the unadjusted-wording fit.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "03_fits/primary")
    parser.add_argument("--force", action="store_true", help="Refit specifications whose outputs already exist.")
    args = parser.parse_args()

    specifications = [s.strip() for s in args.specifications.split(",") if s.strip()]
    unknown = sorted(set(specifications) - set(SPECIFICATIONS))
    if unknown:
        raise ValueError(f"Unknown specifications: {unknown}")
    states = pd.read_csv(args.selected_states)["state"].astype(str).tolist()
    inputs = StateInputs.load(args.input_draws)
    n_available = inputs.observed_draws.shape[0]
    if args.n_draws > n_available:
        raise ValueError(f"Requested {args.n_draws} draws but only {n_available} are available")
    draw_indices = np.sort(np.random.default_rng(args.draw_seed).choice(n_available, args.n_draws, replace=False))
    (args.output_dir / "fits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"input_draw_index": draw_indices, "source_draw_id": inputs.source_draw_ids[draw_indices]}).to_csv(args.output_dir / "paired_input_draw_indices.csv", index=False)

    for name in specifications:
        fit_path = args.output_dir / f"fits/{name}.npz"
        if args.force or not fit_path.exists():
            fit_specification(name, draw_indices, states, inputs, fit_path,
                              evidence_path=args.evidence_path)
        else:
            print(f"using existing fit: {fit_path}")


if __name__ == "__main__":
    main()
