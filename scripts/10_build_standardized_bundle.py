#!/usr/bin/env python3
"""Build the research bundle for the primary fit (input/research_bundle.py), read by scripts/11-13."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from input.research_bundle import build_research_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "results/02_inputs/epi_fit/epi_bundle.npz",
                        help="Epidemiological posterior and CDC inputs (scripts/04b_fit_epi_model.py).")
    parser.add_argument("--fit", type=Path, default=ROOT / "results/03_fits/primary/fits/primary.npz")
    parser.add_argument("--workbook", type=Path, default=ROOT / "results/01_weights/state_model_inputs.xlsx")
    parser.add_argument("--input-draws", type=Path, default=ROOT / "results/02_inputs/state_input_draws.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/03_fits/primary/standardized_research.npz")
    args = parser.parse_args()
    print(json.dumps(build_research_bundle(args.base, args.fit, args.output, args.workbook, args.input_draws), indent=2))


if __name__ == "__main__":
    main()
