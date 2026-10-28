#!/usr/bin/env python3
"""Build the research bundle for the primary fit (artifacts/research_bundle.py), read by scripts/3_model/05 and the results scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifacts.research_bundle import build_research_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "results/02_inputs/epi_fit/epi_bundle.npz",
                        help="Epidemiological posterior and CDC inputs (scripts/3_model/02_fit_epi_model.py).")
    parser.add_argument("--fit", type=Path, default=ROOT / "results/03_fits/primary/fits/primary.npz")
    parser.add_argument("--input-draws", type=Path, default=ROOT / "results/02_inputs/state_inputs.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/03_fits/primary/standardized_research.npz")
    args = parser.parse_args()
    print(json.dumps(build_research_bundle(args.base, args.fit, args.output, args.input_draws), indent=2))


if __name__ == "__main__":
    main()
