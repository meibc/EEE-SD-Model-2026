#!/usr/bin/env python3
"""Profile the PrEP -> AHS feedback b: refit the primary model with b held at each value on a grid and record each
draw's loss summed over all units (nation, divisions, states). Each draw's best b is the minimum of its loss curve
(a parabola through the three grid points around the minimum). Run for the wording-adjusted AHS inputs (the
primary) and the unadjusted ones (with the prevalence ratio from all AMIS cycles, as in unadjusted_fb_pos45).

This chose the feedback values (config/specifications.py): +0.15 is the adjusted best fit, -0.075 and +0.30 the
edges of its range, +0.45 the unadjusted best fit.

Outputs (results/05_sensitivity/feedback_profile/):
  profile_<inputs>.csv   loss by b, draw and unit
  best_b.csv             best b by inputs: median, 95% range across draws, share of draws with b <= 0
  loss_curve.csv         mean summed loss above its minimum, by inputs and b
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path
import sys

for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.specifications import SPECIFICATIONS  # noqa: E402
from data.state_inputs import StateInputs  # noqa: E402
from pipeline.dynamic_fit import DynamicFitRunner  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "05_sensitivity/feedback_profile"
INPUTS = {"adjusted": (RESULTS / "02_inputs/state_inputs.npz", {}),
          "unadjusted": (RESULTS / "02_inputs/state_inputs_ahs_unadjusted.npz", {"evidence": "all_years"})}
GRID = np.round(np.arange(-0.3, 0.9001, 0.075), 3)
STATES = RESULTS / "02_inputs/state_screen/primary_states.csv"


def profile_one(job) -> list[dict]:
    which, b, n_draws = job
    path, change = INPUTS[which]
    name = f"profile_{which}_{b}"
    SPECIFICATIONS[name] = {"feedback": float(b), **change}
    states = pd.read_csv(STATES)["state"].astype(str).tolist()
    runner = DynamicFitRunner(name, StateInputs.load(path), states, np.arange(n_draws))
    rows = []
    for d in range(n_draws):
        for unit, fit in zip(runner.unit_ids, runner.fit_draw(d)):
            rows.append({"b": b, "draw": d, "unit": unit, "loss": fit.diagnostics["final_loss"],
                         "success": fit.diagnostics["success"]})
    return rows


def best_b(profile: pd.DataFrame) -> np.ndarray:
    """Each draw's minimizing b: parabola through the three grid points around the minimum (grid edge otherwise)."""
    total = profile.groupby(["draw", "b"])["loss"].sum().unstack()
    grid = total.columns.to_numpy(dtype=float)
    out = []
    for _, row in total.iterrows():
        y = row.to_numpy()
        i = int(np.argmin(y))
        if 0 < i < len(y) - 1:
            a, c, _ = np.polyfit(grid[i - 1:i + 2], y[i - 1:i + 2], 2)
            out.append(-c / (2 * a) if a > 0 else grid[i])
        else:
            out.append(grid[i])
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-draws", type=int, default=60)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(w, b, args.n_draws) for w in INPUTS for b in GRID]
    with Pool(os.cpu_count() or 4) as pool:
        results = pool.map(profile_one, jobs)
    summary, curves = [], []
    for which in INPUTS:
        profile = pd.DataFrame([r for (w, _, _), rows in zip(jobs, results) if w == which for r in rows])
        profile.to_csv(OUT / f"profile_{which}.csv", index=False)
        best = best_b(profile)
        summary.append({"inputs": which, "draws": len(best), "failures": int((~profile["success"]).sum()),
                        "best_b_median": float(np.median(best)), "best_b_low": float(np.percentile(best, 2.5)),
                        "best_b_high": float(np.percentile(best, 97.5)), "share_b_le_0": float(np.mean(best <= 0)),
                        "share_at_grid_edge": float(np.mean((best <= GRID[0]) | (best >= GRID[-1])))})
        mean = profile.groupby(["draw", "b"])["loss"].sum().groupby("b").mean()
        curves.append(pd.DataFrame({"inputs": which, "b": mean.index, "loss_above_min": (mean - mean.min()).to_numpy()}))
    pd.DataFrame(summary).to_csv(OUT / "best_b.csv", index=False)
    pd.concat(curves).to_csv(OUT / "loss_curve.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
