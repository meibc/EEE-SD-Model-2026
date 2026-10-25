#!/usr/bin/env python3
from __future__ import annotations

"""Where each measure's projected change comes from: own momentum and trend vs the rest of the system.

For every state and draw, the fitted system is projected from the observed last period (2024) for
``--steps`` two-year steps (6 = 2036) in state-anchored logits, and each measure's change is split
exactly (models/sbm/structural.py, trajectory_decomposition) into
  carryover_own      its own 2024 departure from its anchor decaying through its persistence
  carryover_system   the other measures' 2024 departures reaching it through the pathways
  drift_own          its own drift accumulated through its own persistence
  feedback_own       its own drift amplified (+) or damped (-) by loops through other measures
  from_<group>       the drifts of the other measures in each group reaching it through the system
                     (stigma: AHS, GSS, family; engagement: disclosure, healthcare contact, testing;
                     prep: PrEP use; risk: risk behavior)
The logit parts are converted to percentage points at the midpoint of the projected change
(dp ~ p(1 - p) dx), so they sum to the change in percentage points to first order. "system" is
everything other than the measure's own carryover and drift.

Outputs (results/07_structural/ by default):
  system_decomposition.csv          by state and measure, medians across draws (percentage points)
  system_decomposition_draws.csv.gz by state, draw and measure
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.sbm.estimation.transforms import FeatureTransformer  # noqa: E402
from models.sbm.structural import trajectory_decomposition  # noqa: E402

RESULTS = ROOT / "results"
GROUPS = {"stigma": ("stigma_ahs", "stigma_gss", "stigma_family"),
          "engagement": ("out_gid", "seehcp", "hivtest12"),
          "prep": ("prep_used",), "risk": ("risk_behavior",)}
PARTS = ["carryover_own", "carryover_system", "drift_own", "feedback_own", *[f"from_{g}" for g in GROUPS]]


def decompose_state(J, drift, anchors, observed, v_names, steps) -> pd.DataFrame:
    """Draw x measure decomposition for one state, in percentage points."""
    x0 = np.stack([FeatureTransformer(reference_logits=a).transform(y)[0][:, -1] for a, y in zip(anchors, observed)])
    out = trajectory_decomposition(J, drift, x0, steps)
    p_mid = 1 / (1 + np.exp(-(x0 + anchors + out["total"] / 2)))
    scale = 100 * p_mid * (1 - p_mid)
    m = len(v_names)
    by_source = out["drift_by_source"]
    parts = {"total_pp": out["total"], "carryover_own": out["carryover_own"],
             "carryover_system": out["carryover_system"], "drift_own": out["drift_own"],
             "feedback_own": by_source[:, np.arange(m), np.arange(m)]}
    for group, members in GROUPS.items():
        cols = [v_names.index(v) for v in members]
        mask = np.zeros((m, m))
        mask[:, cols] = 1
        mask[np.arange(m), np.arange(m)] = 0                       # a measure's own drift is not "from" its group
        parts[f"from_{group}"] = np.einsum("dij,ij->di", by_source, mask)
    draws = J.shape[0]
    frame = pd.DataFrame({"draw": np.repeat(np.arange(draws), m), "variable": np.tile(v_names, draws)})
    for name, values in parts.items():
        frame[name] = (values * scale).ravel()
    frame["system"] = frame[["carryover_system", "feedback_own", *[f"from_{g}" for g in GROUPS]]].sum(axis=1)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=RESULTS / "03_fits/primary/standardized_research.npz")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "07_structural")
    parser.add_argument("--steps", type=int, default=6, help="Two-year steps after the last observed period.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    b = np.load(args.bundle, allow_pickle=True)
    v_names = b["sem_v_names"].astype(str).tolist()
    frames = []
    for g, state in enumerate(b["geo_ids"].astype(str)):
        frame = decompose_state(np.asarray(b["sem_J_samples"][:, g], float), np.asarray(b["sem_drift_samples"][:, g], float),
                                np.asarray(b["sem_anchor_logits"][:, g], float), np.asarray(b["sem_obs_samples"][:, g], float),
                                v_names, args.steps)
        frames.append(frame.assign(state=state))
    draws = pd.concat(frames, ignore_index=True)
    draws.to_csv(args.output_dir / "system_decomposition_draws.csv.gz", index=False)
    summary = draws.groupby(["state", "variable"])[["total_pp", *PARTS, "system"]].median().reset_index()
    summary.to_csv(args.output_dir / "system_decomposition.csv", index=False)
    for var in ("stigma_ahs", "prep_used"):
        print(f"\n{var}: projected change to {int(b['sem_obs_years'][-1]) + 2 * args.steps} (pp, medians)")
        print(summary[summary.variable.eq(var)].set_index("state")[["total_pp", *PARTS, "system"]]
              .sort_values("total_pp", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
