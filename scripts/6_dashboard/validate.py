#!/usr/bin/env python3
"""Check the dashboard's JavaScript model (scripts/6_dashboard/model.js) against the Python model: for a few states,
paired draws, feedback values and scenarios, the projected measures (dynamic years) and annual incidence must agree.
Requires Node.js. Run from the repository root: python scripts/6_dashboard/validate.py
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dashboard import FITS, SPEC, export  # noqa: E402
from artifacts.standardized_runtime import StandardizedBundle, StandardizedDynamicLoader, StandardizedEpiLoader  # noqa: E402
from config.interventions import INTERVENTION_CODEBOOK  # noqa: E402
from config.projection import TARGET_END_YEAR  # noqa: E402
from core.years import build_model_years  # noqa: E402
from pipeline.projection import ProjectionRunner  # noqa: E402

INTERVENTION_CODEBOOK["_eliminate_ahs"] = {"var": "stigma_ahs", "delta": -1.0, "effect": "probability_relative_to_last_observed"}
SCENARIOS = {   # dashboard scenario -> Python runner settings
    "reduce50_2028": ({"type": "reduce_ahs", "ahsReduction": 0.5, "steps": 1}, dict(intervention_codes=["halve_ahs_from_2024"], intervention_duration_steps=1)),
    "eliminate_2026": ({"type": "reduce_ahs", "ahsReduction": 1.0, "steps": 0}, dict(intervention_codes=["_eliminate_ahs"], intervention_duration_steps=0)),
    "resilient50_2028": ({"type": "resilient_services", "gapClosed": 0.5, "steps": 1}, dict(intervention_codes=[], intervention_duration_steps=1, ratio_gap_closed=0.5)),
    "prevent_2028": ({"type": "prevent_worsening", "steps": 1}, dict(intervention_codes=["hold_ahs_at_2024"], intervention_duration_steps=1)),
    "worsening_2026": ({"type": "worsening_climate", "steps": 0}, dict(intervention_codes=["raise_gss_climate", "raise_family_climate"], intervention_duration_steps=0)),
    "stall_2032": ({"type": "prep_stall", "steps": 3}, dict(intervention_codes=["hold_prep_at_2024"], intervention_duration_steps=3)),
}
STATES, DRAWS = ["CA", "AL", "NY"], [0, 1, 2]
FEEDBACK = list(FITS)

NODE = r"""
const M = require(process.argv[2]); const fs = require('fs');
const data = JSON.parse(fs.readFileSync(process.argv[3])); const cases = JSON.parse(fs.readFileSync(process.argv[4]));
let worstTraj = 0, worstInc = 0;
for (const c of cases) {
  const out = M.runDraw(data, c.state, c.draw, String(c.fb), c.scn);
  for (const [key, traj, inc] of [["base", out.base, out.epiBase.incidence], ["scen", out.scen, out.epiScen.incidence]]) {
    c[key].traj.forEach((row, i) => row.forEach((val, t) => { worstTraj = Math.max(worstTraj, Math.abs(val - traj[i][t])); }));
    c[key].inc.forEach((val, t) => { worstInc = Math.max(worstInc, Math.abs(val - inc[t]) / Math.max(1, Math.abs(val))); });
  }
}
console.log(JSON.stringify({cases: cases.length, max_abs_diff_trajectory: worstTraj, max_rel_diff_incidence: worstInc}));
"""


def main() -> None:
    data = export(max(DRAWS) + 1)
    cases = []
    for name in FEEDBACK:
        bundle = StandardizedBundle(SPEC / f"{name}.npz")
        dl, el, units = StandardizedDynamicLoader(bundle), StandardizedEpiLoader(bundle), bundle.build_units()
        years = build_model_years(el.years, target_end_year=TARGET_END_YEAR)
        base_runner = ProjectionRunner(dl, el, units, model_years=years, intervention_codes=[])
        for label, (scn, kw) in SCENARIOS.items():
            runner = ProjectionRunner(dl, el, units, model_years=years, **kw)
            for s in STATES:
                for k in DRAWS:
                    d, e = data["states"][s]["draws"][k]["d"], data["states"][s]["draws"][k]["e"]
                    b, sc = base_runner.predict_sample(s, d, e), runner.predict_sample(s, d, e)
                    cases.append({"state": s, "draw": k, "fb": FITS[name], "scn": scn, "label": label,
                                  "base": {"traj": np.asarray(b.dynamic_trajectory).tolist(), "inc": np.asarray(b.epi_output.incidence).tolist()},
                                  "scen": {"traj": np.asarray(sc.dynamic_trajectory).tolist(), "inc": np.asarray(sc.epi_output.incidence).tolist()}})
    with tempfile.TemporaryDirectory() as tmp:
        dpath, cpath, jpath = Path(tmp, "data.json"), Path(tmp, "cases.json"), Path(tmp, "check.js")
        dpath.write_text(json.dumps(data)); cpath.write_text(json.dumps(cases)); jpath.write_text(NODE)
        result = subprocess.run(["node", str(jpath), str(Path(__file__).resolve().parent / "model.js"), str(dpath), str(cpath)],
                                capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
