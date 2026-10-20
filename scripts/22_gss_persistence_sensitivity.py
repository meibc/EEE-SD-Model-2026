#!/usr/bin/env python3
from __future__ import annotations

"""Run the focused GSS persistence/reference diagnostic."""

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "stigma_drift_sensitivity",
    HERE / "21_stigma_drift_sensitivity.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

module.OUT = module.ROOT / "output/gss_persistence_sensitivity_deterministic"
module.VARIANTS = {
    f"current_ref_phi_{str(phi).replace('.', 'p')}": {
        "added_drift": (),
        "prep_to_gss": False,
        "construct_refs": False,
        "gss_persistence": phi,
    }
    for phi in (0.85, 0.95, 0.98, 0.99)
} | {
    f"construct_ref_phi_{str(phi).replace('.', 'p')}": {
        "added_drift": (),
        "prep_to_gss": False,
        "construct_refs": True,
        "gss_persistence": phi,
    }
    for phi in (0.85, 0.95, 0.98, 0.99)
}

# The shared summary routine uses a named primary comparator.
module.VARIANTS["primary"] = module.VARIANTS.pop("current_ref_phi_0p85")

if __name__ == "__main__":
    module.main()
