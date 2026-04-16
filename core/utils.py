from __future__ import annotations

import json
from pathlib import Path
import numpy as np

def load_results(filename: Path | str) -> dict:
    """Load results from .npz and optional sidecar .json."""
    path = Path(filename)
    if path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {path}")

    np_results = np.load(path, allow_pickle=True)
    results = {k: np_results[k] for k in np_results.files}

    json_path = path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r") as f:
            other_results = json.load(f)
        results.update(other_results)

    return results
