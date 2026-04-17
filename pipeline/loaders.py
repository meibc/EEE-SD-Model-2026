# pipeline/loaders.py
"""I/O utilities for pipeline outputs."""

from pathlib import Path
import pickle
from typing import Any

from pipeline.results import RunOutput

## General-purpose loaders/savers
def save(obj: Any, path: Path) -> None:
    """Save object to pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path: Path) -> Any:
    """Load object from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)

## SEM specific loaders
def load_sem_output(path: Path) -> RunOutput:
    """Load SEM output from pickle."""
    return load(path)


def load_fit_results(path: Path):
    """Load fit results from saved SEM output."""
    loaded = load(path)
    if isinstance(loaded, RunOutput):
        return loaded.fit
    if isinstance(loaded, dict) and "fit" in loaded:
        return loaded["fit"]
    raise ValueError("Cannot extract fit results from loaded object")