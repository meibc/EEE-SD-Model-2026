from __future__ import annotations

"""SEM model parameters and loader."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from core.utils import load_results
from .params_base import GeoIndexedParamsLoader

@dataclass
class SEMParams:
    """SEM parameters for one geography."""
    J: np.ndarray  # (m, m)


class SEMParamsLoader(GeoIndexedParamsLoader):
    """Load SEM parameters from npz."""
    
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self._cache: dict | None = None
    
    def _load(self) -> dict:
        """Load and cache raw data."""
        if self._cache is not None:
            return self._cache
        
        data = load_results(self.npz_path)
        
        # (S, G, m, m)
        J_samples = np.asarray(data["Jmeans_stack"])
        geos = list(data["unit_order"])
        v_names = list(data["v_names"])
        ts = list(data["ts"])
        
        self._cache = {
            "geo_names": geos,
            "J_samples": J_samples,  # (S, G, m, m)
            "v_names": v_names,
            "ts": ts,
        }
        return self._cache
    
    @property
    def n_samples(self) -> int:
        return self._load()["J_samples"].shape[0]

    @property
    def v_names(self) -> list[str] | None:
        return self._load()["v_names"]

    @property
    def ts(self) -> np.ndarray | None:
        return self._load()["ts"]

    @property
    def J_samples(self) -> np.ndarray:
        return self._load()["J_samples"]
    
    def load_point_estimates(self, unit_id: str) -> SEMParams:
        """Load mean for one geography."""
        data = self._load()
        idx = self._get_geo_idx(unit_id)
        
        # Mean over samples: (S, G, m, m) → (m, m)
        J_mean = data["J_samples"][:, idx, :, :].mean(axis=0)
        
        return SEMParams(J=J_mean)
    
    def load_sample(self, sample_idx: int, unit_id: str) -> SEMParams:
        """Load single sample for one geography."""
        data = self._load()
        geo_idx = self._get_geo_idx(unit_id)
        
        # (S, G, m, m) → (m, m)
        J = data["J_samples"][sample_idx, geo_idx, :, :]
        
        return SEMParams(J=J)
