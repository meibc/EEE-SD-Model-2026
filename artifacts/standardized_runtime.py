"""Reader for the standardized bundle (artifacts/research_bundle.py), used by the scenario pipeline.

Array axes follow one convention: draw first, geography second, measure axes next, and period last.
Thus dynamic matrices are draw x state x target measure x source measure; dynamic trajectories are
draw x state x measure x period; and epidemic posterior parameters are draw x state. A ``sample`` is
reserved for the paired dynamic and epidemic draws created later by ``pipeline/projection.py``.

Key names (``sem_*``, ``cdc_*``) follow the dashboard's standardized schema
(EEE-Model-Dashboard-Integration), which keeps its own copy of this reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.epi.params import EpiParams
from models.dynamic.params import DynamicParams
from data.unit import Unit


@dataclass
class StandardizedBundle:
    """Runtime reader for dashboard-compatible standardized input artifacts."""

    path: Path
    _cache: dict | None = None
    _units_cache: dict[str, Unit] | None = None

    def _load(self) -> dict:
        if self._cache is None:
            # Standardized artifacts are internally generated/trusted.
            # `allow_pickle=True` is needed for object/string arrays in legacy NPZs.
            with np.load(self.path, allow_pickle=True) as raw:
                self._cache = {k: raw[k] for k in raw.files}
            self.validate()
        return self._cache

    def validate(self) -> None:
        """Check every array against the bundle's geographies, measures, periods and draws."""
        d = self._load()
        G, M = len(self.geo_ids), len(self.sem_v_names)
        T, P = len(self.sem_obs_years), len(self.sem_pred_years)
        S = d["sem_J_samples"].shape[0] if "sem_J_samples" in d else None
        C = d["cdc_beta"].shape[0] if "cdc_beta" in d else None
        K = len(self.cdc_raw_names)
        expected = {
            "sem_obs": (G, M, T), "sem_pred": (G, M, P), "sem_fit_J_last": (G, M, M), "sem_fit_drift": (G, M),
            "sem_J_samples": (S, G, M, M), "sem_drift_samples": (S, G, M), "sem_anchor_logits": (S, G, M),
            "sem_ahs_prep_apr": (S, G), "sem_obs_samples": (S, G, M, T),
            "cdc_raw": (G, K, len(self.model_years)), "cdc_raw_native": (G, K, len(self.cdc_native_years)),
            "cdc_kappa_prep": (G,), "cdc_risk0": (G,),
            **{k: (C, G) for k in ("cdc_beta", "cdc_alpha", "cdc_kdx", "cdc_U0", "cdc_post_multiplier")},
        }
        wrong = {k: (d[k].shape, shape) for k, shape in expected.items() if k in d and d[k].shape != shape}
        if wrong:
            raise ValueError(f"Bundle arrays with unexpected shapes (found, expected): {wrong}")

    @property
    def geo_ids(self) -> list[str]:
        return list(np.asarray(self._load()["geo_ids"]).astype(str).tolist())

    @property
    def model_years(self) -> np.ndarray:
        return np.asarray(self._load()["model_years"], dtype=int)

    @property
    def sem_obs_years(self) -> np.ndarray:
        return np.asarray(self._load()["sem_obs_years"], dtype=int)

    @property
    def sem_pred_years(self) -> np.ndarray:
        return np.asarray(self._load()["sem_pred_years"], dtype=int)

    @property
    def sem_v_names(self) -> list[str]:
        return list(np.asarray(self._load()["sem_v_names"]).astype(str).tolist())

    @property
    def cdc_raw_names(self) -> list[str]:
        return list(np.asarray(self._load()["cdc_raw_names"]).astype(str).tolist())

    @property
    def cdc_native_years(self) -> np.ndarray:
        data = self._load()
        if "cdc_native_years" in data:
            return np.asarray(data["cdc_native_years"], dtype=int)
        return self.model_years

    def build_units(self) -> dict[str, Unit]:
        if self._units_cache is not None:
            return self._units_cache

        data = self._load()
        cdc_key = "cdc_raw_native" if "cdc_raw_native" in data else "cdc_raw"
        units: dict[str, Unit] = {}
        for i, geo in enumerate(self.geo_ids):
            units[geo] = Unit(
                id=geo,
                kind="state",
                amis_years=np.asarray(self.sem_obs_years, dtype=float),
                amis_values=np.asarray(data["sem_obs"][i], dtype=float),
                cdc_years=np.asarray(self.cdc_native_years, dtype=float),
                cdc_values=np.asarray(data[cdc_key][i], dtype=float),
                cdc_names=list(self.cdc_raw_names),
            )
        self._units_cache = units
        return units


class _BaseStandardizedLoader:
    """Shared geography indexing for standardized runtime loaders."""

    def __init__(self, bundle: StandardizedBundle):
        self.bundle = bundle
        self._data = bundle._load()
        self._geo_index = {geo: i for i, geo in enumerate(bundle.geo_ids)}

    @property
    def geo_names(self) -> list[str]:
        return self.bundle.geo_ids

    def _idx(self, unit_id: str) -> int:
        return self._geo_index[unit_id]

    def _sample_idx(self, sample_idx: int) -> int:
        s = int(sample_idx)
        if not 0 <= s < self.n_samples:
            raise IndexError(f"Draw {s} is outside this artifact's {self.n_samples} draws")
        return s


class StandardizedDynamicLoader(_BaseStandardizedLoader):
    """Dynamic-model draws (J, drift, anchors, observed inputs) from the standardized bundle."""

    def __init__(self, bundle: StandardizedBundle):
        super().__init__(bundle)
        required = ["sem_J_samples", "sem_fit_J_last", "sem_anchor_logits", "sem_obs_samples"]
        missing = [k for k in required if k not in self._data]
        if missing:
            raise ValueError(f"Standardized bundle is missing required dynamic-model keys: {missing}")

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self._data["sem_J_samples"]).shape[0])

    @property
    def v_names(self) -> list[str]:
        return self.bundle.sem_v_names

    @property
    def ts(self) -> np.ndarray:
        return self.bundle.sem_pred_years

    @property
    def J_samples(self) -> np.ndarray:
        return np.asarray(self._data["sem_J_samples"], dtype=float)

    @property
    def drift_samples(self) -> np.ndarray:
        if "sem_drift_samples" in self._data:
            return np.asarray(self._data["sem_drift_samples"], dtype=float)
        J = self.J_samples
        return np.zeros((J.shape[0], J.shape[1], J.shape[2]), dtype=float)

    @property
    def anchor_logits(self) -> np.ndarray:
        """State anchors, draw x state x measure."""
        return np.asarray(self._data["sem_anchor_logits"], dtype=float)

    def ahs_prep_apr(self, sample_idx: int, unit_id: str) -> float:
        """The draw's AHS-PrEP prevalence ratio from the evidence target (scripts/2_evidence/01)."""
        if "sem_ahs_prep_apr" not in self._data:
            raise ValueError("Bundle has no AHS-PrEP prevalence ratios (fit without the evidence target)")
        return float(self._data["sem_ahs_prep_apr"][self._sample_idx(sample_idx), self._idx(unit_id)])

    def load_sample(self, sample_idx: int, unit_id: str) -> DynamicParams:
        i = self._idx(unit_id)
        s = self._sample_idx(sample_idx)
        return DynamicParams(
            J=self.J_samples[s, i, :, :],
            drift=self.drift_samples[s, i, :],
            reference_logits=self.anchor_logits[s, i, :],
            observed=np.asarray(self._data["sem_obs_samples"][s, i], dtype=float),
        )


class StandardizedEpiLoader(_BaseStandardizedLoader):
    """Epidemic-model posterior draws from the standardized bundle."""

    @property
    def years(self) -> np.ndarray:
        return self.bundle.model_years

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self._data["cdc_beta"]).shape[0])

    def load_sample(self, sample_idx: int, unit_id: str) -> EpiParams:
        i = self._idx(unit_id)
        s = self._sample_idx(sample_idx)
        post = self._data.get("cdc_post_multiplier")
        post_multiplier = 1.0 if post is None else float(np.asarray(post[s, i], dtype=float))
        return EpiParams(
            beta=float(np.asarray(self._data["cdc_beta"][s, i], dtype=float)),
            alpha=float(np.asarray(self._data["cdc_alpha"][s, i], dtype=float)),
            kdx=float(np.asarray(self._data["cdc_kdx"][s, i], dtype=float)),
            U0=float(np.asarray(self._data["cdc_U0"][s, i], dtype=float)),
            # risk0 is a geography-level fixed array, not a posterior draw.
            risk0=float(np.asarray(self._data.get("cdc_risk0", np.ones(len(self.geo_names)))[i], dtype=float)),
            post_multiplier=post_multiplier,
        )
