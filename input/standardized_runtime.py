from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.epi.params import CDCParams
from models.sbm.params import SEMParams
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
        return self._cache

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
    def sem_obs(self) -> np.ndarray:
        """Observed probability-scale SEM trajectories by geography."""
        return np.asarray(self._load()["sem_obs"], dtype=float)

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
    def cdc_raw(self) -> np.ndarray:
        """CDC inputs aligned to the model-year grid."""
        return np.asarray(self._load()["cdc_raw"], dtype=float)

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
                amis_names=list(self.sem_v_names),
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
        # Intentional modulo: SEM and CDC posterior artifacts can have different
        # sample counts, while callers may reuse sampled indices across loaders.
        return int(sample_idx) % self.n_samples


class StandardizedSEMParamsLoader(_BaseStandardizedLoader):
    """Load SEM J/drift draws from dashboard-compatible standardized v8 NPZ."""

    def __init__(self, bundle: StandardizedBundle):
        super().__init__(bundle)
        required = ["sem_J_samples", "sem_fit_J_last", "sem_anchor_logits"]
        missing = [k for k in required if k not in self._data]
        if missing:
            raise ValueError(f"Standardized SEM artifact is missing required keys: {missing}")

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
        """State anchors (draw x state x measure mean logits)."""
        return np.asarray(self._data["sem_anchor_logits"], dtype=float)

    def ahs_prep_apr(self, sample_idx: int, unit_id: str) -> float:
        """The draw's AHS-PrEP prevalence ratio from the evidence target (scripts/08b)."""
        if "sem_ahs_prep_apr" not in self._data:
            raise ValueError("Bundle has no AHS-PrEP prevalence ratios (fit without the evidence target)")
        return float(self._data["sem_ahs_prep_apr"][self._sample_idx(sample_idx), self._idx(unit_id)])

    def load_point_estimates(self, unit_id: str) -> SEMParams:
        i = self._idx(unit_id)
        return SEMParams(
            J=np.nanmean(self.J_samples[:, i, :, :], axis=0),
            drift=np.nanmean(self.drift_samples[:, i, :], axis=0),
            reference_logits=np.nanmedian(self.anchor_logits[:, i, :], axis=0),
        )

    def load_sample(self, sample_idx: int, unit_id: str) -> SEMParams:
        i = self._idx(unit_id)
        s = self._sample_idx(sample_idx)
        observed = None
        if "sem_obs_samples" in self._data:
            observed = np.asarray(self._data["sem_obs_samples"][s, i], dtype=float)
        return SEMParams(
            J=self.J_samples[s, i, :, :],
            drift=self.drift_samples[s, i, :],
            reference_logits=self.anchor_logits[s, i, :],
            observed=observed,
        )


class StandardizedCDCParamsLoader(_BaseStandardizedLoader):
    """Load CDC/hierarchical parameter draws from standardized input artifacts."""

    def __init__(self, bundle: StandardizedBundle):
        super().__init__(bundle)

    @property
    def years(self) -> np.ndarray:
        return self.bundle.model_years

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self._data["cdc_beta"]).shape[0])

    def load_point_estimates(self, unit_id: str) -> CDCParams:
        i = self._idx(unit_id)
        post = self._data.get("cdc_post_multiplier")
        post_multiplier = 1.0 if post is None else float(np.asarray(post[:, i], dtype=float).mean())
        return CDCParams(
            beta=float(np.asarray(self._data["cdc_beta"][:, i], dtype=float).mean()),
            alpha=float(np.asarray(self._data["cdc_alpha"][:, i], dtype=float).mean()),
            kdx=float(np.asarray(self._data["cdc_kdx"][:, i], dtype=float).mean()),
            U0=float(np.asarray(self._data["cdc_U0"][:, i], dtype=float).mean()),
            # kappa_prep and risk0 are geography-level fixed arrays, not posterior draws.
            # kappa_prep and risk0 are geography-level fixed arrays, not posterior draws.
            kappa_prep=float(np.asarray(self._data["cdc_kappa_prep"][i], dtype=float)),
            risk0=float(np.asarray(self._data.get("cdc_risk0", np.ones(len(self.geo_names)))[i], dtype=float)),
            post_multiplier=post_multiplier,
        )

    def load_sample(self, sample_idx: int, unit_id: str) -> CDCParams:
        i = self._idx(unit_id)
        s = self._sample_idx(sample_idx)
        post = self._data.get("cdc_post_multiplier")
        post_multiplier = 1.0 if post is None else float(np.asarray(post[s, i], dtype=float))
        return CDCParams(
            beta=float(np.asarray(self._data["cdc_beta"][s, i], dtype=float)),
            alpha=float(np.asarray(self._data["cdc_alpha"][s, i], dtype=float)),
            kdx=float(np.asarray(self._data["cdc_kdx"][s, i], dtype=float)),
            U0=float(np.asarray(self._data["cdc_U0"][s, i], dtype=float)),
            kappa_prep=float(np.asarray(self._data["cdc_kappa_prep"][i], dtype=float)),
            risk0=float(np.asarray(self._data.get("cdc_risk0", np.ones(len(self.geo_names)))[i], dtype=float)),
            post_multiplier=post_multiplier,
        )
