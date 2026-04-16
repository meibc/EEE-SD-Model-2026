from __future__ import annotations

"""Shared helpers for parameter loaders keyed by geography."""


class GeoIndexedParamsLoader:
    """
    Base helper for loaders that expose `geo_names` and index by unit id.
    Subclasses must implement `_load()` returning a dict with `geo_names`.
    """

    def _load(self) -> dict:
        raise NotImplementedError

    @property
    def geo_names(self) -> list[str]:
        return self._load()["geo_names"]

    def _get_geo_idx(self, unit_id: str) -> int:
        names = self.geo_names
        if unit_id in names:
            return names.index(unit_id)
        raise ValueError(f"Unit '{unit_id}' not found. Available: {names}")

