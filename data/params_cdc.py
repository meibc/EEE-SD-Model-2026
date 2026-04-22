# data/params_cdc.py
"""CDC model parameters loader."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import arviz as az
from .params_base import GeoIndexedParamsLoader


@dataclass
class CDCParams:
    """CDC parameters for a single geography."""
    beta: float
    alpha: float
    kdx: float
    U0: float
    kappa_prep: float


class CDCParamsLoader(GeoIndexedParamsLoader):
    """Load CDC parameters from arviz posterior."""
    
    def __init__(self, nc_path: Path, trans_npz_path: Path):
        self.nc_path = Path(nc_path)
        self.trans_npz_path = Path(trans_npz_path)
        self._cache = None
    
    def _load(self) -> dict:
        """Load and cache raw data."""
        if self._cache is not None:
            return self._cache
        
        # Load posterior
        idata = az.from_netcdf(self.nc_path)
        post = idata.posterior
        
        # Infer geo coordinate from beta_inc dimensions.
        geo_coord = post['beta_inc'].dims[-1]
        geo_names = list(post.coords[geo_coord].values)
        n_geos = len(geo_names)
        
        # Extract: (chains, draws, geos) → (samples, geos)
        beta = post['beta_inc'].values.reshape(-1, n_geos)
        alpha = post['alpha'].values.reshape(-1, n_geos)
        kdx = post['kappa_dx'].values.reshape(-1, n_geos)  # Note: kappa_dx in your file
        U0 = post['U0'].values.reshape(-1, n_geos)
        
        # Load kappa_prep from trans_npz
        trans = np.load(self.trans_npz_path, allow_pickle=True)
        outputs = trans['outputs'].item()
        years = np.asarray(trans['years'], dtype=int)
        kappa_prep = {geo: outputs[geo]['params']['kappa_prep'] for geo in outputs}
        
        self._cache = {
            'geo_names': geo_names,
            'beta': beta,
            'alpha': alpha,
            'kdx': kdx,
            'U0': U0,
            'kappa_prep': kappa_prep,
            'years': years,
        }
        return self._cache
    
    @property
    def n_samples(self) -> int:
        return self._load()['beta'].shape[0]

    @property
    def years(self) -> np.ndarray:
        return self._load()['years']
    
    def load_point_estimates(self, unit_id: str) -> CDCParams:
        """Load posterior means for one geography."""
        data = self._load()
        idx = self._get_geo_idx(unit_id)
        
        return CDCParams(
            beta=float(data['beta'][:, idx].mean()),
            alpha=float(data['alpha'][:, idx].mean()),
            kdx=float(data['kdx'][:, idx].mean()),
            U0=float(data['U0'][:, idx].mean()),
            kappa_prep=float(data['kappa_prep'].get(unit_id, 1.0)),
        )
    
    def load_sample(self, sample_idx: int, unit_id: str) -> CDCParams:
        """Load single posterior sample for one geography."""
        data = self._load()
        geo_idx = self._get_geo_idx(unit_id)
        
        return CDCParams(
            beta=float(data['beta'][sample_idx, geo_idx]),
            alpha=float(data['alpha'][sample_idx, geo_idx]),
            kdx=float(data['kdx'][sample_idx, geo_idx]),
            U0=float(data['U0'][sample_idx, geo_idx]),
            kappa_prep=float(data['kappa_prep'].get(unit_id, 1.0)),
        )
