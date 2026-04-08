from dataclasses import dataclass
import numpy as np

@dataclass
class Unit:
    id: str
    kind: str                           # "nation", "division", "state"

    amis_years: np.ndarray                   # (T,)
    amis_values: np.ndarray                  # (m, T)
    amis_names: list[str]                    # (m,)

    cdc_years: np.ndarray | None = None         # (T_cdc,)
    cdc_values: np.ndarray | None = None        # (m, T_cdc
    cdc_names: list[str] | None = None

    sample_size: float | None = None    # For shrinkage calculation
    census_div: int | None = None       # For state → division mapping

    # Convenience accessors
    @property
    def values(self) -> np.ndarray:
        """Backward compatibility."""
        return self.amis_values
    
    @property
    def var_names(self) -> list[str]:
        """Backward compatibility."""
        return self.amis_names
    
    def get_amis(self, name: str) -> np.ndarray:
        """Get single AMIS variable by name."""
        idx = self.amis_names.index(name)
        return self.amis_values[idx, :]
    
    def get_cdc(self, name: str) -> np.ndarray | None:
        """Get single CDC variable by name."""
        if self.cdc_names is None or self.cdc_values is None:
            return None
        idx = self.cdc_names.index(name)
        return self.cdc_values[idx, :]