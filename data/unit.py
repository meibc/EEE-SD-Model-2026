from dataclasses import dataclass
import numpy as np

@dataclass
class Unit:
    id: str
    kind: str                           # "nation", "division", "state"
    values: np.ndarray                  # (m, T)
    sample_size: float | None = None    # For shrinkage calculation
    census_div: int | None = None       # For state → division mapping

    @property
    def m(self) -> int:
        return self.values.shape[0]

    @property
    def T(self) -> int:
        return self.values.shape[1]