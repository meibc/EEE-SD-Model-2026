
from dataclasses import dataclass, field
import numpy as np

CDC_YEARS = np.array([2017, 2018, 2019, 2020, 2021, 2022])

@dataclass
class JointConfig:
    """Pipeline configuration."""
    hivtest_var: str = 'hivtest12'
    prep_var: str = 'prep_on'
    n_elig_var: str = 'PrEP Eligible'
    gamma_tau: float = 1.0
    cdc_years: np.ndarray = field(default_factory=lambda: CDC_YEARS)



