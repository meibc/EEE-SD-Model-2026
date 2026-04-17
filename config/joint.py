from dataclasses import dataclass

@dataclass
class JointConfig:
    """Joint SEM -> CDC connector configuration."""
    hivtest_var: str = 'hivtest12'
    prep_var: str = 'prep_used'
    n_elig_var: str = 'PrEP Eligible'


