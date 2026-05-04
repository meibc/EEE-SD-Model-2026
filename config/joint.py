from dataclasses import dataclass

@dataclass
class JointConfig:
    """Joint SEM -> CDC connector configuration."""
    hivtest_var: str = 'hivtest12'
    prep_var: str = 'prep_used'
    risk_var: str = 'risk_behavior'
    n_elig_var: str = 'PrEP Eligible'
    prevalence_var: str = 'Estimated HIV prevalence (MSM)'
    viral_suppression_var: str = 'HIV viral suppression'

