from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SEMConfig:
    """SEM model configuration."""

    # Data
    data_path: Path = Path("Factor Analysis Final.xlsx")

    # Variables
    v_names: list[str] = field(default_factory=lambda: [
        'stigma_ahs', 'stigma_gss', 'stigma_family',
        'out_gid', 'seehcp', 'risk_behavior', 'prep_used',
        'hivtest12'
    ])

    # Simulation
    n_steps: int = 6
