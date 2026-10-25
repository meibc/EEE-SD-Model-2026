from dataclasses import dataclass


@dataclass
class ShrinkageConfig:
    """Weights of the shrinkage of each unit's J toward its parent's J."""

    wJ: float = 10.0            # division -> national J
    wJ_state: float = 5.0       # state -> division J, at state_n_ref respondents
    state_n_ref: float = 300.0  # state weight = wJ_state * state_n_ref / n
