from dataclasses import dataclass

@dataclass
class ShrinkageConfig:
    """for tuning the shrinkage of state-level estimates towards their division-level J"""

    state_size_ref: float = None  # fallback ref; auto-set from median(state n) if None
    # state shrink tuning
    state_shrink_n_ref: float = 300.0  # ref sample size for scaling
    state_shrink_n_low: float = 250.0  # below this: very strong shrink
    state_shrink_n_high: float = 350.0  # up to this: ramped-up shrink
    state_shrink_boost_super: float = 100000.0  # multiplier if n < n_low
    state_shrink_boost_mid_max: float = 50.0  # max multiplier at n = n_low (ramps down to 1 at n_high)
    state_shrink_cap: float = None  # cap on wJ_override (None to disable)

    wJ_state: float = 5.0  # base shrinkage for states → their 
    wJ: float = 10.0   # shrinkage to US J (census divisions)