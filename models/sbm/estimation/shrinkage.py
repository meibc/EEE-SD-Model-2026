import numpy as np

class ShrinkageCalculator:
    def __init__(self, config):
        self.config = config

    def get_params(self, unit, us_J, div_Js):
        J_ref = None
        wJ = None

        if unit.id in ("USA", "US"):
            return None, None

        elif unit.kind == "division" and us_J is not None:
            J_ref = us_J
            wJ = self.config.wJ

        elif unit.kind == "state":
            div_id = unit.census_div

            if div_id in div_Js:
                J_ref = div_Js[div_id]

            wJ = self._compute_weight(unit)

        return J_ref, wJ
    
    def _compute_weight(self, unit):
        config = self.config

        n_ref = (
            config.state_shrink_n_ref
            or config.state_size_ref
            or 1.0
        )
        n_unit = unit.sample_size or n_ref

        base_wJ = config.wJ_state * (n_ref / max(n_unit, 1.0))

        n_low = config.state_shrink_n_low
        n_high = config.state_shrink_n_high

        if n_unit < n_low:
            wJ = max(
                base_wJ * config.state_shrink_boost_super,
                config.state_shrink_boost_super * config.wJ_state,
            )
        elif n_unit < n_high:
            frac = (n_high - n_unit) / max(n_high - n_low, 1e-6)
            boost = 1 + frac * (config.state_shrink_boost_mid_max - 1)
            wJ = base_wJ * boost
        else:
            wJ = base_wJ

        if config.state_shrink_cap is not None:
            wJ = min(wJ, config.state_shrink_cap)

        return wJ
