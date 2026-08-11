# models/epi/prediction/predictor.py
"""CDC epi model predictor."""

import numpy as np
from pipeline.results import CDCInputs, CDCOutput
from data.params_cdc import CDCParams


class CDCPredictor:
    """Deterministic CDC epi model."""
    
    def __init__(self, params: CDCParams):
        self.params = params
    
    def predict(self, inputs: CDCInputs, unit_id: str) -> CDCOutput:
        """Run simulation."""
        p = self.params
        T = len(inputs.years)
        
        # PrEP populations (bounded)
        effective_prep_cov = np.minimum(p.kappa_prep * inputs.prep_on, 0.99)
        prep_on_count = effective_prep_cov * inputs.N_elig
        prep_off_count = np.maximum(0, inputs.N_elig - prep_on_count)
        
        # Risk behavior ratio
        rb = np.asarray(inputs.risk_behavior, dtype=float)
        # Guard against invalid values from alignment/extrapolation.
        rb = np.clip(rb, 1e-6, None)
        rb_baseline = rb[0]
        rb_ratio = rb / rb_baseline
        # Optional cap to limit extreme tail inflation in Monte Carlo.
        rb_ratio = np.clip(rb_ratio, 1e-3, 50.0)
        
        # Incidence with risk behavior
        incidence = p.beta * prep_off_count * np.power(rb_ratio, p.alpha)
        delta = 1 - np.exp(-p.kdx * inputs.tau)
        
        # Initialize
        undiagnosed = np.zeros(T)
        diagnosed = np.zeros(T)
        
        undiagnosed[0] = p.U0
        diagnosed[0] = np.maximum(0, p.U0 * delta[0])
        
        # Simulate
        for t in range(1, T):
            undiagnosed[t] = np.maximum(0, undiagnosed[t-1] + incidence[t-1] - diagnosed[t-1])
            diagnosed[t] = np.maximum(0, undiagnosed[t] * delta[t])
        
        return CDCOutput(
            unit_id=unit_id,
            years=inputs.years,
            prep_on_count=prep_on_count,
            incidence=incidence,
            diagnosed=diagnosed,
            undiagnosed=undiagnosed,
        )
