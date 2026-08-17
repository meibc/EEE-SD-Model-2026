# models/epi/prediction/predictor.py
"""CDC epi model predictor."""

import numpy as np
from pipeline.results import CDCInputs, CDCOutput
from models.epi.params import CDCParams


class CDCPredictor:
    """Deterministic CDC epi model."""
    
    def __init__(self, params: CDCParams):
        self.params = params
    
    def predict(self, inputs: CDCInputs, unit_id: str) -> CDCOutput:
        """Run simulation."""
        p = self.params
        T = len(inputs.years)
        
        # Standardized v7 incidence equation.
        # Inc_t = beta_s * no_vs_{s,t} * (1 - kappa_prep_s * prep_t)
        #         * (risk_t / risk0_s) ** alpha
        prep_on_count = p.kappa_prep * inputs.prep_on * inputs.N_elig
        prep_off_prop = np.clip(1.0 - inputs.prep_on, 1e-6, 1.0)

        rb = np.clip(np.asarray(inputs.risk_behavior, dtype=float), 1e-6, None)
        rb_ratio = np.clip(rb / max(float(p.risk0), 0.05), 0.5, 3.0)

        incidence = p.beta * inputs.no_vs * prep_off_prop * np.power(rb_ratio, p.alpha)

        post_indicator = (np.asarray(inputs.years, dtype=int) >= 2021).astype(float)
        diagnosis_multiplier = 1.0 + post_indicator * (float(p.post_multiplier) - 1.0)
        delta = 1 - np.exp(-p.kdx * inputs.tau * diagnosis_multiplier)
        
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
