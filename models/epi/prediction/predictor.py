# models/epi/prediction/predictor.py
"""Epidemic model: incidence, diagnoses and the undiagnosed population."""

import numpy as np
from models.epi.params import POST_YEAR, RISK_BASELINE_FLOOR, RISK_RATIO_BOUNDS, EpiInputs, EpiOutput, EpiParams


class EpiPredictor:
    """Discrete-time dynamic HIV incidence and diagnosis model."""
    
    def __init__(self, params: EpiParams):
        self.params = params
    
    def predict(self, inputs: EpiInputs, unit_id: str) -> EpiOutput:
        """Run simulation."""
        p = self.params
        T = len(inputs.years)
        
        # Incidence (as calibrated, pipeline/epi_fit.py):
        # Inc_t = beta_s * no_vs_{s,t} * (1 - prep_t) * (risk_t / risk0_s) ** alpha
        prep_on_count = inputs.prep_on * inputs.N_elig
        prep_off_prop = np.clip(1.0 - inputs.prep_on, 1e-6, 1.0)

        rb = np.clip(np.asarray(inputs.risk_behavior, dtype=float), 1e-6, None)
        rb_ratio = np.clip(rb / max(float(p.risk0), RISK_BASELINE_FLOOR), *RISK_RATIO_BOUNDS)

        incidence = p.beta * inputs.no_vs * prep_off_prop * np.power(rb_ratio, p.alpha)

        post_indicator = (np.asarray(inputs.years, dtype=int) >= POST_YEAR).astype(float)
        diagnosis_multiplier = 1.0 + post_indicator * (float(p.post_multiplier) - 1.0)
        delta = 1 - np.exp(-p.kdx * inputs.hiv_testing * diagnosis_multiplier)
        
        # Initialize
        undiagnosed = np.zeros(T)
        diagnosed = np.zeros(T)
        
        undiagnosed[0] = p.U0
        diagnosed[0] = np.maximum(0, p.U0 * delta[0])
        
        # Simulate
        for t in range(1, T):
            undiagnosed[t] = np.maximum(0, undiagnosed[t-1] + incidence[t-1] - diagnosed[t-1])
            diagnosed[t] = np.maximum(0, undiagnosed[t] * delta[t])
        
        return EpiOutput(
            unit_id=unit_id,
            years=inputs.years,
            prep_on_count=prep_on_count,
            incidence=incidence,
            diagnosed=diagnosed,
            undiagnosed=undiagnosed,
        )
