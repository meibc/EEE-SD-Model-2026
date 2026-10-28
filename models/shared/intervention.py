import numpy as np
from config.interventions import INTERVENTION_CODEBOOK
from core.math.transforms import PROB_EPS, logit


class MeasureIntervention:
    """Change to one measure's probability, ramped linearly from start_t to end_t and held after.

    Effects:
      probability_relative_to_last_observed  move toward (1 + delta) x the measure's last observed (2024)
                                             level and hold there, never above the baseline path
      probability_gap_to_benchmark           close the fraction -delta of the gap between the current
                                             probability and the benchmark
    """

    def __init__(self, var_idx, start_t, end_t, delta, effect, benchmark=None):
        self.var_idx = var_idx
        self.start_t = start_t
        self.end_t = end_t
        self.delta = float(delta)
        self.effect = effect
        self.benchmark = benchmark
        # Probabilities at the last observed period (set by the caller for
        # "probability_relative_to_last_observed").
        self.anchor_p = None

    def _scale(self, t):
        if t < self.start_t:
            return 0.0
        if t >= self.end_t:
            return 1.0
        return float((t - self.start_t) / max(self.end_t - self.start_t, 1))

    def apply(self, X_next, t, predictor, baseline_p=None):
        X_new = X_next.copy()
        scale = self._scale(t)
        ref_logit = predictor.reference_logits
        if ref_logit is None:
            ref_logit = np.zeros(X_new.shape[0], dtype=float)
        p_current = predictor.inverse_x(X_new)[self.var_idx]

        if self.effect == "probability_relative_to_last_observed":
            # A measure projected to fall below the target follows its baseline.
            if baseline_p is None or self.anchor_p is None:
                raise ValueError("probability_relative_to_last_observed requires baseline_p and anchor_p")
            p_target = min(float(self.anchor_p[self.var_idx]) * max(0.0, 1.0 + self.delta),
                           float(baseline_p[self.var_idx]))
            p_new = np.clip((1 - scale) * float(p_current) + scale * p_target, PROB_EPS, 1 - PROB_EPS)
        elif self.effect == "probability_gap_to_benchmark":
            if self.benchmark is None:
                raise ValueError("probability_gap_to_benchmark requires benchmark")
            benchmark = float(np.clip(self.benchmark, PROB_EPS, 1 - PROB_EPS))
            gap = max(float(p_current) - benchmark, 0.0)
            closure = np.clip(-self.delta * scale, 0.0, 1.0)
            p_new = np.clip(float(p_current) - closure * gap, PROB_EPS, 1 - PROB_EPS)
        else:
            raise ValueError(f"Unknown state intervention effect: {self.effect}")

        X_new[self.var_idx] = logit(p_new, PROB_EPS) - ref_logit[self.var_idx]
        return X_new


def build_measure_interventions(unit, dynamic_years, v_names, codes, duration_steps=3):
    """Interventions for the given codes, starting after the unit's last observed period."""
    start_t = int(np.searchsorted(dynamic_years, int(unit.amis_years[-1]), side="right"))
    end_t = min(start_t + int(duration_steps), len(dynamic_years) - 1)
    if start_t >= len(dynamic_years):
        return []
    interventions = []
    for code in codes:
        spec = INTERVENTION_CODEBOOK.get(code)
        if spec is None or spec["var"] not in v_names:
            continue
        interventions.append(MeasureIntervention(var_idx=v_names.index(spec["var"]), start_t=start_t, end_t=end_t,
                                               delta=spec["delta"], effect=spec["effect"],
                                               benchmark=spec.get("benchmark")))
    return interventions
