import numpy as np
from config.interventions import INTERVENTION_CODEBOOK
from core.transforms import PROB_EPS, logit


def intervention_window(unit, dynamic_years, duration_steps):
    """(start, end) steps of the intervention ramp: from the first year after the unit's last observed period to
    full strength duration_steps two-year steps later."""
    start = int(np.searchsorted(dynamic_years, int(unit.amis_years[-1]), side="right"))
    return start, min(start + int(duration_steps), len(dynamic_years) - 1)


def ramp(t, start, end):
    """Fraction of full strength in step t: 0 before start, linear to 1 at end, 1 after."""
    if t < start:
        return 0.0
    if t >= end:
        return 1.0
    return float((t - start) / max(end - start, 1))


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

    def apply(self, X_next, t, transformer, baseline_p=None):
        X_new = X_next.copy()
        scale = ramp(t, self.start_t, self.end_t)
        ref_logit = transformer.reference_logits
        if ref_logit is None:
            ref_logit = np.zeros(X_new.shape[0], dtype=float)
        p_current = transformer.inverse_transform(X_new)[self.var_idx]

        if self.effect == "probability_relative_to_last_observed":
            # A measure projected to fall below the target follows its baseline.
            if baseline_p is None or self.anchor_p is None:
                raise ValueError("probability_relative_to_last_observed requires baseline_p and anchor_p")
            p_target = min(float(self.anchor_p[self.var_idx]) * max(0.0, 1.0 + self.delta),
                           float(baseline_p[self.var_idx]))
            p_new = np.clip((1 - scale) * float(p_current) + scale * p_target, PROB_EPS, 1 - PROB_EPS)
        elif self.effect == "probability_relative_to_baseline":
            # Change relative to the measure's own baseline path (e.g. +0.6 = 60% higher than it would be).
            if baseline_p is None:
                raise ValueError("probability_relative_to_baseline requires baseline_p")
            p_target = float(baseline_p[self.var_idx]) * max(0.0, 1.0 + self.delta)
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
    start_t, end_t = intervention_window(unit, dynamic_years, duration_steps)
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
