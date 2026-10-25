import numpy as np
from config.interventions import INTERVENTION_CODEBOOK, REL_CODEBOOK


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class StateIntervention:
    """State intervention in centered-logit simulation space."""

    def __init__(self, var_idx, start_t, end_t, delta, mode="linear", effect="centered_logit_additive", benchmark=None):
        self.var_idx = var_idx
        self.start_t = start_t
        self.end_t = end_t
        self.delta = float(delta)
        self.mode = mode
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
        frac = (t - self.start_t) / max(self.end_t - self.start_t, 1)
        if self.mode == "step":
            return 1.0
        if self.mode == "sigmoid":
            return 1 / (1 + np.exp(-10 * (frac - 0.5)))
        return float(frac)

    def apply(self, X_next, X_prev, t, predictor, baseline_p=None):
        X_new = X_next.copy()
        scale = self._scale(t)
        ref_logit = predictor.reference_logits
        if ref_logit is None:
            ref_logit = np.zeros(X_new.shape[0], dtype=float)

        if self.effect == "centered_logit_additive":
            X_new[self.var_idx] += self.delta * scale
            return X_new

        if self.effect == "probability_relative_shift":
            # Shift, not pin: lower the measure's set point so that, with no feedback, it settles
            # at (1 + delta) x baseline; each step adds (1 - persistence) x the logit gap, and
            # the measure keeps responding to its drivers (feedback included).
            if baseline_p is None or predictor.current_J is None:
                raise ValueError("probability_relative_shift requires baseline_p and the current J")
            b = float(np.clip(baseline_p[self.var_idx], 1e-6, 1 - 1e-6))
            gap = _logit(b * max(0.0, 1.0 + self.delta)) - _logit(b)
            X_new[self.var_idx] += scale * (1.0 - predictor.current_J[self.var_idx, self.var_idx]) * gap
            return X_new

        p_current = predictor._inverse_x(X_new)[self.var_idx]

        if self.effect == "probability_relative":
            multiplier = max(0.0, 1.0 + self.delta * scale)
            p_new = np.clip(p_current * multiplier, 1e-6, 1 - 1e-6)
        elif self.effect == "probability_relative_to_baseline":
            if baseline_p is None:
                raise ValueError("probability_relative_to_baseline requires baseline_p")
            multiplier = max(0.0, 1.0 + self.delta)
            p_target = np.clip(float(baseline_p[self.var_idx]) * multiplier, 1e-6, 1 - 1e-6)
            p_new = np.clip((1 - scale) * float(p_current) + scale * p_target, 1e-6, 1 - 1e-6)
        elif self.effect == "probability_absolute_to_baseline":
            # Fixed dose: move toward the baseline probability minus a fixed amount.
            if baseline_p is None:
                raise ValueError("probability_absolute_to_baseline requires baseline_p")
            p_target = np.clip(float(baseline_p[self.var_idx]) + self.delta, 1e-6, 1 - 1e-6)
            p_new = np.clip((1 - scale) * float(p_current) + scale * p_target, 1e-6, 1 - 1e-6)
        elif self.effect == "probability_relative_to_last_observed":
            # Move toward (1 + delta) x the last observed level and hold there; never above the
            # baseline path, so a measure projected to fall below the target follows its baseline.
            if baseline_p is None or self.anchor_p is None:
                raise ValueError("probability_relative_to_last_observed requires baseline_p and anchor_p")
            p_target = min(float(self.anchor_p[self.var_idx]) * max(0.0, 1.0 + self.delta),
                           float(baseline_p[self.var_idx]))
            p_new = np.clip((1 - scale) * float(p_current) + scale * p_target, 1e-6, 1 - 1e-6)
        elif self.effect == "probability_gap_to_benchmark":
            if self.benchmark is None:
                raise ValueError("probability_gap_to_benchmark requires benchmark")
            benchmark = float(np.clip(self.benchmark, 1e-6, 1 - 1e-6))
            gap = max(float(p_current) - benchmark, 0.0)
            closure = np.clip(-self.delta * scale, 0.0, 1.0)
            p_new = np.clip(float(p_current) - closure * gap, 1e-6, 1 - 1e-6)
        else:
            raise ValueError(f"Unknown state intervention effect: {self.effect}")

        X_new[self.var_idx] = _logit(p_new) - ref_logit[self.var_idx]
        return X_new


class RelationshipIntervention:
    """Time-varying intervention on one J matrix entry."""

    def __init__(
        self,
        i: int,
        j: int,
        start_t: int,
        end_t: int,
        delta: float,
        mode: str = "linear",
        effect: str = "additive",
        target=None,
    ):
        self.i = i
        self.j = j
        self.start_t = start_t
        self.end_t = end_t
        self.delta = float(delta)
        self.mode = mode
        self.effect = effect
        self.target = None if target is None else float(target)

    def _scale(self, t):
        if t < self.start_t:
            return 0.0
        if t >= self.end_t:
            return 1.0
        return float((t - self.start_t) / max(self.end_t - self.start_t, 1))

    def apply(self, J_base, t):
        J_new = J_base.copy()
        scale = self._scale(t)
        if self.effect == "attenuate":
            attenuation = np.clip(self.delta * scale, 0.0, 1.0)
            J_new[self.i, self.j] = J_base[self.i, self.j] * (1 - attenuation)
        elif self.effect == "attenuate_positive":
            attenuation = np.clip(self.delta * scale, 0.0, 1.0)
            current = J_base[self.i, self.j]
            J_new[self.i, self.j] = current * (1 - attenuation) if current > 0 else current
        elif self.effect == "positive_to_target":
            if self.target is None:
                raise ValueError("positive_to_target requires target")
            move = np.clip(self.delta * scale, 0.0, 1.0)
            current = J_base[self.i, self.j]
            J_new[self.i, self.j] = (1 - move) * current + move * self.target if current > 0 else current
        elif self.effect == "multiplicative":
            J_new[self.i, self.j] = J_base[self.i, self.j] * (1 + self.delta * scale)
        elif self.effect == "additive":
            J_new[self.i, self.j] += self.delta * scale
        else:
            raise ValueError(f"Unknown relationship intervention effect: {self.effect}")
        return J_new


def _intervention_window(unit, sem_years, duration_steps=3):
    last_obs_year = int(unit.amis_years[-1])
    start_t = int(np.searchsorted(sem_years, last_obs_year, side="right"))
    end_t = min(start_t + int(duration_steps), len(sem_years) - 1)
    return start_t, end_t


def build_state_interventions(unit, sem_years, v_names, codes, duration_steps=3):
    start_t, end_t = _intervention_window(unit, sem_years, duration_steps)
    if start_t >= len(sem_years):
        return []
    interventions = []
    for code in codes:
        spec = INTERVENTION_CODEBOOK.get(code)
        if spec is None or spec["var"] not in v_names:
            continue
        interventions.append(
            StateIntervention(
                var_idx=v_names.index(spec["var"]),
                start_t=start_t,
                end_t=end_t,
                delta=spec["delta"],
                mode=spec.get("mode", "linear"),
                effect=spec.get("effect", "centered_logit_additive"),
                benchmark=spec.get("benchmark"),
            )
        )
    return interventions


def build_relationship_interventions(unit, sem_years, v_names, codes, duration_steps=3):
    start_t, end_t = _intervention_window(unit, sem_years, duration_steps)
    if start_t >= len(sem_years):
        return []
    interventions = []
    for code in codes:
        spec = REL_CODEBOOK.get(code)
        if spec is None:
            continue
        src = spec["from"]
        tgt = spec["to"]
        if src not in v_names or tgt not in v_names:
            continue
        interventions.append(
            RelationshipIntervention(
                i=v_names.index(tgt),
                j=v_names.index(src),
                start_t=start_t,
                end_t=end_t,
                delta=spec["delta"],
                mode=spec.get("mode", "linear"),
                effect=spec.get("effect", "additive"),
                target=spec.get("target"),
            )
        )
    return interventions
