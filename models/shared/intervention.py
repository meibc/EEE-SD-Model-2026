import numpy as np

class StateIntervention:
    """
    Operates in X (logit) space by default.

    delta can be:
      - float (constant logit shift)
      - callable(t, X_next, X_prev) -> float

    space:
      - "logit" (default)
      - "prob" (optional: applies in probability space)
    """

    def __init__(self, var_idx, start_t, end_t, delta, mode="linear", space="logit"):
        self.var_idx = var_idx
        self.start_t = start_t
        self.end_t = end_t
        self.delta = delta
        self.mode = mode
        self.space = space

    def _scale(self, t):
        if t < self.start_t:
            return 0.0
        if t >= self.end_t:
            return 1.0

        span = max(self.end_t - self.start_t, 1)
        frac = (t - self.start_t) / span

        if self.mode == "linear":
            return frac
        elif self.mode == "step":
            return 1.0
        elif self.mode == "sigmoid":
            return 1 / (1 + np.exp(-10 * (frac - 0.5)))
        else:
            return frac

    def apply(self, X_next, X_prev, t, transforms=None):
        # copy to avoid mutation bugs
        X_new = X_next.copy()

        # compute adjustment
        if callable(self.delta):
            adj = self.delta(t, X_new, X_prev)
        else:
            adj = self.delta * self._scale(t)

        # apply
        if self.space == "logit":
            X_new[self.var_idx] += adj

        elif self.space == "prob":
            if transforms is None:
                raise ValueError("Transforms required for prob-space intervention")

            y = transforms.inverse_logit(X_new[self.var_idx])
            y = np.clip(y + adj, 1e-6, 1 - 1e-6)
            X_new[self.var_idx] = transforms.logit(y)

        return X_new

class RelationshipIntervention:
    """
    Applies intervention to system structure (J matrix).
    """

    def __init__(
        self,
        i: int,
        j: int,
        start_t: int,
        end_t: int,
        delta: float,
        mode: str = "linear",
    ):
        self.i = i  # target
        self.j = j  # source
        self.start_t = start_t
        self.end_t = end_t
        self.delta = delta
        self.mode = mode

    def _scale(self, t):
        if t < self.start_t:
            return 0.0
        if t >= self.end_t:
            return 1.0

        span = max(self.end_t - self.start_t, 1)
        frac = (t - self.start_t) / span

        if self.mode == "linear":
            return frac
        elif self.mode == "step":
            return 1.0
        return frac

    def apply(self, J_base, t):
        J_new = J_base.copy()

        adj = self.delta * self._scale(t)
        J_new[self.i, self.j] += adj

        return J_new
    

INTERVENTION_CODEBOOK = {
    "reduce_gss": {
        "var": "stigma_gss",
        "delta": -1.0,
        "mode": "linear",
        "space": "logit",
        "label": "Reduce general social stigma",
    },
    "reduce_family_stigma": {
        "var": "stigma_family",
        "delta": -1.0,
        "mode": "linear",
        "space": "logit",
    },
    "reduce_ahs": {
        "var": "stigma_ahs",
        "delta": -1.0,
        "mode": "linear",
        "space": "logit",
    },
    "increase_seehcp": {
        "var": "seehcp",
        "delta": 0.3,
        "mode": "linear",
        "space": "logit",
    },
    "reduce_risk": {
        "var": "risk_behavior",
        "delta": -0.5,
        "mode": "linear",
        "space": "logit",
    },
}

REL_CODEBOOK = {
    "weaken_stigma_to_care": {
        "from": "stigma_ahs",
        "to": "seehcp",
        "delta": -1.5,
    },
    "weaken_stigma_to_prep": {
        "from": "stigma_ahs",
        "to": "prep_used",
        "delta": -0.5,
    },
    "strengthen_care_feedback": {
        "from": "seehcp",
        "to": "stigma_ahs",
        "delta": -0.2,
    },
    'strengthen_prep_feedback': {
        "from": "prep_used",
        "to": "stigma_ahs",
        "delta": -0.2,
    },
    'weaken_stigma_to_hivtest': {
        "from": "stigma_ahs",
        "to": "hivtest12",
        "delta": -0.5,
    }
}

def build_state_interventions(
    unit,
    sem_years,
    v_names,
    codes,
    duration_steps=3,
):
    last_obs_year = int(unit.amis_years[-1])
    start_t = int(np.searchsorted(sem_years, last_obs_year, side="right"))

    if start_t >= len(sem_years):
        return []

    end_t = min(start_t + duration_steps, len(sem_years) - 1)

    interventions = []

    for code in codes:
        spec = INTERVENTION_CODEBOOK.get(code)
        if spec is None:
            continue

        var = spec["var"]
        if var not in v_names:
            continue

        interventions.append(
            StateIntervention(
                var_idx=v_names.index(var),
                start_t=start_t,
                end_t=end_t,
                delta=spec["delta"],
                mode=spec.get("mode", "linear"),
                space=spec.get("space", "logit"),
            )
        )

    return interventions

def build_relationship_interventions(
    unit,
    sem_years,
    v_names,
    codes,
    duration_steps=3,
):
    last_obs_year = int(unit.amis_years[-1])
    start_t = int(np.searchsorted(sem_years, last_obs_year, side="right"))

    if start_t >= len(sem_years):
        return []

    end_t = min(start_t + duration_steps, len(sem_years) - 1)

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
            )
        )

    return interventions