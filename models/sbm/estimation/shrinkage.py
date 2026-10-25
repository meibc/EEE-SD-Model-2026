class ShrinkageCalculator:
    """Reference J and shrinkage weight for each unit.

    The nation is not shrunk; each division is shrunk toward the national J with
    weight wJ; each state toward its division's J with weight
    wJ_state * state_n_ref / n, where n is the state's respondent count (states
    sheet), so larger samples are shrunk less.
    """

    def __init__(self, config):
        self.config = config

    def get_params(self, unit, us_J, div_Js):
        if unit.kind == "division" and us_J is not None:
            return us_J, self.config.wJ
        if unit.kind == "state":
            n = unit.sample_size or self.config.state_n_ref
            return div_Js.get(unit.census_div), self.config.wJ_state * self.config.state_n_ref / max(n, 1.0)
        return None, None
