"""Completeness and layout checks for the model-input draws."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from input.state_input_draws import dense_array, validate_complete  # noqa: E402


class InputDrawTests(unittest.TestCase):
    def setUp(self):
        self.units = ["USA", "div_1", "MA", "CT"]
        self.years = np.array([2018, 2020])
        rows = [(d, u, y, v, 0.1 * (d + 1) + (0.01 if u == "div_1" else 0))
                for d in (1, 2) for u in self.units for y in self.years for v in ("a", "b")
                if not (u == "CT" and y == 2020)]
        self.data = pd.DataFrame(rows, columns=["draw", "unit", "year", "variable", "model_draw"])

    def test_missing_state_cells_are_rejected(self):
        values = dense_array(self.data, [1, 2], self.units, ["a", "b"], self.years)
        self.assertTrue(np.isnan(values[..., 1, 3]).all())
        with self.assertRaisesRegex(ValueError, "incomplete after R preparation"):
            validate_complete(values)

    def test_complete_inputs_are_accepted(self):
        complete = self.data.copy()
        missing_rows = [
            (draw, "CT", 2020, variable, 0.1 * (draw + 1))
            for draw in (1, 2)
            for variable in ("a", "b")
        ]
        complete = pd.concat(
            [
                complete,
                pd.DataFrame(
                    missing_rows,
                    columns=["draw", "unit", "year", "variable", "model_draw"],
                ),
            ],
            ignore_index=True,
        )
        values = dense_array(complete, [1, 2], self.units, ["a", "b"], self.years)
        validate_complete(values)

    def test_unknown_labels_rejected(self):
        bad = self.data.assign(unit=self.data["unit"].replace("MA", "XX"))
        with self.assertRaisesRegex(ValueError, "unit"):
            dense_array(bad, [1, 2], self.units, ["a", "b"], self.years)


if __name__ == "__main__":
    unittest.main()
