"""Projection starting point: observed periods are held fixed, then propagated."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dynamic.prediction.predictor import Predictor  # noqa: E402


class FixedPrefixTests(unittest.TestCase):
    def setUp(self):
        self.J = np.array([[0.9, 0.0], [0.1, 0.8]])
        self.u = np.array([0.05, 0.0])
        self.prefix = np.array([[0.0, 0.2, 0.5, 0.9], [1.0, 0.7, 0.4, 0.1]])

    def test_prefix_is_held_and_projection_starts_from_last_column(self):
        _, X = Predictor().predict_trajectory(self.J, self.prefix[:, 0], self.u, 7, fixed_prefix=self.prefix)
        np.testing.assert_allclose(X[:, :4], self.prefix)
        np.testing.assert_allclose(X[:, 4], self.J @ self.prefix[:, 3] + self.u)
        np.testing.assert_allclose(X[:, 6], self.J @ (self.J @ X[:, 4] + self.u) + self.u)

    def test_without_prefix_propagates_from_first_period(self):
        _, X = Predictor().predict_trajectory(self.J, self.prefix[:, 0], self.u, 4)
        np.testing.assert_allclose(X[:, 1], self.J @ self.prefix[:, 0] + self.u)


if __name__ == "__main__":
    unittest.main()
