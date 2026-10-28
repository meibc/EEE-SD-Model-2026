"""Scenario PrEP from the AHS change (models/shared/prep_translation.py)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.shared.prep_translation import (  # noqa: E402
    contextual_translation, prevalence_ratio_translation, retain_prep_fraction,
)


class PrevalenceRatioTranslationTests(unittest.TestCase):
    def setUp(self):
        self.prep = np.array([0.20, 0.30])
        self.ahs = np.array([0.10, 0.20])

    def test_no_change_in_ahs_leaves_prep(self):
        np.testing.assert_allclose(prevalence_ratio_translation(self.prep, self.ahs, self.ahs, 0.55), self.prep)

    def test_no_association_leaves_prep(self):
        np.testing.assert_allclose(prevalence_ratio_translation(self.prep, self.ahs, self.ahs / 2, 1.0), self.prep)

    def test_hand_computed_example(self):
        # P0 = 0.20 / (1 - 0.10 x 0.45); halving AHS: P_s = P0 x (1 - 0.05 x 0.45)
        expected = 0.20 / (1 - 0.10 * 0.45) * (1 - 0.05 * 0.45)
        out = prevalence_ratio_translation(np.array([0.20]), np.array([0.10]), np.array([0.05]), 0.55)
        np.testing.assert_allclose(out, [expected])
        self.assertAlmostEqual(expected, 0.20471, places=5)

    def test_less_stigma_more_prep_when_ratio_below_one(self):
        out = prevalence_ratio_translation(self.prep, self.ahs, self.ahs / 2, 0.55)
        self.assertTrue(np.all(out > self.prep))

    def test_prep_among_men_without_ahs_held(self):
        r = 0.55
        out = prevalence_ratio_translation(self.prep, self.ahs, self.ahs / 2, r)
        np.testing.assert_allclose(out / (1 - self.ahs / 2 * (1 - r)), self.prep / (1 - self.ahs * (1 - r)))


class ContextualTranslationTests(unittest.TestCase):
    def test_ten_point_drop_multiplies_by_inverse_ratio(self):
        out = contextual_translation(np.array([0.20]), np.array([0.15]), np.array([0.05]), 0.80)
        np.testing.assert_allclose(out, [0.20 / 0.80])

    def test_no_change_in_ahs_leaves_prep(self):
        np.testing.assert_allclose(contextual_translation(np.array([0.2]), np.array([0.1]), np.array([0.1]), 0.7), [0.2])


class RetainPrepFractionTests(unittest.TestCase):
    def setUp(self):
        self.baseline = np.array([[0.10, 0.12], [0.20, 0.22], [0.50, 0.55]])   # AHS, PrEP, other
        self.scenario = np.array([[0.05, 0.06], [0.25, 0.30], [0.60, 0.65]])

    def test_fraction_zero_is_baseline(self):
        np.testing.assert_allclose(retain_prep_fraction(self.baseline, self.scenario, 1, 0.0), self.baseline)

    def test_fraction_one_keeps_scenario_prep_only(self):
        out = retain_prep_fraction(self.baseline, self.scenario, 1, 1.0)
        np.testing.assert_allclose(out[1], self.scenario[1])
        np.testing.assert_allclose(out[[0, 2]], self.baseline[[0, 2]])

    def test_half_is_half_the_logit_change(self):
        logit = lambda p: np.log(p / (1 - p))
        out = retain_prep_fraction(self.baseline, self.scenario, 1, 0.5)
        np.testing.assert_allclose(logit(out[1]), (logit(self.baseline[1]) + logit(self.scenario[1])) / 2)


if __name__ == "__main__":
    unittest.main()
