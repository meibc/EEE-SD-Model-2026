"""State anchoring, evidence-calibrated prior centres, placebo permutation, sign-free bounds and
shifted interventions."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dynamic import DynamicConfig  # noqa: E402
from models.dynamic.estimation.calibration import (  # noqa: E402
    evidence_slope,
    sustained_shift_response,
    sustained_shift_response_grad,
)
from models.dynamic.estimation.jacobian import get_bounds  # noqa: E402
from models.dynamic.estimation.transforms import FeatureTransformer  # noqa: E402
from models.dynamic.prediction.predictor import Predictor  # noqa: E402
from models.shared.intervention import MeasureIntervention  # noqa: E402
from pipeline.dynamic_fit import _permute_periods  # noqa: E402


class AnchorTests(unittest.TestCase):
    def test_anchored_logits_average_zero_and_invert(self):
        Y = np.array([[0.10, 0.12, 0.15, 0.20], [0.30, 0.25, 0.40, 0.35]])
        anchor = FeatureTransformer.state_anchor(Y)
        t = FeatureTransformer(reference_logits=anchor)
        X, _ = t.transform(Y)
        np.testing.assert_allclose(X.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(t.inverse_transform(X), Y, atol=1e-12)

    def test_covariance_unchanged_by_anchor(self):
        Y = np.array([[0.10, 0.12, 0.15, 0.20], [0.30, 0.25, 0.40, 0.35]])
        S = np.array([[0.05, 0.01], [0.01, 0.08]])
        _, a = FeatureTransformer(reference_logits=FeatureTransformer.state_anchor(Y)).transform(Y, S)
        _, b = FeatureTransformer(reference_probs=np.array([0.2, 0.3])).transform(Y, S)
        np.testing.assert_allclose(a, b)


class CalibrationTests(unittest.TestCase):
    def test_no_association_gives_zero(self):
        self.assertAlmostEqual(evidence_slope(0.15, 0.3, 1.0, 0.06), 0.0, places=12)

    def test_slope_reproduces_probability_change(self):
        ahs, prep, apr, bench = 0.15, 0.30, 0.7, 0.06
        slope = evidence_slope(ahs, prep, apr, bench)
        p0 = prep / (1 - ahs * (1 - apr))
        logit = lambda q: np.log(q / (1 - q))
        target = logit(p0 * (1 - bench * (1 - apr)))
        self.assertAlmostEqual(logit(prep) + slope * (logit(bench) - logit(ahs)), target, places=12)
        self.assertLess(slope, 0)

    def test_shift_response_matches_simulation(self):
        rng = np.random.default_rng(5)
        J = np.diag(rng.uniform(0.3, 0.8, 8)) + rng.normal(0, 0.05, (8, 8))
        d = np.zeros(8)
        for _ in range(6):
            d = J @ d + np.eye(8)[0]
        self.assertAlmostEqual(sustained_shift_response(J, 0, 6, 6), d[6] / d[0], places=12)
        J[:, 0] = 0.0
        J[0, 0] = 0.5
        self.assertAlmostEqual(sustained_shift_response(J, 0, 6, 6), 0.0, places=12)


class EvidenceDrawTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from types import SimpleNamespace
        import pandas as pd
        self.tmp = Path(tempfile.mkdtemp()) / "evidence.csv"
        pd.DataFrame({"state": ["AA", "POOLED"], "log_apr_pooled": [np.log(0.5), np.log(0.7)],
                      "se_pooled": [0.2, 0.1]}).to_csv(self.tmp, index=False)
        self.config = DynamicConfig()
        v = self.config.v_names
        values = np.full((len(v), 4), 0.2)
        values[v.index("stigma_ahs")] = 0.12
        self.units = [SimpleNamespace(id=uid, amis_values=values) for uid in ("AA", "BB")]

    def test_state_ratio_and_fallback_to_pooled(self):
        import pandas as pd
        from pipeline.dynamic_fit import evidence_slopes
        exact = self.tmp.with_name("evidence_exact.csv")          # zero SEs: draws equal the pooled ratios
        pd.read_csv(self.tmp).assign(se_pooled=0.0).to_csv(exact, index=False)
        _, apr = evidence_slopes(self.units, self.config, self.config.v_names, np.arange(3), exact)
        np.testing.assert_allclose(apr, np.tile([0.5, 0.7], (3, 1)))

    def test_draws_paired_by_source_index(self):
        from pipeline.dynamic_fit import evidence_slopes
        s1, a1 = evidence_slopes(self.units, self.config, self.config.v_names, np.array([4, 9]), self.tmp)
        s2, a2 = evidence_slopes(self.units, self.config, self.config.v_names, np.array([9]), self.tmp)
        np.testing.assert_array_equal(a1[1], a2[0])
        np.testing.assert_array_equal(s1[1], s2[0])
        self.assertFalse(np.allclose(a1[0], a1[1]))


class ShiftResponseGradientTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(6)
        J = np.diag(rng.uniform(0.3, 0.8, 8)) + rng.normal(0, 0.08, (8, 8))
        r, g = sustained_shift_response_grad(J, 0, 6, 6)
        self.assertAlmostEqual(r, sustained_shift_response(J, 0, 6, 6), places=12)
        h = 1e-6
        fd = np.array([[(sustained_shift_response(J + h * np.outer(np.eye(8)[i], np.eye(8)[j]), 0, 6, 6)
                         - sustained_shift_response(J - h * np.outer(np.eye(8)[i], np.eye(8)[j]), 0, 6, 6)) / (2 * h)
                        for j in range(8)] for i in range(8)])
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-8)


class PlaceboTests(unittest.TestCase):
    def test_only_named_rows_permuted_and_never_identity(self):
        Y = np.arange(32, dtype=float).reshape(8, 4)
        for seed in range(30):
            out = _permute_periods(Y, [0, 3], np.random.default_rng(seed))
            np.testing.assert_array_equal(np.delete(out, [0, 3], axis=0), np.delete(Y, [0, 3], axis=0))
            self.assertFalse(np.array_equal(out[0], Y[0]))
            np.testing.assert_array_equal(np.sort(out[0]), Y[0])


class SignFreeTests(unittest.TestCase):
    def test_sign_free_pathways_have_both_bounds(self):
        config = DynamicConfig()
        sign, _, _ = config.structural_prior_arrays()
        M = np.eye(8, dtype=int) + config.allowed_mask()
        LB, UB = get_bounds(sign, M, (-2, 2))
        v = config.v_names
        free = np.zeros((8, 8))
        free[M == 1] = np.arange(int(M.sum()))
        for src in ("seehcp", "prep_used"):
            k = int(free[v.index("stigma_ahs"), v.index(src)])
            self.assertEqual((LB[k], UB[k]), (-2, 2))

    def test_persistence_non_negative(self):
        config = DynamicConfig()
        sign, _, _ = config.structural_prior_arrays()
        M = np.eye(8, dtype=int) + config.allowed_mask()
        LB, _ = get_bounds(sign, M, (-2, 2))
        diag = np.eye(8, dtype=bool)[M == 1]
        np.testing.assert_array_equal(LB[diag], -1.0)


class LastObservedCutTests(unittest.TestCase):
    def run_cut(self, baseline_level):
        pred = Predictor(reference_logits=np.zeros(2))
        iv = MeasureIntervention(var_idx=0, start_t=1, end_t=1, delta=-0.5, effect="probability_relative_to_last_observed")
        iv.anchor_p = np.array([0.2, 0.5])                       # last observed level
        baseline = np.full((2, 10), 0.5)
        baseline[0] = baseline_level
        probs, _ = pred.predict_trajectory(np.diag([0.6, 0.8]), np.zeros(2), np.zeros(2), 10,
                                           state_interventions=[iv], baseline_probs=baseline)
        return probs[0, 1:]

    def test_held_at_half_the_last_observed_level(self):
        np.testing.assert_allclose(self.run_cut(0.4), 0.1, atol=1e-9)    # baseline rises; cut holds at 0.1

    def test_never_above_baseline(self):
        np.testing.assert_allclose(self.run_cut(0.05), 0.05, atol=1e-9)  # baseline falls below target


if __name__ == "__main__":
    unittest.main()
