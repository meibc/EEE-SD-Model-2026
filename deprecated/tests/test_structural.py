"""Checks of the AHS -> PrEP coupling strength and its channel decomposition."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.sbm.structural import (coupling_strength, source_free_counterfactual, source_history_counterfactual,  # noqa: E402
                                   trajectory_decomposition)

V = ["stigma_ahs", "stigma_gss", "stigma_family", "out_gid", "seehcp", "risk_behavior", "prep_used", "hivtest12"]
A, P = V.index("stigma_ahs"), V.index("prep_used")


def random_J(rng, draws=50):
    """Stable random matrices with the permitted sign pattern out of AHS."""
    J = rng.normal(0, 0.15, (draws, 8, 8)) + np.eye(8) * 0.6
    J[:, :, A] = 0.0
    J[:, A, A] = 0.7
    for node, sign in [("prep_used", -1), ("out_gid", -1), ("seehcp", -1), ("hivtest12", -1)]:
        J[:, V.index(node), A] = sign * rng.uniform(0, 0.3, draws)
    J /= np.maximum(1.0, np.abs(np.linalg.eigvals(J)).max(axis=1) / 0.95)[:, None, None]
    return J


class CouplingTests(unittest.TestCase):
    def test_total_is_sum_of_matrix_powers(self):
        J = random_J(np.random.default_rng(0))
        out = coupling_strength(J, V, horizon=6)
        expected = sum(np.linalg.matrix_power(J[d], h)[P, A] for h in range(1, 7) for d in [0])
        self.assertAlmostEqual(out["total"][0], expected, places=12)

    def test_channels_sum_to_total(self):
        out = coupling_strength(random_J(np.random.default_rng(1)), V, horizon=6)
        np.testing.assert_allclose(out["prep_used"] + out["engagement"], out["total"], atol=1e-12)

    def test_no_indirect_exits_means_all_direct(self):
        J = random_J(np.random.default_rng(2))
        for node in ["out_gid", "seehcp", "hivtest12"]:
            J[:, V.index(node), A] = 0.0
        out = coupling_strength(J, V, horizon=6)
        np.testing.assert_allclose(out["engagement"], 0.0, atol=1e-15)
        np.testing.assert_allclose(out["prep_used"], out["total"], atol=1e-12)

    def test_no_direct_edge_means_no_direct_channel(self):
        J = random_J(np.random.default_rng(3))
        J[:, P, A] = 0.0
        out = coupling_strength(J, V, horizon=6)
        np.testing.assert_allclose(out["prep_used"], 0.0, atol=1e-15)

    def test_rejects_exit_outside_channels(self):
        J = random_J(np.random.default_rng(4))
        J[:, V.index("risk_behavior"), A] = 0.2   # an exit not among the reported channels
        J[:, P, V.index("risk_behavior")] = 0.3
        with self.assertRaises(AssertionError):
            coupling_strength(J, V, horizon=6)


class HistoryCounterfactualTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.J = random_J(rng, draws=20)
        self.c = rng.normal(0, 0.1, (20, 8))
        self.X = rng.normal(0, 1, (20, 8, 4))

    def test_no_change_in_source_gives_no_effect(self):
        X = self.X.copy()
        X[:, A, :] = X[:, A, :1]
        f, h = source_history_counterfactual(self.J, self.c, X, V)
        np.testing.assert_allclose(f, h)

    def test_no_path_out_of_source_gives_no_effect_elsewhere(self):
        J = self.J.copy()
        J[:, :, A] = 0.0
        J[:, A, A] = 0.7
        f, h = source_history_counterfactual(J, self.c, self.X, V)
        others = [i for i in range(8) if i != A]
        np.testing.assert_allclose(f[:, others], h[:, others], atol=1e-12)

    def test_effect_is_drift_free_and_follows_matrix_powers(self):
        f, h = source_history_counterfactual(self.J, self.c, self.X, V)
        f0, h0 = source_history_counterfactual(self.J, np.zeros_like(self.c), self.X, V)
        np.testing.assert_allclose(f - h, f0 - h0, atol=1e-12)
        # One step after the first change: effect = J[:, :, A] x (change in source at t=1), then propagated.
        d1 = self.X[:, A, 1] - self.X[:, A, 0]
        expected_t2 = np.einsum("dij,dj->di", self.J, np.eye(8)[A][None] * d1[:, None])
        expected_t2[:, A] = self.X[:, A, 2] - self.X[:, A, 0]
        np.testing.assert_allclose((f - h)[:, :, 2], expected_t2, atol=1e-12)


class SourceFreeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.J = random_J(rng, draws=10)
        self.c = rng.normal(0, 0.1, (10, 8))
        self.X = rng.normal(0, 1, (10, 8, 4))

    def test_benchmark_equal_to_observed_constant_source_gives_no_gap_within_observed(self):
        X = self.X.copy()
        X[:, A, :] = 0.3
        f, h = source_free_counterfactual(self.J, self.c, X, V, benchmark=np.full(10, 0.3))
        np.testing.assert_allclose(f, h)

    def test_source_held_and_projected(self):
        f, h = source_free_counterfactual(self.J, self.c, self.X, V, benchmark=np.full(10, -1.0), extra_steps=3)
        self.assertEqual(f.shape, (10, 8, 7))
        np.testing.assert_allclose(h[:, A, :], -1.0)
        np.testing.assert_allclose(f[:, A, :4], self.X[:, A, :])
        # after the observed periods the source evolves through the fitted system
        expected = np.einsum("dj,dj->d", self.J[:, A, :], f[:, :, 4]) + self.c[:, A]
        np.testing.assert_allclose(f[:, A, 5], expected)


class TrajectoryDecompositionTests(unittest.TestCase):
    def test_parts_match_simulation(self):
        rng = np.random.default_rng(11)
        J = np.stack([np.diag(rng.uniform(0.3, 0.9, 8)) + rng.normal(0, 0.05, (8, 8)) for _ in range(3)])
        c, x0 = rng.normal(0, 0.1, (3, 8)), rng.normal(0, 0.3, (3, 8))
        x = x0.copy()
        for _ in range(6):
            x = np.einsum("dij,dj->di", J, x) + c
        out = trajectory_decomposition(J, c, x0, 6)
        np.testing.assert_allclose(out["total"], x - x0)

    def test_diagonal_system_has_no_system_parts(self):
        J = np.diag([0.5, 0.8, 1.0])[None]
        out = trajectory_decomposition(J, np.array([[0.1, -0.2, 0.3]]), np.array([[1.0, 0.5, -0.5]]), 4)
        np.testing.assert_allclose(out["carryover_system"], 0, atol=1e-12)
        np.testing.assert_allclose(out["drift_by_source"], 0, atol=1e-12)
        np.testing.assert_allclose(out["drift_own"][0, 2], 4 * 0.3)


if __name__ == "__main__":
    unittest.main()
