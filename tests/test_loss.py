"""The joint loss gradient matches finite differences."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dynamic import DynamicConfig  # noqa: E402
from models.dynamic.estimation.loss import joint_loss  # noqa: E402


class LossGradientTests(unittest.TestCase):
    def setUp(self):
        config = DynamicConfig()
        self.sign, self.prior_mean, self.prior_tau = config.structural_prior_arrays()
        self.drift_enabled, self.drift_mean, self.drift_tau = config.drift_prior_arrays()
        m = len(config.v_names)
        self.M = np.eye(m, dtype=int) + config.allowed_mask()
        rng = np.random.default_rng(3)
        self.Xbar = rng.normal(0, 1, (m, 4))
        self.Sigma = np.cov(rng.normal(size=(m, 60)))
        self.Sigma_eta = 0.16 ** 2 * np.diag(np.diag(self.Sigma))
        self.J_ref = 0.8 * np.eye(m)
        self.u = np.zeros(m)
        self.rng = rng

    def check(self, theta, w_stab):
        def f(t):
            return joint_loss(t, self.Xbar, self.Sigma, self.Sigma_eta, 0.01, 10.0, self.u, self.M,
                              J_ref=self.J_ref, wJ=2.0, w_stab=w_stab, eps=1e-3, stability_fd_eps=1e-7,
                              structural_prior_mean=self.prior_mean, structural_prior_tau=self.prior_tau,
                              drift_enabled=self.drift_enabled, drift_prior_mean=self.drift_mean,
                              drift_prior_tau=self.drift_tau, evidence_target=getattr(self, "evidence_target", None))
        _, grad = f(theta)
        h = 1e-6
        fd = np.array([(f(theta + h * e)[0] - f(theta - h * e)[0]) / (2 * h) for e in np.eye(theta.size)])
        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-4)

    def test_gradient_without_stability_penalty(self):
        theta = self.rng.normal(0, 0.1, int(self.M.sum()) + int(self.drift_enabled.sum()))
        self.check(theta, w_stab=0.0)

    def test_gradient_with_total_effect_target(self):
        theta = self.rng.normal(0, 0.05, int(self.M.sum()) + int(self.drift_enabled.sum()))
        self.evidence_target = {"source": 0, "target": 6, "horizon": 6, "value": -0.03, "tau": 0.05}
        self.check(theta, w_stab=0.0)

    def test_gradient_with_active_stability_penalty(self):
        # Diagonal J - I of +0.05 puts the spectral radius above the 0.999 threshold.
        JmI = np.where(np.eye(self.M.shape[0], dtype=bool), 0.05, 0.01 * self.sign)
        theta = np.concatenate([JmI[self.M == 1], np.zeros(int(self.drift_enabled.sum()))])
        self.check(theta, w_stab=1e4)


if __name__ == "__main__":
    unittest.main()
