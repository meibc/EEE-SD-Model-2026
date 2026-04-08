"""Prediction functions."""
import numpy as np
from core.math.transforms import Transforms


class Predictor:
    def __init__(self):
        self.transforms = Transforms()

    def predict_next(self, J, X_current, u):
        """One-step prediction: X_t = J @ X_{t-1} + u"""
        return J @ X_current + u

    def predict_rolling(self, J_sequence, Xbar, u) -> tuple[np.ndarray, np.ndarray]:
        """
        Rolling one-step-ahead predictions (for validation).
        Uses OBSERVED X at t-1 to predict t.

        Args:
            J_sequence: (m, m, T) - J estimates at each time
            Xbar: (m, T) - observed states in X space
            u: (m,) - constant input

        Returns:
            Ypred: (m, T+1) - predictions in probability space
            Xpred: (m, T+1) - predictions in logit space
        """
        m, T = Xbar.shape
        Xpred = np.full((m, T + 1), np.nan)
        
        # Bootstrap first two points with observed
        t_start = 2
        Xpred[:, :t_start] = Xbar[:, :t_start]

        # Rolling: predict t using observed t-1
        for t in range(2, T + 1):
            J_t = J_sequence[:, :, t - 1]
            Xpred[:, t] = self.predict_next(J_t, Xbar[:, t - 1], u)

        Ypred = self.transforms.inverse_logit(Xpred)
        return Ypred, Xpred

    def predict_trajectory(self, J, X0, u, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Recursive trajectory simulation (for forecasting).
        Uses PREDICTED X at t-1 to predict t.

        Args:
            J: (m, m) - single J matrix (typically final estimate)
            X0: (m,) - initial state in X space
            u: (m,) - constant input
            n_steps: Number of steps to simulate

        Returns:
            Ypred: (m, n_steps) - predictions in probability space
            Xpred: (m, n_steps) - predictions in logit space
        """
        m = X0.shape[0]
        Xpred = np.zeros((m, n_steps))
        Xpred[:, 0] = X0

        for t in range(1, n_steps):
            Xpred[:, t] = self.predict_next(J, Xpred[:, t - 1], u)

        Ypred = self.transforms.inverse_logit(Xpred)
        return Ypred, Xpred
