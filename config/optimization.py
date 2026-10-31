from dataclasses import dataclass


@dataclass
class OptimConfig:
    """Optimization configuration (one instance per fit; specifications override fields)."""

    # Data-fit weights
    wX: float = 10.0      # trajectory fit
    wR: float = 0.01      # covariance consistency
    # Innovation SD in the covariance target, as a fraction of each measure's SD:
    # Sigma_eta = sigma_eta_scale^2 * diag(SigmaX).
    sigma_eta_scale: float = 0.16

    # Starting values and bounds (on J - I and on the drift, centred-logit units per two-year step)
    B0_diag: float = 0.5
    B0_off_diag: float = 0.2
    bounds: tuple = (-2, 2)
    drift_bounds: float = 1.5

    # Stability: penalty w_stab x (spectral radius - (1 - stability_eps))^2 above the threshold
    w_stab: float = 1e4
    stability_eps: float = 1e-3
    stability_fd_eps: float = 1e-7

    # Evidence target on the total AHS -> PrEP response (models/dynamic/estimation/calibration.py): horizon in
    # two-year steps (6 = 2024-2036) and tolerance on the centred-logit scale.
    evidence_horizon: int = 6
    evidence_tau: float = 0.01

    # L-BFGS-B optimizer settings. Some fits need more than 500 iterations to converge.
    maxiter: int = 5000
    ftol: float = 1e-9
    gtol: float = 1e-6
    maxls: int = 50
