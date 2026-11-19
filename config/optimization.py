from dataclasses import dataclass


@dataclass
class OptimConfig:
    """Optimization configuration (one instance per fit; specifications override fields)."""

    # Data-fit weights
    wX: float = 10.0      # trajectory fit

    # Starting values and bounds (on J - I and on the drift, centred-logit units per two-year step)
    B0_diag: float = 0.5
    B0_off_diag: float = 0.2
    bounds: tuple = (-2, 2)
    drift_bounds: float = 1.5

    # Stability: penalty w_stab x (spectral radius - (1 - stability_eps))^2 above the threshold
    w_stab: float = 1e4
    stability_eps: float = 1e-3
    stability_fd_eps: float = 1e-7

    # L-BFGS-B optimizer settings. Some fits need more than 500 iterations to converge.
    maxiter: int = 5000
    ftol: float = 1e-9
    gtol: float = 1e-6
    maxls: int = 50
