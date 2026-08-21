from dataclasses import dataclass

class OptimConfig: 
    """Optimization configuration"""

    # wR: float = 0.01  # weight for covariance fit
    # wX: float = 10.0  # weight for trend fit
    # wR: float = 0.01

    wX: float = 10  # weight for trend fit
    wR: float = 0.01

    B0_diag: float = 0.5
    B0_off_diag: float = 0.2
    bounds: tuple = (-2, 2)
    sigma_eta_scale: float = 0.4*2*2/10

    flag_reinitialize_JmI0: bool = True  # whether to re-initialize JmI0 for each t (instead of warm-starting from previous t's solution)

    # Loss normalization / weighting controls
    use_diag_whitening_x: bool = False
    normalize_x_by_counts: bool = False
    normalize_r_by_counts: bool = False
    normalize_components_by_baseline: bool = False
    norm_eps: float = 1e-8

    # Bound optimized selected drift terms in centered-logit units per SEM step.
    drift_bounds: float = 1.5

    # Stability penalty for the discrete-time SEM transition matrix.
    w_stab: float = 10.0
    stability_eps: float = 1e-3
    stability_fd_eps: float = 1e-7

    # L-BFGS-B optimizer settings.
    maxiter: int = 500
    ftol: float = 1e-9
    gtol: float = 1e-6
    maxls: int = 50
