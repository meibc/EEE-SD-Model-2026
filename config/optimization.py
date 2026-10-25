class OptimConfig:
    """Optimization configuration (one instance per fit; specifications override attributes)."""

    wX: float = 10  # weight for trajectory fit
    wR: float = 0.01  # weight for covariance fit

    B0_diag: float = 0.5
    B0_off_diag: float = 0.2
    bounds: tuple = (-2, 2)
    # Innovation SD in the covariance target, as a fraction of each measure's SD:
    # Sigma_eta = sigma_eta_scale^2 * diag(SigmaX).
    sigma_eta_scale: float = 0.16

    # Bound optimized selected drift terms in centered-logit units per SEM step.
    drift_bounds: float = 1.5

    # Stability penalty for the discrete-time SEM transition matrix.
    w_stab: float = 1e4
    stability_eps: float = 1e-3
    stability_fd_eps: float = 1e-7

    # Evidence target on the total AHS -> PrEP response (models/sbm/calibration.py): horizon in
    # two-year steps (6 = 2024-2036) and tolerance on the centred-logit scale.
    total_effect_horizon: int = 6
    total_effect_tau: float = 0.01

    # L-BFGS-B optimizer settings. Some fits need more than 500 iterations to converge
    # (docs/analysis_decisions.md, 2026-10-23).
    maxiter: int = 5000
    ftol: float = 1e-9
    gtol: float = 1e-6
    maxls: int = 50
