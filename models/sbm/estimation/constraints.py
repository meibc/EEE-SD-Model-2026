import numpy as np


def get_bounds(sign_mask, M, bounds_range):
    """Bounds on the free entries of J - I.

    Off-diagonal entries: ``bounds_range``, with sign-constrained pathways bounded at 0 on the
    other side. Diagonal entries: J - I >= -1, i.e. non-negative persistence (no oscillation),
    and at most bounds_range[1].
    """
    n_free = int(np.sum(M))
    sign_free = sign_mask[M == 1]
    diag_free = np.eye(M.shape[0], dtype=bool)[M == 1]

    LB = np.full(n_free, bounds_range[0], dtype=float)
    UB = np.full(n_free, bounds_range[1], dtype=float)

    LB[sign_free > 0] = 0
    UB[sign_free < 0] = 0
    LB[diag_free] = np.maximum(LB[diag_free], -1.0)

    return LB, UB
