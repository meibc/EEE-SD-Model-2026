import numpy as np

def get_bounds(sign_mask, M, bounds_range):
    """Parameter bounds respecting sign constraints."""
    n_free = int(np.sum(M))
    sign_free = sign_mask[M == 1]
    
    LB = np.full(n_free, bounds_range[0])
    UB = np.full(n_free, bounds_range[1])
    
    LB[sign_free > 0] = 0
    UB[sign_free < 0] = 0
    
    return LB, UB