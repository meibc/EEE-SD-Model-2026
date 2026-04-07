import numpy as np

class Transforms:
    @staticmethod
    def logit(p, epsilon=1e-10):
        """Logit transformation with numerical stability."""
        # Clip to avoid log(0) or log(1)
        p_clipped = np.clip(p, epsilon, 1 - epsilon)
        return np.log(p_clipped / (1 - p_clipped))
    
    @staticmethod
    def inverse_logit(x):
        """Inverse logit (sigmoid) transformation."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def logit_jacobian(p, epsilon=1e-10):
        """Jacobian of logit transformation."""
        p_clipped = np.clip(p, epsilon, 1 - epsilon)
        return np.diag(1 / p_clipped / (1 - p_clipped))