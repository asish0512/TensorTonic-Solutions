import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """
    g = x;
    for i in np.asarray(gradients_F):
        g = g + g @ np.asarray(i)
    return np.asarray(g)

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """
    g = x;
    for i in np.asarray(gradients_F):
        g = g @ np.asarray(i)
    return np.asarray(g)
