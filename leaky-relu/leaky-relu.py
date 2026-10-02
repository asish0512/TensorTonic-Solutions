import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """
    # Write code here
    x = np.asarray(x, dtype = float)
    for i in range(len(x)):
        if x[i] < 0.0:
            x[i] = np.multiply(x[i], alpha)
    return np.array(x)