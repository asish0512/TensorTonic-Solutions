import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x = np.asarray(x, dtype = float)
    e = np.exp(-x)
    return 1 / (e+1)