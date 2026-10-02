import numpy as np

def relu(x):
    return np.maximum(0, x)

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    return relu(relu(np.asarray(x) @ np.asarray(W1).T) @ np.asarray(W2).T + x)   