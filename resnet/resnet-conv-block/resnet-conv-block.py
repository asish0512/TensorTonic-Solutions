import numpy as np

def relu(x):
    return np.maximum(0, x)

def conv_block(x, W1, W2, Ws):
    """
    Returns the projection residual-block output as a nested list.
    """
    return relu(relu(np.asarray(x) @ np.asarray(W1)) @ np.asarray(W2) + np.asarray(x) @ np.asarray(Ws))   