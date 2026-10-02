import numpy as np

def relu(x):
    return np.maximum(0, x)

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.asarray(x)
    h = relu(x @ np.asarray(W1))
    h = relu(h @ np.asarray(W2))
    h = h @ np.asarray(W3)
    shortcut = x if Ws is None else x @ np.asarray(Ws)
    
    return relu(h + shortcut).tolist()