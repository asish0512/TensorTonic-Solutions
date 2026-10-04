import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.asarray(x)
    conv1, W1_b1, W2_b1 = map(np.asarray, (conv1, W1_b1, W2_b1))
    W1_b2, W2_b2, Ws_b2, fc = map(np.asarray, (W1_b2, W2_b2, Ws_b2, fc))

    relu = lambda a: np.maximum(0, a)

    # Stem
    h = relu(x @ conv1)

    # Block 1: identity shortcut (shapes match)
    out = relu(h @ W1_b1)
    out = out @ W2_b1
    h = relu(out + h)

    # Block 2: projection shortcut (dimension change)
    out = relu(h @ W1_b2)
    out = out @ W2_b2
    h = relu(out + h @ Ws_b2)

    # Classifier head
    return (h @ fc).tolist()