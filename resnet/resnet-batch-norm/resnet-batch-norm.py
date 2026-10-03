import numpy as np

def batch_norm (x, gamma, beta):
    x_norm = (x - x.mean(axis = 0))/np.sqrt(x.var(axis = 0) + 0.00001)
    return gamma * x_norm + beta


def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    if mode == 'post':
        out = x @ W1
        out = batch_norm(out, gamma1, beta1)
        out = np.maximum(out, 0)
        out = out @ W2
        out = batch_norm(out, gamma2, beta2)
        out = out + x
        out = np.maximum(out, 0)
    else:
        out = batch_norm(x, gamma1, beta1)
        out = np.maximum(out, 0)
        out =  out @ W1
        out = batch_norm(out, gamma2, beta2)
        out = np.maximum(out, 0)
        out = out @ W2
        out = out + x
    return {"output": out, "mode": mode}


        

