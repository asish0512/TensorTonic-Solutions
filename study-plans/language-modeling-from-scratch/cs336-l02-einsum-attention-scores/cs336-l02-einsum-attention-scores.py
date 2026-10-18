import torch

def attention_scores(q: torch.Tensor, k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns scores of shape (batch, heads, query_length, key_length).
    """
    B, S_q, D = q.shape
    _, S_k, _ = k.shape
    d_h = D // num_heads

    q_h = q.reshape(B, S_q, num_heads, d_h)
    k_h = k.reshape(B, S_k, num_heads, d_h)

    A = torch.einsum("bihr,bjhr->bhij",q_h, k_h)
    return A / d_h ** 0.5