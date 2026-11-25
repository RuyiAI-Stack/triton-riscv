import torch

from .cross_attention import cross_attention


def _scaled_dot_product_attention_math(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    _ = dropout_p
    if is_causal:
        length = query.shape[-2]
        causal = torch.triu(
            torch.full((length, length), float("-inf"), device=query.device), diagonal=1
        )
        attn_mask = causal if attn_mask is None else attn_mask + causal
    return cross_attention(query, key, value, attn_mask=attn_mask, scale=scale)
