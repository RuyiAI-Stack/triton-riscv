import torch

from .cross_attention import cross_attention


def _batched_mm(a, b):
    from .mm import mm

    if a.ndim == 2 and b.ndim == 2:
        return mm(a, b)
    if b.ndim == 2:
        b = b.expand(a.shape[0], *b.shape)
    if a.ndim == 2:
        a = a.expand(b.shape[0], *a.shape)
    batch = a.reshape(-1, a.shape[-2], a.shape[-1])
    rhs = b.reshape(-1, b.shape[-2], b.shape[-1])
    if rhs.shape[0] == 1 and batch.shape[0] != 1:
        rhs = rhs.expand(batch.shape[0], -1, -1)
    if batch.shape[0] == 1 and rhs.shape[0] != 1:
        batch = batch.expand(rhs.shape[0], -1, -1)
    out = torch.empty(
        batch.shape[0],
        batch.shape[1],
        rhs.shape[-1],
        dtype=a.dtype,
        device=a.device,
    )
    for i in range(batch.shape[0]):
        out[i] = mm(batch[i], rhs[i])
    return out.reshape(a.shape[:-1] + (rhs.shape[-1],))


def _native_multi_head_attention(
    query,
    key,
    value,
    embed_dim,
    num_heads,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights=False,
    average_attn_weights=True,
):
    _ = (need_weights, average_attn_weights)
    qkv = _batched_mm(query, qkv_weight.transpose(0, 1))
    if qkv_bias is not None:
        qkv = qkv + qkv_bias
    q, k, v = qkv.chunk(3, dim=-1)
    head = embed_dim // num_heads

    def split(tensor):
        return tensor.view(tensor.shape[0], tensor.shape[1], num_heads, head).transpose(
            1, 2
        )

    out = cross_attention(split(q), split(k), split(v), attn_mask=mask)
    out = (
        out.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], embed_dim)
    )
    projected = _batched_mm(out, proj_weight.transpose(0, 1))
    if proj_bias is not None:
        projected = projected + proj_bias
    return projected
