import torch


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


def _sdpa_backward(grad, query, key, value, scale):
    from .softmax import softmax

    scores = _batched_mm(query, key.transpose(-1, -2)) * scale
    probs = softmax(scores, dim=-1)
    dvalue = _batched_mm(probs.transpose(-1, -2), grad)
    dprobs = _batched_mm(grad, value.transpose(-1, -2))
    dscores = probs * (dprobs - (probs * dprobs).sum(dim=-1, keepdim=True))
    dquery = _batched_mm(dscores, key) * scale
    dkey = _batched_mm(dscores.transpose(-1, -2), query) * scale
    return dquery, dkey, dvalue


def cudnn_attention_backward(grad_out, query, key, value, out=None, **_kwargs):
    _ = out
    scale = query.shape[-1] ** -0.5
    return _sdpa_backward(grad_out, query, key, value, scale)
