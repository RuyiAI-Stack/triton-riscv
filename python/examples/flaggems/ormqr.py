import torch

from .orgqr import orgqr


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


def ormqr(input, tau, other, left=True, transpose=False):
    q = orgqr(input[..., : tau.shape[-1]].contiguous(), tau)
    if left and not transpose:
        return _batched_mm(q, other)
    if left and transpose:
        return _batched_mm(q.transpose(-1, -2), other)
    if not left and not transpose:
        return _batched_mm(other, q)
    return _batched_mm(other, q.transpose(-1, -2))


__all__ = ["ormqr"]
