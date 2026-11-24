import torch

from .inverse import inverse


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


def _eye_like_batch(A):
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    return eye.expand(A.shape).contiguous()


def matrix_power(A, n):
    if n == 0:
        return _eye_like_batch(A)
    base = inverse(A) if n < 0 else A
    exp = abs(int(n))
    result = None
    while exp:
        if exp & 1:
            result = base if result is None else _batched_mm(result, base)
        if exp > 1:
            base = _batched_mm(base, base)
        exp >>= 1
    return result
