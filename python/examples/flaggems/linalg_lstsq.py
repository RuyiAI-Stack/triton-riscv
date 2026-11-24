from collections import namedtuple
import torch

from .linalg_eigvalsh import _singular_values
from .linalg_pinv import linalg_pinv


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


def linalg_lstsq(A, B, rcond=None, driver=None):
    _ = driver
    x = linalg_pinv(A, rtol=1e-5 if rcond is None else rcond)
    # pinv is n x m, B is m x k or m
    if B.ndim == A.ndim - 1:
        sol = _batched_mm(x, B.unsqueeze(-1)).squeeze(-1)
    else:
        sol = _batched_mm(x, B)
    sv = _singular_values(A)
    rank = int((sv > (1e-5 if rcond is None else rcond) * sv[..., -1:]).sum().item())
    return namedtuple("Lstsq", ["solution", "residuals", "rank", "singular_values"])(
        sol, torch.empty(0, device=A.device), rank, sv
    )


__all__ = ["linalg_lstsq"]
