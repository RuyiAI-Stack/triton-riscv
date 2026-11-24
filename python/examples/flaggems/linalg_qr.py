import torch

from .geqrf import geqrf
from .orgqr import orgqr


def linalg_qr(A, mode="reduced"):
    if mode not in ("reduced", "complete"):
        raise NotImplementedError(f"linalg_qr: mode {mode} is not implemented")
    factor, tau = geqrf(A)
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    q = orgqr(factor[..., :k].contiguous(), tau)
    r = torch.triu(factor)
    if mode == "reduced":
        r = r[..., :k, :]
    elif m != n:
        raise NotImplementedError("linalg_qr complete mode supports square matrices")
    return q, r


__all__ = ["linalg_qr"]
