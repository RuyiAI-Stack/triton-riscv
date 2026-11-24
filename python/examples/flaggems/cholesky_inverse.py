import torch

from .cholesky_solve import cholesky_solve


def cholesky_inverse(L, upper=False):
    n = L.shape[-1]
    eye = torch.eye(n, dtype=L.dtype, device=L.device)
    eye = eye.expand(L.shape).contiguous()
    return cholesky_solve(eye, L, upper=upper)
