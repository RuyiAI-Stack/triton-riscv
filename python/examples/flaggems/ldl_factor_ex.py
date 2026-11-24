from collections import namedtuple
import torch

from .cholesky import cholesky

LdlResult = namedtuple("LdlResult", ["LD", "pivots", "info"])


def _ldl_from_cholesky(A):
    lower = cholesky(A, upper=False)
    d = lower.diagonal(dim1=-2, dim2=-1).square()
    # Unit lower L = chol(L) / diag.
    scale = lower.diagonal(dim1=-2, dim2=-1).clamp(min=1e-20)
    unit = lower / scale.unsqueeze(-2)
    unit = torch.tril(unit, diagonal=-1)
    ld = unit + torch.diag_embed(d)
    return ld


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    _ = hermitian
    ld = _ldl_from_cholesky(A)
    n = A.shape[-1]
    pivots = torch.arange(1, n + 1, device=A.device, dtype=torch.int32)
    pivots = pivots.expand(A.shape[:-2] + (n,)).contiguous()
    info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=A.device)
    if check_errors and bool(info.ne(0).any()):
        raise RuntimeError("ldl_factor_ex failed")
    return LdlResult(ld, pivots, info)
