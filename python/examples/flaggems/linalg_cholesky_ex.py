from collections import namedtuple
import torch

from .cholesky import _chol_kernel
from .linalg_lu_factor import _batch_count, _block_for, _check_batch_square

CholeskyExResult = namedtuple("CholeskyExResult", ["L", "info"])


def linalg_cholesky_ex(A, upper=False, check_errors=False):
    n = _check_batch_square(A, "linalg_cholesky_ex")
    src = A.transpose(-1, -2).contiguous() if upper else A
    batch_shape = A.shape[:-2]
    batch = _batch_count(batch_shape)
    work = src.reshape(batch, n, n).contiguous()
    out = torch.empty_like(work)
    info = torch.zeros((batch,), dtype=torch.int32, device=A.device)
    if batch > 0 and n > 0:
        _chol_kernel[(batch,)](work, out, info, n, BLOCK=_block_for(n))
    factor = out.reshape(A.shape)
    if upper:
        factor = factor.transpose(-1, -2).contiguous()
    info = info.reshape(batch_shape)
    if check_errors and bool(info.ne(0).any()):
        raise RuntimeError("linalg_cholesky_ex: factorization failed")
    return CholeskyExResult(factor, info)


__all__ = ["linalg_cholesky_ex"]
