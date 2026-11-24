from collections import namedtuple
import torch

from .linalg_lu_factor import lu_factor
from .linalg_lu_solve import lu_solve

InvExResult = namedtuple("InvExResult", ["inverse", "info"])


def linalg_inv_ex(A, *, check_errors=False):
    n = A.shape[-1]
    lu, pivots, info = lu_factor(A)
    eye = torch.eye(n, dtype=A.dtype, device=A.device).expand(A.shape).contiguous()
    inv = lu_solve(lu, pivots, eye)
    if check_errors and bool(info.ne(0).any()):
        raise RuntimeError("linalg_inv_ex: matrix is singular")
    return InvExResult(inv, info.to(torch.int32))


__all__ = ["linalg_inv_ex"]
