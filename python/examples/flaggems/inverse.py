import torch

from .linalg_lu_factor import _check_batch_square, lu_factor
from .linalg_lu_solve import lu_solve


def inverse(A):
    n = _check_batch_square(A, "inverse")
    eye = torch.eye(n, dtype=A.dtype, device=A.device).expand(A.shape).contiguous()
    lu, pivots, info = lu_factor(A)
    if bool(info.ne(0).any()):
        raise RuntimeError("inverse: matrix is singular")
    return lu_solve(lu, pivots, eye)


__all__ = ["inverse"]
