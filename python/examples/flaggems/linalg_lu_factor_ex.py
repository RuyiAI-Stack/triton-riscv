from collections import namedtuple
import torch

from .linalg_lu_factor import lu_factor

LuExResult = namedtuple("LuExResult", ["LU", "pivots", "info"])


def linalg_lu_factor_ex(A, *, pivot=True, check_errors=False):
    _ = pivot
    lu, pivots, info = lu_factor(A)
    if check_errors and bool(info.ne(0).any()):
        raise RuntimeError("linalg_lu_factor_ex: singular matrix")
    return LuExResult(lu, pivots.to(torch.int32), info)


__all__ = ["linalg_lu_factor_ex"]
