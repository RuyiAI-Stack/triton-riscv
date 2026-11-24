from collections import namedtuple

from .linalg_lu_factor import lu_factor
from .linalg_lu_solve import lu_solve

SolveExResult = namedtuple("SolveExResult", ["result", "info"])


def linalg_solve_ex(A, B, *, check_errors=False):
    lu, pivots, info = lu_factor(A)
    result = lu_solve(lu, pivots, B)
    if check_errors and bool(info.ne(0).any()):
        raise RuntimeError("linalg_solve_ex: singular matrix")
    return SolveExResult(result, info)
