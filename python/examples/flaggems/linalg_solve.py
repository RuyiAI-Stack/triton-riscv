from .linalg_lu_factor import lu_factor
from .linalg_lu_solve import lu_solve


def linalg_solve(A, B):
    lu, pivots, info = lu_factor(A)
    if bool(info.ne(0).any()):
        raise RuntimeError("linalg_solve: singular matrix")
    return lu_solve(lu, pivots, B)
