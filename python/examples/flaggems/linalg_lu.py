from .linalg_lu_factor import lu_factor
from .lu_unpack import lu_unpack


def linalg_lu(A, *, pivot=True):
    _ = pivot
    lu, pivots, _info = lu_factor(A)
    return lu_unpack(lu, pivots)
