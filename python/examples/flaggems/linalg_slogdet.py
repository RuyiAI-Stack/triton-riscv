from collections import namedtuple

from .linalg_det import _det_from_lu
from .linalg_lu_factor import lu_factor

SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])


def linalg_slogdet(A):
    lu, pivots, _ = lu_factor(A)
    sign, logabs = _det_from_lu(lu, pivots)
    return SlogdetResult(sign, logabs)
