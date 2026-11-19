"""aten::special_erfinv -- the out-of-place erfinv.

Reuses the erfinv_ port: the ATen calc_erfinv rational approximation
followed by two fp64 Newton steps that only need erf, exp, sqrt and log
(libdevice.erfinv does not lower on this toolchain).  See erfinv_.py for
the algorithm notes and the measured error bounds.
"""

from .erfinv_ import erfinv
from .erfinv_ import erfinv_kernel_ as _special_erfinv_kernel

__all__ = ["special_erfinv", "_special_erfinv_kernel"]


def special_erfinv(x):
    return erfinv(x)
