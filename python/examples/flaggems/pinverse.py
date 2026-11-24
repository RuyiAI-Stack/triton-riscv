from .linalg_pinv import linalg_pinv


def pinverse(A, rcond=1e-15):
    return linalg_pinv(A, rtol=rcond)


__all__ = ["pinverse"]
