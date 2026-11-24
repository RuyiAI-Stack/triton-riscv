from .linalg_det import linalg_det


def _linalg_det(A):
    return linalg_det(A)


__all__ = ["_linalg_det"]
