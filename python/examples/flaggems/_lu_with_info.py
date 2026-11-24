from .linalg_lu_factor_ex import linalg_lu_factor_ex


def _lu_with_info(A):
    return linalg_lu_factor_ex(A)


__all__ = ["_lu_with_info"]
