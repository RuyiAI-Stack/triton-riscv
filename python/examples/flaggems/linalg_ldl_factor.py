from .ldl_factor_ex import ldl_factor_ex


def linalg_ldl_factor(A, *, hermitian=False):
    result = ldl_factor_ex(A, hermitian=hermitian)
    return result.LD, result.pivots
