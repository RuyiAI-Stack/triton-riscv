from .cholesky import cholesky


def linalg_cholesky(A, upper=False):
    return cholesky(A, upper=upper)
