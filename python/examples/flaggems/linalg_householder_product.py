from .orgqr import orgqr


def linalg_householder_product(A, tau):
    return orgqr(A, tau)


__all__ = ["linalg_householder_product"]
