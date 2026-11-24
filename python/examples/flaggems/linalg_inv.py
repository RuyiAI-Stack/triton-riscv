from .inverse import inverse


def linalg_inv(A):
    return inverse(A)


__all__ = ["linalg_inv"]
