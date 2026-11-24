from .inverse import inverse
from .linalg_eigvalsh import _singular_values
from .linalg_matrix_norm import linalg_matrix_norm


def linalg_cond(A, p=None):
    sv = _singular_values(A)
    if p in (None, 2):
        return sv[..., -1] / sv[..., 0].clamp(min=1e-20)
    if p == -2:
        return sv[..., 0].clamp(min=1e-20).reciprocal() * sv[..., -1]
    if p in (1, -1, float("inf"), float("-inf")):
        return linalg_matrix_norm(inverse(A), ord=p) * linalg_matrix_norm(A, ord=p)
    raise NotImplementedError(f"linalg_cond: ord {p} is not implemented")


__all__ = ["linalg_cond"]
