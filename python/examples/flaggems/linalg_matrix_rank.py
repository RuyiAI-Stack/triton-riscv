import torch

from .linalg_eigvalsh import _singular_values


def linalg_matrix_rank(A, tol=None, hermitian=False, *, atol=None, rtol=None):
    _ = hermitian
    sv = _singular_values(A)
    if tol is not None:
        cutoff = tol
    else:
        cutoff = sv[..., -1] * max(A.shape[-2:]) * torch.finfo(A.dtype).eps
        if rtol is not None:
            cutoff = sv[..., -1] * rtol
        if atol is not None:
            cutoff = cutoff + atol
    return (sv > cutoff.unsqueeze(-1)).sum(dim=-1)
