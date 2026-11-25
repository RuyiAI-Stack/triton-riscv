import torch

from .linalg_lu_factor import lu_factor


def _det_from_lu(lu, pivots):
    diag = torch.diagonal(lu, dim1=-2, dim2=-1)
    # A swap is recorded when the 1-based pivot is not the current row.
    n = lu.shape[-1]
    rows = torch.arange(1, n + 1, device=lu.device)
    swaps = (pivots != rows).to(diag.dtype)
    # Each recorded swap flips the sign. An even-length cycle of swaps is not
    # what LAPACK stores: it stores the raw pivot row, and the number of
    # nontrivial pivots has the same parity as the permutation.
    sign = torch.where(swaps.sum(dim=-1) % 2 == 0, 1.0, -1.0).to(diag.dtype)
    logabs = torch.log(diag.abs()).sum(dim=-1)
    sign = sign * torch.where(diag < 0, -1.0, 1.0).prod(dim=-1)
    return sign, logabs


def linalg_det(A):
    lu, pivots, info = lu_factor(A)
    sign, logabs = _det_from_lu(lu, pivots)
    return sign * torch.exp(logabs)
