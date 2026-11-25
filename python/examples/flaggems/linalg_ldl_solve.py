import torch

from .linalg_solve_triangular import solve_triangular


def linalg_ldl_solve(LD, pivots, B, *, hermitian=False):
    _ = hermitian
    _ = pivots
    unit = torch.tril(LD, diagonal=-1) + torch.diag_embed(
        torch.ones(LD.shape[:-1], dtype=LD.dtype, device=LD.device)
    )
    diag = LD.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    y = solve_triangular(unit, B, upper=False, unitriangular=True)
    y = y / diag
    return solve_triangular(unit, y, upper=False, transpose=True, unitriangular=True)
