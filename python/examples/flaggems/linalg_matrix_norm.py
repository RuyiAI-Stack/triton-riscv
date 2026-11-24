import torch

from .linalg_eigvalsh import _singular_values


def linalg_matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    _ = dim
    work = A if dtype is None else A.to(dtype)
    if ord in ("fro", None):
        val = torch.sqrt(work.square().sum(dim=(-2, -1)))
    elif ord == 1:
        val = work.abs().sum(dim=-2).max(dim=-1).values
    elif ord == -1:
        val = work.abs().sum(dim=-2).min(dim=-1).values
    elif ord == float("inf"):
        val = work.abs().sum(dim=-1).max(dim=-1).values
    elif ord == float("-inf"):
        val = work.abs().sum(dim=-1).min(dim=-1).values
    elif ord in (2, -2, "nuc"):
        sv = _singular_values(work)
        if ord == 2:
            val = sv[..., -1]
        elif ord == -2:
            val = sv[..., 0]
        else:
            val = sv.sum(dim=-1)
    else:
        raise NotImplementedError(f"linalg_matrix_norm: ord {ord}")
    if keepdim:
        val = val.unsqueeze(-1).unsqueeze(-1)
    return val


__all__ = ["linalg_matrix_norm"]
