import torch

from .linalg_lu_factor import _block_for
from .linalg_lu_solve import _apply_pivots_kernel


def _apply_lu_perm(batch, n, pivots):
    eye = torch.eye(n, dtype=torch.float32, device=pivots.device)
    eye = eye.expand(batch, n, n).contiguous().clone()

    piv = pivots.reshape(batch, n).to(torch.int32).contiguous()
    if batch and n:
        _apply_pivots_kernel[(batch,)](
            eye, piv, n, n, BLOCK=_block_for(n), NRHS=_block_for(n)
        )
    return eye


def lu_unpack(LU, pivots, unpack_data=True, unpack_pivots=True):
    n = LU.shape[-1]
    batch = 1
    for dim in LU.shape[:-2]:
        batch *= dim
    packed = LU.reshape(batch, n, n)
    piv = pivots.reshape(batch, n)
    p = _apply_lu_perm(batch, n, piv) if unpack_pivots else torch.empty(0)
    lower = torch.tril(packed, diagonal=-1)
    diag = torch.eye(n, dtype=LU.dtype, device=LU.device)
    l_factor = (lower + diag).reshape(LU.shape)
    u_factor = torch.triu(packed).reshape(LU.shape)
    if not unpack_data:
        l_factor = torch.empty(0, device=LU.device)
        u_factor = torch.empty(0, device=LU.device)
    return p.reshape(LU.shape), l_factor, u_factor


__all__ = ["lu_unpack"]
