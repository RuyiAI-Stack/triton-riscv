import torch
import triton
import triton.language as tl

from .linalg_lu_factor import _batch_count, _block_for


@triton.jit
def _apply_pivots_kernel(
    B_ptr, piv_ptr, n, nrhs, BLOCK: tl.constexpr, NRHS: tl.constexpr
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, NRHS)
    mask = (rows[:, None] < n) & (cols[None, :] < nrhs)
    base = pid * n * nrhs
    idx = rows[:, None] * nrhs + cols[None, :]
    X = tl.load(B_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    for k in range(BLOCK):
        active = k < n
        p = tl.load(piv_ptr + pid * n + k, mask=active, other=k + 1) - 1
        is_k = rows[:, None] == k
        is_p = rows[:, None] == p
        row_k = tl.sum(tl.where(is_k, X, 0.0), axis=0)
        row_p = tl.sum(tl.where(is_p, X, 0.0), axis=0)
        swapped = tl.where(is_p, row_k[None, :], tl.where(is_k, row_p[None, :], X))
        X = tl.where(active, swapped, X)
    tl.store(B_ptr + base + idx, X, mask=mask)


@triton.jit
def _lu_solve_kernel(LU_ptr, X_ptr, n, nrhs, BLOCK: tl.constexpr, NRHS: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, NRHS)
    mask_a = (rows[:, None] < n) & (rows[None, :] < n)
    mask_x = (rows[:, None] < n) & (cols[None, :] < nrhs)
    a_idx = rows[:, None] * n + rows[None, :]
    x_idx = rows[:, None] * nrhs + cols[None, :]
    base_a = pid * n * n
    base_x = pid * n * nrhs
    LU = tl.load(LU_ptr + base_a + a_idx, mask=mask_a, other=0.0).to(tl.float32)
    X = tl.load(X_ptr + base_x + x_idx, mask=mask_x, other=0.0).to(tl.float32)
    # Unit lower L occupies the strict lower triangle. U occupies the upper triangle.
    for i in range(BLOCK):
        active = i < n
        row = tl.sum(tl.where(rows[:, None] == i, LU, 0.0), axis=0)
        dots = tl.sum(tl.where((rows < i)[:, None], row[:, None] * X, 0.0), axis=0)
        xi = tl.sum(tl.where(rows[:, None] == i, X, 0.0), axis=0)
        xi = tl.where(active, xi - dots, xi)
        X = tl.where((rows[:, None] == i) & active, xi[None, :], X)
    for rev in range(BLOCK):
        i = BLOCK - 1 - rev
        active = i < n
        row = tl.sum(tl.where(rows[:, None] == i, LU, 0.0), axis=0)
        diag = tl.sum(tl.where(rows == i, row, 0.0))
        safe = tl.where(tl.abs(diag) > 1.0e-20, diag, 1.0)
        dots = tl.sum(tl.where((rows > i)[:, None], row[:, None] * X, 0.0), axis=0)
        xi = tl.sum(tl.where(rows[:, None] == i, X, 0.0), axis=0)
        xi = tl.where(active, (xi - dots) / safe, xi)
        X = tl.where((rows[:, None] == i) & active, xi[None, :], X)
    tl.store(X_ptr + base_x + x_idx, X, mask=mask_x)


def lu_solve(LU, pivots, B):
    n = LU.shape[-1]
    vector = B.ndim == LU.ndim - 1 and B.shape[-1] == n
    if vector:
        nrhs = 1
    elif B.ndim == LU.ndim and B.shape[-2] == n:
        nrhs = B.shape[-1]
    else:
        raise RuntimeError("lu_solve: B is incompatible with LU")
    batch_shape = LU.shape[:-2]
    batch = _batch_count(batch_shape)
    if vector:
        rhs = B.reshape(batch, n, 1).contiguous()
        squeeze = True
    else:
        rhs = B.reshape(batch, n, nrhs).contiguous()
        squeeze = False
    lu = LU.reshape(batch, n, n).contiguous()
    piv = pivots.reshape(batch, n).to(torch.int32).contiguous()
    X = rhs.clone()
    block = _block_for(n)
    nrhs_block = _block_for(nrhs)
    if batch > 0 and n > 0:
        _apply_pivots_kernel[(batch,)](X, piv, n, nrhs, BLOCK=block, NRHS=nrhs_block)
        _lu_solve_kernel[(batch,)](lu, X, n, nrhs, BLOCK=block, NRHS=nrhs_block)
    if squeeze:
        return X.reshape(B.shape)
    return X.reshape(B.shape)


def linalg_lu_solve(LU, pivots, B, *, left=True, adjoint=False):
    if not left or adjoint:
        raise NotImplementedError("linalg_lu_solve supports left, non-adjoint systems")
    return lu_solve(LU, pivots, B)


__all__ = ["linalg_lu_solve"]
