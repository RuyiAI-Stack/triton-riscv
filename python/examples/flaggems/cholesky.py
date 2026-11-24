import torch
import triton
import triton.language as tl

from .linalg_lu_factor import _batch_count, _block_for, _check_batch_square


@triton.jit
def _chol_kernel(A_ptr, L_ptr, info_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = (offs[:, None] < n) & (offs[None, :] < n)
    base = pid * n * n
    idx = offs[:, None] * n + offs[None, :]
    W = tl.load(A_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    info = 0
    for j in range(BLOCK):
        active = j < n
        rowj = tl.sum(tl.where(offs[:, None] == j, W, 0.0), axis=0)
        dots_jj = tl.sum(tl.where(offs < j, rowj * rowj, 0.0))
        ajj = tl.sum(tl.where(offs == j, rowj, 0.0))
        pivot = ajj - dots_jj
        bad = active & (pivot <= 1.0e-20)
        info = tl.where(bad & (info == 0), j + 1, info)
        ljj = tl.sqrt(tl.maximum(pivot, 0.0))
        W = tl.where((offs[:, None] == j) & (offs[None, :] == j) & active, ljj, W)
        left = offs[None, :] < j
        dots = tl.sum(tl.where(left, W * rowj[None, :], 0.0), axis=1)
        coli = tl.sum(tl.where(offs[None, :] == j, W, 0.0), axis=1)
        safe = tl.where(ljj > 1.0e-20, ljj, 1.0)
        lij = (coli - dots) / safe
        write = (offs[:, None] > j) & (offs[None, :] == j) & active & (~bad)
        W = tl.where(write, lij[:, None], W)
    W = tl.where(offs[None, :] > offs[:, None], 0.0, W)
    tl.store(L_ptr + base + idx, tl.where(mask, W, 0.0), mask=mask)
    tl.store(info_ptr + pid, info)


def cholesky(A, upper=False):
    n = _check_batch_square(A, "cholesky")
    src = A.transpose(-1, -2).contiguous() if upper else A
    batch_shape = src.shape[:-2]
    batch = _batch_count(batch_shape)
    work = src.reshape(batch, n, n).contiguous()
    out = torch.empty_like(work)
    info = torch.empty((batch,), dtype=torch.int32, device=A.device)
    if batch > 0 and n > 0:
        _chol_kernel[(batch,)](work, out, info, n, BLOCK=_block_for(n))
    if bool(info.ne(0).any()):
        raise RuntimeError("cholesky: input is not positive definite")
    factor = out.reshape(A.shape)
    return factor.transpose(-1, -2).contiguous() if upper else factor
