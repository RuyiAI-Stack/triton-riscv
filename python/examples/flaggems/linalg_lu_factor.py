import torch
import triton
import triton.language as tl


_MAX_N = 32


def _check_batch_square(A, name):
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError(f"{name}: expected a batch of square matrices")
    n = A.shape[-1]
    if n > _MAX_N:
        raise NotImplementedError(f"{name}: Triton tile supports n<={_MAX_N}, got {n}")
    if A.dtype != torch.float32:
        raise NotImplementedError(f"{name}: only float32 is implemented")
    return n


def _batch_count(shape):
    batch = 1
    for dim in shape:
        batch *= dim
    return batch


def _block_for(n):
    block = 1
    while block < n:
        block *= 2
    return max(block, 1)


@triton.jit
def _lu_factor_kernel(A_ptr, LU_ptr, piv_ptr, info_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = (offs[:, None] < n) & (offs[None, :] < n)
    base = pid * n * n
    idx = offs[:, None] * n + offs[None, :]
    W = tl.load(A_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    info = 0
    for k in range(BLOCK):
        active = k < n
        col = tl.sum(tl.where(offs[None, :] == k, W, 0.0), axis=1)
        score = tl.where((offs >= k) & (offs < n) & active, tl.abs(col), -1.0)
        _, p = tl.max(score, axis=0, return_indices=True)
        is_k = offs[:, None] == k
        is_p = offs[:, None] == p
        row_k = tl.sum(tl.where(is_k, W, 0.0), axis=0)
        row_p = tl.sum(tl.where(is_p, W, 0.0), axis=0)
        swapped = tl.where(is_p, row_k[None, :], tl.where(is_k, row_p[None, :], W))
        W = tl.where(active, swapped, W)
        tl.store(piv_ptr + pid * n + k, p + 1, mask=active)

        diag = tl.sum(tl.where(is_k & (offs[None, :] == k), W, 0.0))
        singular = active & (tl.abs(diag) <= 1.0e-20)
        info = tl.where(singular & (info == 0), k + 1, info)
        safe = tl.where(tl.abs(diag) > 1.0e-20, diag, 1.0)
        is_col = offs[None, :] == k
        colk = tl.sum(tl.where(is_col, W, 0.0), axis=1)
        below = offs > k
        new_col = tl.where(below, colk / safe, colk)
        W = tl.where(is_col & active, new_col[:, None], W)
        rowk = tl.sum(tl.where(is_k, W, 0.0), axis=0)
        update = new_col[:, None] * rowk[None, :]
        trail = (offs[:, None] > k) & (offs[None, :] > k) & mask
        W = tl.where(trail, W - update, W)
    tl.store(LU_ptr + base + idx, W, mask=mask)
    tl.store(info_ptr + pid, info)


def lu_factor(A):
    n = _check_batch_square(A, "lu_factor")
    batch_shape = A.shape[:-2]
    batch = _batch_count(batch_shape)
    work = A.reshape(batch, n, n).contiguous()
    lu = torch.empty_like(work)
    pivots = torch.empty((batch, n), dtype=torch.int32, device=A.device)
    info = torch.empty((batch,), dtype=torch.int32, device=A.device)
    if batch == 0 or n == 0:
        return (
            lu.reshape(A.shape),
            pivots.reshape(batch_shape + (n,)),
            info.reshape(batch_shape),
        )
    block = _block_for(n)
    _lu_factor_kernel[(batch,)](work, lu, pivots, info, n, BLOCK=block)
    return (
        lu.reshape(A.shape),
        pivots.reshape(batch_shape + (n,)).to(torch.int64),
        info.reshape(batch_shape),
    )


def linalg_lu_factor(A):
    lu, pivots, _info = lu_factor(A)
    return lu, pivots.to(torch.int32)
