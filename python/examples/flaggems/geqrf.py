import torch
import triton
import triton.language as tl

from .linalg_lu_factor import _MAX_N, _batch_count, _block_for


@triton.jit
def _geqrf_kernel(A_ptr, tau_ptr, m, n, k_tau, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = (offs[:, None] < m) & (offs[None, :] < n)
    base = pid * m * n
    A = tl.load(
        A_ptr + base + offs[:, None] * n + offs[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    kmax = n if m > n else m
    for k in range(BLOCK):
        active = k < kmax
        col = tl.sum(tl.where(offs[None, :] == k, A, 0.0), axis=1)
        alpha = tl.sum(tl.where(offs == k, col, 0.0))
        tail = tl.sum(tl.where(offs > k, col * col, 0.0))
        xnorm = tl.sqrt(tail)
        sign = tl.where(alpha < 0, -1.0, 1.0)
        beta = -sign * tl.sqrt(alpha * alpha + xnorm * xnorm)
        no_reflect = active & (xnorm == 0) & (alpha >= 0)
        tau = tl.where(no_reflect, 0.0, (beta - alpha) / tl.where(beta == 0, 1.0, beta))
        tau = tl.where(active, tau, 0.0)
        denom = alpha - beta
        safe = tl.where(tl.abs(denom) > 0, denom, 1.0)
        v = tl.where(offs > k, col / safe, 0.0)
        v = tl.where(offs == k, 1.0, v)
        v = tl.where(no_reflect, tl.where(offs == k, 1.0, 0.0), v)
        stored = tl.where(offs > k, v, col)
        stored = tl.where(offs == k, tl.where(no_reflect, alpha, beta), stored)
        A = tl.where((offs[None, :] == k) & active, stored[:, None], A)
        tl.store(tau_ptr + pid * k_tau + k, tau, mask=active)
        # Apply H to the trailing columns. Rows below k and the pivot row.
        for j in range(BLOCK):
            use = active & (j > k) & (j < n)
            colj = tl.sum(tl.where(offs[None, :] == j, A, 0.0), axis=1)
            dot = tl.sum(tl.where(offs >= k, v * colj, 0.0))
            colj = colj - tau * v * dot
            A = tl.where((offs[None, :] == j) & use, colj[:, None], A)
    tl.store(A_ptr + base + offs[:, None] * n + offs[None, :], A, mask=mask)


def geqrf(A):
    if A.ndim < 2:
        raise RuntimeError("geqrf: expected a matrix")
    if A.dtype != torch.float32:
        raise NotImplementedError("geqrf: only float32 is implemented")
    m, n = A.shape[-2], A.shape[-1]
    if max(m, n) > _MAX_N:
        raise NotImplementedError(f"geqrf: tile supports size<={_MAX_N}")
    batch = _batch_count(A.shape[:-2])
    work = A.reshape(batch, m, n).contiguous().clone()
    k = min(m, n)
    tau = torch.zeros((batch, k), dtype=A.dtype, device=A.device)
    block = _block_for(max(m, n))
    if batch > 0 and k > 0:
        _geqrf_kernel[(batch,)](work, tau, m, n, k, BLOCK=block)
    return work.reshape(A.shape), tau.reshape(A.shape[:-2] + (k,))
