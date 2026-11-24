import torch
import triton
import triton.language as tl

from .linalg_lu_factor import _batch_count, _block_for


@triton.jit
def _orgqr_kernel(A_ptr, tau_ptr, Q_ptr, m, n, k, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = (offs[:, None] < m) & (offs[None, :] < n)
    base = pid * m * n
    reflectors = tl.load(
        A_ptr + base + offs[:, None] * n + offs[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    Q = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0)
    for rev in range(BLOCK):
        col = k - 1 - rev
        active = (col >= 0) & (col < k)
        v = tl.sum(tl.where(offs[None, :] == col, reflectors, 0.0), axis=1)
        v = tl.where(offs > col, v, 0.0)
        v = tl.where(offs == col, 1.0, v)
        tau = tl.load(tau_ptr + pid * k + col, mask=active, other=0.0)
        for j in range(BLOCK):
            use = active & (j < n) & (j >= col)
            colj = tl.sum(tl.where(offs[:, None] == j, Q, 0.0), axis=0)
            # Q is stored row-major; column j is Q[:, j].
            colj = tl.sum(tl.where(offs[None, :] == j, Q, 0.0), axis=1)
            dot = tl.sum(tl.where(offs >= col, v * colj, 0.0))
            colj = colj - tau * v * dot
            Q = tl.where((offs[None, :] == j) & use, colj[:, None], Q)
    tl.store(Q_ptr + base + offs[:, None] * n + offs[None, :], Q, mask=mask)


def orgqr(A, tau):
    m, n = A.shape[-2], A.shape[-1]
    k = tau.shape[-1]
    batch = _batch_count(A.shape[:-2])
    reflectors = A.reshape(batch, m, n).contiguous()
    tau_b = tau.reshape(batch, k).contiguous()
    out = torch.empty_like(reflectors)
    if batch > 0:
        _orgqr_kernel[(batch,)](
            reflectors, tau_b, out, m, n, k, BLOCK=_block_for(max(m, n))
        )
    return out.reshape(A.shape)


__all__ = ["orgqr"]
