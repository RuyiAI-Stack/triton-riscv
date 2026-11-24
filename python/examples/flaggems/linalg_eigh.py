from collections import namedtuple
import torch
import triton
import triton.language as tl

from .linalg_lu_factor import _batch_count, _block_for, _check_batch_square

EigResult = namedtuple("EigResult", ["eigenvalues", "eigenvectors"])


@triton.jit
def _jacobi_rotate_kernel(A_ptr, V_ptr, n, p, q, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = (offs[:, None] < n) & (offs[None, :] < n)
    base = pid * n * n
    idx = offs[:, None] * n + offs[None, :]
    A = tl.load(A_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    V = tl.load(V_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    app = tl.sum(tl.where((offs == p)[:, None] & (offs == p)[None, :], A, 0.0))
    aqq = tl.sum(tl.where((offs == q)[:, None] & (offs == q)[None, :], A, 0.0))
    apq = tl.sum(tl.where((offs == p)[:, None] & (offs == q)[None, :], A, 0.0))
    small = tl.abs(apq) <= 1.0e-12 * (tl.abs(app) + tl.abs(aqq) + 1.0)
    tau = (aqq - app) / (2.0 * tl.where(tl.abs(apq) > 0, apq, 1.0))
    tsign = tl.where(tau < 0, -1.0, 1.0)
    t = tsign / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    c = tl.where(small, 1.0, 1.0 / tl.sqrt(1.0 + t * t))
    s = tl.where(small, 0.0, t * c)
    col_p = tl.sum(tl.where(offs[None, :] == p, A, 0.0), axis=1)
    col_q = tl.sum(tl.where(offs[None, :] == q, A, 0.0), axis=1)
    A = tl.where(offs[None, :] == p, (c * col_p - s * col_q)[:, None], A)
    A = tl.where(offs[None, :] == q, (s * col_p + c * col_q)[:, None], A)
    row_p = tl.sum(tl.where(offs[:, None] == p, A, 0.0), axis=0)
    row_q = tl.sum(tl.where(offs[:, None] == q, A, 0.0), axis=0)
    A = tl.where(offs[:, None] == p, (c * row_p - s * row_q)[None, :], A)
    A = tl.where(offs[:, None] == q, (s * row_p + c * row_q)[None, :], A)
    vp = tl.sum(tl.where(offs[None, :] == p, V, 0.0), axis=1)
    vq = tl.sum(tl.where(offs[None, :] == q, V, 0.0), axis=1)
    V = tl.where(offs[None, :] == p, (c * vp - s * vq)[:, None], V)
    V = tl.where(offs[None, :] == q, (s * vp + c * vq)[:, None], V)
    tl.store(A_ptr + base + idx, A, mask=mask)
    tl.store(V_ptr + base + idx, V, mask=mask)


def linalg_eigh(A, UPLO="L"):
    n = _check_batch_square(A, "linalg_eigh")
    _ = UPLO
    batch = _batch_count(A.shape[:-2])
    work = 0.5 * (A + A.transpose(-1, -2))
    work = work.reshape(batch, n, n).contiguous()
    vecs = (
        torch.eye(n, dtype=A.dtype, device=A.device)
        .expand(batch, n, n)
        .contiguous()
        .clone()
    )
    block = _block_for(n)
    if batch > 0 and n > 1:
        for _sweep in range(10):
            for p in range(n):
                for q in range(p + 1, n):
                    _jacobi_rotate_kernel[(batch,)](work, vecs, n, p, q, BLOCK=block)
    eig = torch.diagonal(work.reshape(A.shape), dim1=-2, dim2=-1)
    vecs = vecs.reshape(A.shape)
    order = torch.argsort(eig, dim=-1)
    eig = torch.gather(eig, -1, order)
    vecs = torch.gather(vecs, -1, order.unsqueeze(-2).expand_as(vecs))
    return EigResult(eig, vecs)


__all__ = ["linalg_eigh"]
