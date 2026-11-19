import torch
import triton
import triton.language as tl

# Shifted Chebyshev U*_n: the U recurrence on s = 2x - 1
# (U*_0 = 1, U*_1 = 4x - 2, U*_k = 2s U*_{k-1} - U*_{k-2});
# n < 0 degrades to 0.


@triton.jit
def shifted_chebyshev_polynomial_u_kernel(
    x_ptr, n_ptr, out_ptr, n_elements, MAXDEG: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    n = tl.load(n_ptr + offsets, mask=mask, other=0).to(tl.int32)

    s = 2.0 * x - 1.0
    res = tl.where(n == 0, 1.0, 2.0 * s)
    res = tl.where(n < 0, 0.0, res)
    p0 = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    p1 = 2.0 * s
    for k in range(2, MAXDEG + 1):
        p2 = 2.0 * s * p1 - p0
        p0 = p1
        p1 = p2
        res = tl.where(n == k, p2, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(x, n, out=None):
    if not isinstance(n, torch.Tensor):
        n = torch.full((), n, dtype=torch.int64, device=x.device)
    x_b, n_b = torch.broadcast_tensors(x, n.to(torch.int64))
    x_c = x_b.contiguous()
    n_c = n_b.contiguous()
    if out is None:
        out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    if n_elements == 0:
        return out
    maxdeg = max(int(n_c.max().item()), 1)
    grid = (triton.cdiv(n_elements, 1024),)
    shifted_chebyshev_polynomial_u_kernel[grid](
        x_c, n_c, out, n_elements, MAXDEG=maxdeg, BLOCK_SIZE=1024
    )
    return out


def special_shifted_chebyshev_polynomial_u(x, n):
    return _run(x, n)


def special_shifted_chebyshev_polynomial_u_(x, n):
    res = _run(x, n)
    x.copy_(res)
    return x
