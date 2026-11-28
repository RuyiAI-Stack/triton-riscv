import torch
import triton
import triton.language as tl

# Legendre P_n via P_0 = 1, P_1 = x,
# P_k = ((2k - 1) x P_{k-1} - (k-1) P_{k-2}) / k; n < 0 degrades to 0.


@triton.jit
def special_legendre_polynomial_p_forward(
    x_ptr, n_ptr, out_ptr, n_elements, MAXDEG: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    n = tl.load(n_ptr + offsets, mask=mask, other=0).to(tl.int32)

    res = tl.where(n == 0, 1.0, x)
    res = tl.where(n < 0, 0.0, res)
    p0 = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    p1 = x
    for k in range(2, MAXDEG + 1):
        p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1) * p0) / k
        p0 = p1
        p1 = p2
        res = tl.where(n == k, p2, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_legendre_polynomial_p(x, n):
    if not isinstance(n, torch.Tensor):
        n = torch.full((), n, dtype=torch.int64, device=x.device)
    x_b, n_b = torch.broadcast_tensors(x, n.to(torch.int64))
    x_c = x_b.contiguous()
    n_c = n_b.contiguous()
    out = torch.empty_like(x_c)
    n_elements = x_c.numel()
    if n_elements == 0:
        return out
    maxdeg = max(int(n_c.max().item()), 1)
    grid = (triton.cdiv(n_elements, 1024),)
    special_legendre_polynomial_p_forward[grid](
        x_c, n_c, out, n_elements, MAXDEG=maxdeg, BLOCK_SIZE=1024
    )
    return out
