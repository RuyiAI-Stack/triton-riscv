import torch
import triton
import triton.language as tl

# psi (digamma), the same recurrence + asymptotic series as the digamma_
# port (reflection below 0.5, recurrence to >= 8, Bernoulli tail), with
# the special-namespace boundary contract pinned on top (probed on torch
# CPU): psi(+-0) = -inf, psi at negative integers = nan, psi(+inf) =
# +inf, psi(-inf) = nan, nan stays nan.


@triton.jit
def _psi_flat32(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    pi = 3.1415926535897932384626433832795028841971

    reflect = x < 0.5
    xr = tl.where(reflect, 1.0 - x, x)

    s = tl.zeros((BLOCK_SIZE,), tl.float32)
    y = xr
    for _ in range(0, 8):
        m = y < 8.0
        s = s - tl.where(m, 1.0 / y, 0.0)
        y = tl.where(m, y + 1.0, y)

    r = 1.0 / y
    r2 = r * r
    t2 = r2
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    series = (
        (-0.5 * r)
        + (-1.0 / 12.0) * t2
        + (1.0 / 120.0) * t4
        + (-1.0 / 252.0) * t6
        + (1.0 / 240.0) * t8
    )
    psi_y = tl.log(y) + s + series

    cot = tl.cos(pi * x) / tl.sin(pi * x)
    res = tl.where(reflect, psi_y - pi * cot, psi_y)

    res = tl.where(x == 0.0, float("-inf"), res)
    res = tl.where((x < 0.0) & (x == tl.floor(x)), float("nan"), res)
    res = tl.where(x == float("inf"), float("inf"), res)
    res = tl.where(x != x, x, res)

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(A, out=None):
    A_c = A.contiguous()
    if out is None:
        out = torch.empty_like(A_c)
        out_c = out
    else:
        out_c = out if out.is_contiguous() else out.contiguous()
    n_elements = A_c.numel()
    if n_elements == 0:
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    _psi_flat32[grid](A_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_psi(A):
    return _run(A)


def special_psi_out(A, out):
    return _run(A, out=out)
