import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# erfc semantics: erfc(+inf) = 0, erfc(-inf) = 2, erfc(nan) = nan and
# erfc(+-0) = 1.
#
# libdevice.erfc does not lower on this toolchain, so the port reuses the
# hybrid measured in the foreach unary family: plain 1 - erf(x) cancels to
# exactly 0 once erf rounds to 1 (fp32, x >= ~3.5), so above the measured
# crossover 2.85 a 12-term asymptotic series is used; the negative side
# has no cancellation and keeps 1 - erf(x) for every x <= 2.85.


@triton.jit
def _erfc_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    # the seed must not read x's value: x * 0.0 is nan at +inf.
    t = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    s = t
    for k in range(1, 12):
        t = t * (-(2.0 * k - 1.0) / (2.0 * x * x))
        s = s + t
    asym = tl.exp(-x * x) / (x * 1.7724538509055159) * s
    res = tl.where(x > 2.85, asym, 1.0 - libdevice.erf(x))

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _run(x, out=None):
    x_c = x.contiguous()
    if out is None:
        out = torch.empty_like(x_c)
        out_c = out
    else:
        out_c = out if out.is_contiguous() else out.contiguous()
    n_elements = x_c.numel()
    if n_elements == 0:
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    _erfc_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_erfc(x):
    return _run(x)
