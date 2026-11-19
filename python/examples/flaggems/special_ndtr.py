import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ndtr(x) = Phi(x) = erfc(-x / sqrt(2)) / 2.  erfc has no lowering on
# this toolchain, so the port inlines the hybrid measured in the erfc /
# foreach-unary families: 1 - erf(t) below 2.85 and a 12-term asymptotic
# series above (both tails of Phi need the series -- plain 1 - erf
# cancels to 0 for t >= 3.5 in fp32).  Boundary contract (probed on
# torch CPU): Phi(-inf) = 0, Phi(+inf) = 1, Phi(0) = 0.5, nan stays nan.


@triton.jit
def ndtr_func(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    t = -x * 0.70710678118654752440
    # the series seed must not read t's value: t * 0.0 is nan at -inf.
    s0 = tl.zeros((BLOCK_SIZE,), tl.float32) + 1.0
    tt = s0
    for k in range(1, 12):
        tt = tt * (-(2.0 * k - 1.0) / (2.0 * t * t))
        s0 = s0 + tt
    asym = tl.exp(-t * t) / (t * 1.7724538509055159) * s0
    erfc_t = tl.where(t > 2.85, asym, 1.0 - libdevice.erf(t))
    res = 0.5 * erfc_t

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def special_ndtr(A):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, 1024),)
    ndtr_func[grid](A_c, out, n_elements, BLOCK_SIZE=1024)
    return out
