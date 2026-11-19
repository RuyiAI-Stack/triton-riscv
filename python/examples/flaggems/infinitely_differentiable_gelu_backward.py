import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# gelu (erf form) backward: f(x) = x * Phi(x) with Phi(x) = (1 + erf(x/sqrt2))/2,
# so f'(x) = Phi(x) + x * phi(x). libdevice.erf lowers on this toolchain and was
# measured exact vs torch (fp32 5.96e-8, fp64 1.11e-16) before this port.
#
# libdevice has no fp16/bf16 signatures, so non-fp64 inputs are computed in
# fp32 (the suite convention); fp64 keeps native double math.


@triton.jit
def _infinitely_differentiable_gelu_backward_kernel(
    g_ptr, x_ptr, out_ptr, n_elements, IS64: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if IS64:
        gc = g.to(tl.float64)
        xc = x.to(tl.float64)
    else:
        gc = g.to(tl.float32)
        xc = x.to(tl.float32)
    cdf = 0.5 * (1.0 + libdevice.erf(xc * 0.7071067811865476))
    pdf = 0.3989422804014327 * tl.exp(-0.5 * xc * xc)
    res = gc * (cdf + xc * pdf)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def infinitely_differentiable_gelu_backward(grad, self_input):
    shape = torch.broadcast_shapes(grad.shape, self_input.shape)
    g = grad.expand(shape).contiguous()
    x = self_input.expand(shape).contiguous()
    out = torch.empty(shape, dtype=g.dtype, device=g.device)
    n = out.numel()
    if n:
        _infinitely_differentiable_gelu_backward_kernel[(triton.cdiv(n, 1024),)](
            g, x, out, n, IS64=g.dtype == torch.float64, BLOCK=1024
        )
    return out
