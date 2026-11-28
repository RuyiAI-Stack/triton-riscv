import torch
import triton
import triton.language as tl

# mish backward: f(x) = x * tanh(softplus(x)),
# f'(x) = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x).
# softplus is evaluated in the overflow-safe shift form
# max(x, 0) + log(1 + exp(-|x|)); the naive log is acceptable because the
# residual is only a tiny additive term whose absolute error is below the
# measured 1e-5 budget (and vanishes again inside tanh for large |x|).
#
# tanh(sp) is written as (1 - u) / (1 + u) with u = exp(-2 sp) <= 1:
# libdevice.tanh emits a tt.extern_elementwise that this backend cannot tile
# for fp64. Non-fp64 inputs compute in fp32 (libdevice has no fp16/bf16
# signatures), fp64 keeps native double math.


@triton.jit
def _mish_backward_kernel(
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
    sp = tl.maximum(xc, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(xc)))
    u = tl.exp(-2.0 * sp)
    t = (1.0 - u) / (1.0 + u)
    sig = 1.0 / (1.0 + tl.exp(-xc))
    res = gc * (t + xc * (1.0 - t * t) * sig)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def mish_backward(grad_output, input):
    shape = torch.broadcast_shapes(grad_output.shape, input.shape)
    g = grad_output.expand(shape).contiguous()
    x = input.expand(shape).contiguous()
    out = torch.empty(shape, dtype=g.dtype, device=g.device)
    n = out.numel()
    if n:
        _mish_backward_kernel[(triton.cdiv(n, 1024),)](
            g, x, out, n, IS64=g.dtype == torch.float64, BLOCK=1024
        )
    return out
