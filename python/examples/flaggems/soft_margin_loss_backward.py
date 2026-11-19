import torch
import triton
import triton.language as tl

# soft-margin backward: loss = log(1 + exp(-x*t)), so
# dL/dx = -t * sigmoid(-x*t) = -t / (1 + exp(x*t)).
# reduction=1 (mean) divides by numel(self); 0/2 leave the elementwise value.
#
# Non-fp64 inputs compute in fp32 (libdevice has no fp16/bf16 signatures).


@triton.jit
def _soft_margin_loss_backward_kernel(
    g_ptr,
    x_ptr,
    t_ptr,
    out_ptr,
    n_elements,
    scale,
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(t_ptr + offsets, mask=mask, other=0.0)
    if IS64:
        xc = x.to(tl.float64)
        tc = t.to(tl.float64)
    else:
        xc = x.to(tl.float32)
        tc = t.to(tl.float32)
    sig_neg = 1.0 / (1.0 + tl.exp(xc * tc))
    res = g * (-tc) * sig_neg * scale
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def soft_margin_loss_backward(grad_output, self_t, target, reduction=1):
    if int(reduction) not in (0, 1, 2):
        raise ValueError("reduction must be 0 (none), 1 (mean) or 2 (sum)")
    shape = torch.broadcast_shapes(self_t.shape, target.shape)
    x = self_t.expand(shape).contiguous()
    t = target.expand(shape).contiguous()
    g = grad_output.expand(shape).contiguous()
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    n = out.numel()
    if n:
        n_total = x.numel()
        scale = 1.0 / n_total if int(reduction) == 1 else 1.0
        _soft_margin_loss_backward_kernel[(triton.cdiv(n, 1024),)](
            g, x, t, out, n, scale, IS64=x.dtype == torch.float64, BLOCK=1024
        )
    return out
