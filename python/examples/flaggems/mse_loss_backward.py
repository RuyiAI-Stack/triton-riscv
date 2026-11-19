import torch
import triton
import triton.language as tl

# mse backward: grad * 2 * (self - target) / D, with D = numel(self) for
# reduction=1 (mean) and D = 1 for reduction 0 (none) and 2 (sum).
# Probed on torch CPU; grad_output may be 0-dim for the reduced modes and is
# broadcast to the elementwise shape here.


@triton.jit
def _mse_loss_backward_kernel(
    g_ptr, x_ptr, t_ptr, out_ptr, n_elements, scale, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(t_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, g * (x - t) * scale, mask=mask)


def mse_loss_backward(grad_output, self_t, target, reduction=1):
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
        scale = 2.0 / n_total if int(reduction) == 1 else 2.0
        _mse_loss_backward_kernel[(triton.cdiv(n, 1024),)](
            g, x, t, out, n, scale, BLOCK=1024
        )
    return out
