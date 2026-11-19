import torch
import triton
import triton.language as tl


@triton.jit
def _hardswish_backward_kernel(g_ptr, x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Probed on torch CPU: x <= -3 -> 0, x >= 3 -> grad, interior
    # (2x + 3) / 6; the seams themselves were pinned at exactly 0 and 1.
    interior = g * (2.0 * x + 3.0) / 6.0
    res = tl.where(x <= -3.0, 0.0, tl.where(x >= 3.0, g, interior))
    tl.store(out_ptr + offsets, res, mask=mask)


def hardswish_backward(grad_output, self_t):
    shape = torch.broadcast_shapes(grad_output.shape, self_t.shape)
    g = grad_output.expand(shape).contiguous()
    x = self_t.expand(shape).contiguous()
    out = torch.empty(shape, dtype=g.dtype, device=g.device)
    n = out.numel()
    if n:
        _hardswish_backward_kernel[(triton.cdiv(n, 1024),)](g, x, out, n, BLOCK=1024)
    return out
