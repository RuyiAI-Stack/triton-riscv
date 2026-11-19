import torch
import triton
import triton.language as tl


@triton.jit
def _hardsigmoid_backward_kernel(
    g_ptr, x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Probed on torch CPU: the active band is the open interval (-3, 3)
    # and the slope there is exactly 1/6.
    active = (x > -3.0) & (x < 3.0)
    tl.store(out_ptr + offsets, tl.where(active, g / 6.0, 0.0), mask=mask)


def hardsigmoid_backward(grad_output, self_t):
    shape = torch.broadcast_shapes(grad_output.shape, self_t.shape)
    g = grad_output.expand(shape).contiguous()
    x = self_t.expand(shape).contiguous()
    out = torch.empty(shape, dtype=g.dtype, device=g.device)
    n = out.numel()
    if n:
        _hardsigmoid_backward_kernel[(triton.cdiv(n, 1024),)](g, x, out, n, BLOCK=1024)
    return out
