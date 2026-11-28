import torch
import triton
import triton.language as tl


@triton.jit
def _hardtanh_backward_kernel(
    g_ptr, x_ptr, out_ptr, n_elements, min_val, max_val, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Probed on torch CPU: only the open interval (min_val, max_val)
    # passes the gradient; the clamp points themselves are zeroed.
    keep = (x > min_val) & (x < max_val)
    tl.store(out_ptr + offsets, tl.where(keep, g, 0.0), mask=mask)


def hardtanh_backward(grad_output, self_t, min_val=-1.0, max_val=1.0):
    shape = torch.broadcast_shapes(grad_output.shape, self_t.shape)
    g = grad_output.expand(shape).contiguous()
    x = self_t.expand(shape).contiguous()
    out = torch.empty(shape, dtype=g.dtype, device=g.device)
    n = out.numel()
    if n:
        _hardtanh_backward_kernel[(triton.cdiv(n, 1024),)](
            g, x, out, n, float(min_val), float(max_val), BLOCK=1024
        )
    return out
