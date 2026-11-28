import torch
import triton
import triton.language as tl

# logit backward: g / (x * (1 - x)) on the clamped domain. Probed on torch
# CPU: the clamp endpoints are inclusive (x == eps carries
# g / (eps * (1 - eps))) while x == 0 and x == 1 fall outside and yield 0.
# eps=None means no clamp.


@triton.jit
def _logit_backward_kernel(
    g_ptr, x_ptr, out_ptr, n_elements, eps, HAS_EPS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if HAS_EPS:
        inside = (x >= eps) & (x <= 1.0 - eps)
    else:
        inside = (x > 0.0) & (x < 1.0)
    res = tl.where(inside, g / (x * (1.0 - x)), 0.0)
    tl.store(out_ptr + offsets, res, mask=mask)


def logit_backward(grad_output, self_t, eps=None):
    shape = torch.broadcast_shapes(grad_output.shape, self_t.shape)
    g = grad_output.expand(shape).contiguous()
    x = self_t.expand(shape).contiguous()
    out = torch.empty(shape, dtype=g.dtype, device=g.device)
    n = out.numel()
    if n:
        eps_val = float(eps) if eps is not None else 0.0
        _logit_backward_kernel[(triton.cdiv(n, 1024),)](
            g, x, out, n, eps_val, HAS_EPS=eps is not None, BLOCK=1024
        )
    return out
