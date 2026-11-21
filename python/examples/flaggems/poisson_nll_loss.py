import torch
import triton
import triton.language as tl

# poisson_nll_loss, probed on torch CPU:
#   log_input=True : e^x - t * x
#   log_input=False: x - t * log(x + eps)
#   full=True adds the Stirling term t*ln(t) - t + 0.5*ln(2*pi*t) ONLY where
#   t > 1 (t = 2.0 adds it; t = 0.5 and t = 1.0 do not).
# The kernel writes elementwise contributions; the reduction runs on the
# host. Non-fp64 inputs compute in fp32 (libdevice has no fp16/bf16
# signatures).


@triton.jit
def _poisson_nll_loss_kernel(
    x_ptr,
    t_ptr,
    out_ptr,
    n_elements,
    eps,
    LOG_INPUT: tl.constexpr,
    FULL: tl.constexpr,
    IS64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(t_ptr + offsets, mask=mask, other=0.0)
    if IS64:
        xc = x.to(tl.float64)
        tc = t.to(tl.float64)
    else:
        xc = x.to(tl.float32)
        tc = t.to(tl.float32)
    if LOG_INPUT:
        loss = tl.exp(xc) - tc * xc
    else:
        loss = xc - tc * tl.log(xc + eps)
    if FULL:
        stirling = tc * tl.log(tc) - tc + 0.5 * tl.log(6.283185307179586 * tc)
        loss = tl.where(tc > 1.0, loss + stirling, loss)
    tl.store(out_ptr + offsets, loss.to(out_ptr.dtype.element_ty), mask=mask)


def poisson_nll_loss(self, target, log_input=True, full=False, eps=1e-8, reduction=1):
    """Poisson negative log likelihood with the t > 1 Stirling gate."""
    x = self.contiguous()
    t = target.contiguous()
    shape = torch.broadcast_shapes(x.shape, t.shape)
    x = x.expand(shape).contiguous()
    t = t.expand(shape).contiguous()
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    n = out.numel()
    if n:
        _poisson_nll_loss_kernel[(triton.cdiv(n, 1024),)](
            x,
            t,
            out,
            n,
            float(eps),
            LOG_INPUT=bool(log_input),
            FULL=bool(full),
            IS64=x.dtype == torch.float64,
            BLOCK=1024,
        )
    if reduction == 0:
        return out
    if reduction == 2:
        return out.sum()
    return out.sum() / n if n else out.sum()
