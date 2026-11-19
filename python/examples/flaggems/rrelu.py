import torch
import triton
import triton.language as tl

# rrelu: randomized leaky ReLU.
# eval: x >= 0 -> x, else x * (lower + upper) / 2 (a constant slope).
# train: x >= 0 -> x, else x * U(lower, upper) (element-wise sampled slope).
# The slopes are drawn from the torch RNG (deterministic under
# torch.manual_seed); the transform runs in the Triton kernel.
# Mirrors the upstream argument checks: the generator argument is not
# supported (assert generator is None) and lower must be <= upper.


@triton.jit
def _rrelu_kernel(x_ptr, slope_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    slope = tl.load(slope_ptr + offsets, mask=mask, other=0.0)
    res = tl.where(x >= 0, x, x * slope)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def _apply(x, slope):
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        _rrelu_kernel[(triton.cdiv(n, 1024),)](x, slope, out, n, BLOCK=1024)
    return out


def rrelu(self, lower=0.125, upper=0.33333333333333331, training=False, generator=None):
    """Randomized leaky ReLU (eval slope = (lower + upper) / 2) The generator argument is not supported."""
    assert generator is None, "generator is not supported in FlagGems"
    assert lower <= upper, "lower must be no greater than upper"
    # Build the slope from the shape (not *_like(self)): a strided input
    # would give the slope tensor non-contiguous strides, while the kernel
    # reads it as a flat buffer.
    x = self.contiguous()
    if not training:
        slope = torch.full(
            self.shape, (lower + upper) * 0.5, dtype=x.dtype, device=x.device
        )
    else:
        slope = (
            torch.rand(self.shape, dtype=x.dtype, device=x.device) * (upper - lower)
            + lower
        )
    return _apply(x, slope)
