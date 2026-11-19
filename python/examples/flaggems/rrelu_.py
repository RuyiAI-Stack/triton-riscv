import torch
import triton
import triton.language as tl

# In-place rrelu: same semantics as rrelu (eval slope = (lower + upper) / 2,
# train slope = U(lower, upper) on the negatives); the input is mutated and
# returned. The generator argument is not supported (assert generator is
# None) and lower must be <= upper, mirroring the upstream checks.


@triton.jit
def _rrelu_kernel(x_ptr, slope_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    slope = tl.load(slope_ptr + offsets, mask=mask, other=0.0)
    res = tl.where(x >= 0, x, x * slope)
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def rrelu_(
    self, lower=0.125, upper=0.33333333333333331, training=False, generator=None
):
    """In-place randomized leaky ReLU; the input is mutated and returned The generator argument is not supported."""
    assert generator is None, "generator is not supported in FlagGems"
    assert lower <= upper, "lower must be no greater than upper"
    assert self.is_contiguous(), "rrelu_ requires a contiguous tensor"
    if not training:
        slope = torch.full_like(self, (lower + upper) * 0.5)
    else:
        slope = torch.rand_like(self) * (upper - lower) + lower
    n = self.numel()
    if n:
        _rrelu_kernel[(triton.cdiv(n, 1024),)](self, slope, self, n, BLOCK=1024)
    return self
