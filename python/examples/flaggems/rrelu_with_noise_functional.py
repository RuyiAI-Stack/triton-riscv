import torch
import triton
import triton.language as tl

# rrelu_with_noise_functional: the functional flavor of rrelu_with_noise
# returning (output, noise).
# train: the caller's noise tensor is written through (x < 0 slots get the
# U(lower, upper) sample, x >= 0 slots get exactly 1.0) and the output is
# x * noise; eval leaves the noise tensor untouched and returns it as-is,
# with output = x * (lower + upper) / 2 on the negatives.
# The generator argument is not supported (assert generator is None) and
# the noise shape must match self, mirroring the upstream checks.


@triton.jit
def _rrelu_with_noise_functional_kernel(
    x_ptr, slope_ptr, out_ptr, noise_ptr, n_elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    slope = tl.load(slope_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x >= 0, x, x * slope)
    noise = tl.where(x >= 0, 1.0, slope)
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(noise_ptr + offsets, noise.to(noise_ptr.dtype.element_ty), mask=mask)


def rrelu_with_noise_functional(
    self, noise, lower=0.125, upper=0.33333333333333331, training=False, generator=None
):
    """Functional rrelu_with_noise returning (output, noise).

    The generator argument is not supported; the noise shape must match self.
    """
    assert generator is None, "generator is not supported in FlagGems"
    assert noise.shape == self.shape, "noise tensor must have the same shape as self"
    x = self.contiguous()
    if not training:
        slope = torch.full_like(x, (lower + upper) * 0.5)
    else:
        slope = torch.rand_like(x) * (upper - lower) + lower
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        scratch = torch.empty_like(slope)
        _rrelu_with_noise_functional_kernel[(triton.cdiv(n, 1024),)](
            x, slope, out, scratch, n, BLOCK=1024
        )
        if training:
            noise.copy_(scratch.reshape(noise.shape))
    return out, noise
