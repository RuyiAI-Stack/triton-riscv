import torch
import triton
import triton.language as tl

# rrelu_with_noise (aten::rrelu_with_noise): the caller's noise tensor is
# the sampling scratch space.
# train: for x < 0 the noise slot is overwritten with a U(lower, upper)
# sample and the output is x * noise; for x >= 0 the noise slot is reset to
# exactly 1.0 and the output is x (probed on torch CPU).
# eval: output = x * (lower + upper) / 2 on the negatives and the noise
# tensor is left untouched (bit-identical).
# The generator argument is not supported (assert generator is None) and
# the noise shape must match self, mirroring the upstream checks.


@triton.jit
def _rrelu_with_noise_kernel(
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


def rrelu_with_noise(
    self, noise, lower=0.125, upper=0.33333333333333331, training=False, generator=None
):
    """Randomized leaky ReLU sampling through the caller's noise tensor.

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
        # The kernel also materializes the noise view (1.0 on positives);
        # in eval mode it is discarded so the caller's tensor stays intact.
        scratch = torch.empty_like(slope)
        _rrelu_with_noise_kernel[(triton.cdiv(n, 1024),)](
            x, slope, out, scratch, n, BLOCK=1024
        )
        if training:
            noise.copy_(scratch.reshape(noise.shape))
    return out
