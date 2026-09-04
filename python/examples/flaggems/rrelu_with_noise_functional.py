import torch
import triton
import triton.language as tl

from .uniform import uniform_


@triton.jit
def rrelu_with_noise_kernel(
    x_ptr,
    noise_ptr,
    out_ptr,
    n_elements,
    slope,
    TRAINING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    noise = tl.load(noise_ptr + offsets, mask=mask) if TRAINING else slope
    out = tl.where(x >= 0, x, x * noise)
    if TRAINING:
        tl.store(noise_ptr + offsets, tl.where(x > 0, 1.0, noise), mask=mask)
    tl.store(out_ptr + offsets, out, mask=mask)


def rrelu_with_noise_functional(
    self,
    noise,
    lower=0.125,
    upper=0.33333333333333331,
    training=False,
    generator=None,
):
    return _rrelu_with_noise_impl(self, noise, lower, upper, training, generator)


def _rrelu_with_noise_impl(self, noise, lower, upper, training, generator):
    if noise.shape != self.shape:
        raise AssertionError("noise tensor must have the same shape as self")
    out = torch.empty_like(self)
    out_contiguous = torch.empty_like(self, memory_format=torch.contiguous_format)
    slope = (lower + upper) * 0.5
    n_elements = self.numel()
    if training:
        noise_out = torch.empty_like(noise)
        noise_contiguous = torch.empty_like(
            noise, memory_format=torch.contiguous_format
        )
        uniform_(noise_contiguous, lower, upper, generator=generator)
    else:
        noise_out = noise.clone()
        noise_contiguous = noise_out
    if n_elements == 0:
        return out, noise_out
    grid = (triton.cdiv(n_elements, 1024),)
    rrelu_with_noise_kernel[grid](
        self.contiguous(),
        noise_contiguous,
        out_contiguous,
        n_elements,
        slope,
        training,
        BLOCK_SIZE=1024,
    )
    out.copy_(out_contiguous.view_as(self))
    if training:
        noise_out.copy_(noise_contiguous.view_as(noise))
    return out, noise_out
