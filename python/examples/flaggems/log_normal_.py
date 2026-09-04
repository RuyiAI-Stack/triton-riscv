import torch
import triton
import triton.language as tl

from .rand import philox_backend_seed_offset
from .randn import randn_kernel

UNROLL = 4


@triton.jit
def log_normal_transform_kernel(
    val_ptr,
    out_ptr,
    n_elements,
    mean: tl.constexpr,
    std: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(val_ptr + offsets, mask=mask, other=0.0)
    out = tl.exp(val * std + mean)
    tl.store(out_ptr + offsets, out, mask=mask)


def volume(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def log_normal_distribution(
    shape, device, dtype, mean, std, *, generator=None, out=None
):
    if out is None:
        out = torch.empty(shape, device=device, dtype=dtype)
    N = volume(shape)
    if N == 0:
        return out

    def grid_fn(meta):
        return (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    # Generate float32 normal first, then transform and convert to target dtype
    temp_out = torch.empty(shape, device=device, dtype=torch.float32)

    BLOCK = 1024
    randn_kernel[grid_fn](temp_out, N, philox_seed, philox_offset, BLOCK=BLOCK)

    # Transform to log-normal and store in output
    grid = (triton.cdiv(N, 1024),)
    log_normal_transform_kernel[grid](temp_out, out, N, mean, std, BLOCK_SIZE=BLOCK)
    return out


def log_normal_(self, mean=1.0, std=2.0, *, generator=None):
    shape = self.shape
    device = self.device
    dtype = self.dtype
    if not self.is_contiguous():
        work = torch.empty(shape, device=device, dtype=dtype)
        log_normal_distribution(
            shape, device, dtype, mean, std, generator=generator, out=work
        )
        self.copy_(work)
        return self

    # Generate log-normal distribution in-place
    log_normal_distribution(
        shape, device, dtype, mean, std, generator=generator, out=self
    )
    return self
