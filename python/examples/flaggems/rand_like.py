import torch
import triton

from .rand import philox_backend_seed_offset, rand_kernel


UNROLL = 4


def rand_like(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()

    BLOCK = 1024

    def grid_fn(meta):
        return (triton.cdiv(N, BLOCK * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)

    rand_kernel[grid_fn](out, N, philox_seed, philox_offset, BLOCK=BLOCK)
    return out
