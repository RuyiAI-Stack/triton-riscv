import torch
import triton

from .ones import ones_kernel


def ones_like(
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

    def grid_fn(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    ones_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out
