import torch
import triton

from .zeros import zeros_kernel


def zeros_like(
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
    grid = (triton.cdiv(N, 1024),)
    zeros_kernel[grid](out, N, BLOCK_SIZE=1024)
    return out
