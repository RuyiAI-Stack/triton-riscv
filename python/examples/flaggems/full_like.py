import torch
import triton

from .full import (
    check_dtype,
    full_kernel,
)


def full_like(
    x,
    fill_value,
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
    fill_value = check_dtype(fill_value, dtype, device)

    out = torch.empty(x.size(), device=device, dtype=dtype)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    if isinstance(fill_value, torch.Tensor):
        fill_tensor = fill_value
    else:
        fill_tensor = torch.tensor(fill_value, device=device, dtype=dtype)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    full_kernel[grid](out, fill_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
