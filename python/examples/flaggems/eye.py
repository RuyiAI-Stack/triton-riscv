import torch
import triton

from .eye_m import eye_kernel


def eye(
    size,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=None,
):
    """Triton-based implementation of torch.eye(n, n)."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = "cpu"
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye.")

    out = torch.empty(
        (size, size),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    BLOCK_SIZE = 32
    grid = (triton.cdiv(size, BLOCK_SIZE), triton.cdiv(size, BLOCK_SIZE))

    eye_kernel[grid](
        out,
        size,
        size,
        BLOCK_SIZE,
        BLOCK_SIZE,
    )
    return out
