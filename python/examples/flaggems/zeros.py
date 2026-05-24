import torch
import triton
import triton.language as tl


@triton.jit
def zeros_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, 0.0, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    out = torch.empty(size, device=device, dtype=dtype)
    n_elements = out.numel()
    num_blocks = triton.cdiv(n_elements, 1024)
    zeros_kernel[(num_blocks,)](out, n_elements, BLOCK_SIZE=1024)
    return out


def zero_(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, 1024)
    zeros_kernel[(num_blocks,)](x, n_elements, BLOCK_SIZE=1024)
    return x
