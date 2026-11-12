import torch
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel_(
    x_ptr,
    n_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # hardtanh(x) = clamp(x, min_val, max_val) in the input dtype, so integer
    # inputs stay integers. PyTorch's clamp propagates NaN while
    # tl.maximum/tl.minimum drop it, so NaN is restored.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.minimum(tl.maximum(x, min_val), max_val)
    y = tl.where(x != x, x, y)

    tl.store(x_ptr + offsets, y, mask=mask)


def hardtanh_(x: torch.Tensor, min_val=-1.0, max_val=1.0):
    x_c = x.contiguous()
    n_elements = x_c.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    hardtanh_kernel_[grid](x_c, n_elements, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE)

    if x_c.data_ptr() != x.data_ptr():
        x.copy_(x_c)
    return x
