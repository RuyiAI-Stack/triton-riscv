import torch
import triton
import triton.language as tl


@triton.jit
def logical_not_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    res = ~x.to(tl.int1)
    tl.store(out_ptr + offsets, res, mask=mask)


def logical_not(A):
    assert isinstance(A, torch.Tensor)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    out = torch.empty_like(A, dtype=torch.bool)
    A_c = A.contiguous()
    logical_not_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
