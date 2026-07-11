import torch
import triton
import triton.language as tl


@triton.jit
def logical_not_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    IS_BOOL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if IS_BOOL:
        res = (x == 0).to(tl.int8)
    else:
        res = ~x
    tl.store(out_ptr + offsets, res, mask=mask)


def logical_not(A):
    assert isinstance(A, torch.Tensor)
    n_elements = A.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    is_bool = A.dtype == torch.bool
    out = torch.empty_like(A, dtype=torch.uint8 if is_bool else A.dtype)
    A_c = A.to(torch.uint8) if is_bool else A.contiguous()
    logical_not_kernel[grid](
        A_c, out, n_elements, IS_BOOL=is_bool, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.bool() if is_bool else out
