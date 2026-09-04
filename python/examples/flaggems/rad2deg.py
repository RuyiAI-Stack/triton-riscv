import torch
import triton
import triton.language as tl


@triton.jit
def rad2deg_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x.to(tl.float32) * 57.29577951308232
    tl.store(out_ptr + offsets, y, mask=mask)


def rad2deg(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("rad2deg expects a torch.Tensor")
    A_c = A.contiguous()
    out_dtype = A_c.dtype if A_c.dtype.is_floating_point else torch.float32
    out = torch.empty(A_c.shape, dtype=out_dtype, device=A_c.device)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out.view_as(A)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rad2deg_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def rad2deg_(A):
    result = rad2deg(A)
    A.copy_(result)
    return A
