import torch
import triton
import triton.language as tl


@triton.jit
def ilshift_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x << y, mask=mask)


def __ilshift__(A, B):
    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, device=A.device, dtype=A.dtype)
    A, B = torch.broadcast_tensors(A, B)
    A_c = A.contiguous()
    B_c = B.contiguous()
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    ilshift_kernel[grid](
        A_c,
        B_c,
        A_c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    if A_c is not A:
        A.copy_(A_c)
    return A
