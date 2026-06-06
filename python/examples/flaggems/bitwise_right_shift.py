import torch
import triton
import triton.language as tl


@triton.jit
def bitwise_right_shift_kernel(
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
    tl.store(out_ptr + offsets, x >> y, mask=mask)


def bitwise_right_shift(A, B):
    A, B = torch.broadcast_tensors(A, B)
    A_c = A.contiguous()
    B_c = B.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    bitwise_right_shift_kernel[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def bitwise_right_shift_out(A, B, out):
    result = bitwise_right_shift(A, B)
    out.copy_(result.view_as(out))
    return out
