import torch
import triton
import triton.language as tl


@triton.jit
def nan_to_num_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    nan_val,
    posinf_val,
    neginf_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_nan = x != x
    x_posinf = x == float("inf")
    x_neginf = x == -float("inf")
    x = tl.where(x_nan, nan_val, x)
    x = tl.where(x_posinf, posinf_val, x)
    x = tl.where(x_neginf, neginf_val, x)
    tl.store(out_ptr + offsets, x, mask=mask)


def nan_to_num(A, nan=None, posinf=None, neginf=None):
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    nan_to_num_kernel[grid](
        A_c, out, n_elements, nan, posinf, neginf, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view_as(A)
