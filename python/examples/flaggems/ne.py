import torch
import triton
import triton.language as tl


@triton.jit
def ne_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    tl.store(out_ptr + offsets, (x_f32 != y_f32).to(tl.uint8), mask=mask)


@triton.jit
def ne_scalar_kernel(
    x_ptr,
    y_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    y_f32 = y_val.to(tl.float32)
    tl.store(out_ptr + offsets, (x_f32 != y_f32).to(tl.uint8), mask=mask)


def ne(A, B):
    if isinstance(B, torch.Tensor):
        A, B = torch.broadcast_tensors(A, B)
        A_c = A.contiguous()
        B_c = B.contiguous()
        out = torch.empty_like(A_c, dtype=torch.uint8)
        n_elements = A_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        ne_kernel[grid](A_c, B_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out.view_as(A).to(torch.bool)
    else:
        return ne_scalar(A, B)


def ne_scalar(A, B):
    A_c = A.contiguous()
    out = torch.empty_like(A_c, dtype=torch.uint8)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    ne_scalar_kernel[grid](A_c, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A).to(torch.bool)
