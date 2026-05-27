import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(x >= 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, output, mask=mask)


def leaky_relu(A, negative_slope=0.01):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        A_c,
        out,
        n_elements,
        negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view_as(A)


def leaky_relu_(A, negative_slope=0.01):
    if not A.is_contiguous():
        raise RuntimeError("leaky_relu_ requires a contiguous tensor")
    n_elements = A.numel()
    if n_elements == 0:
        return A
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        A,
        A,
        n_elements,
        negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return A


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    if out is None:
        return leaky_relu(A, negative_slope)
    A_c = A.contiguous()
    n_elements = A_c.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        A_c,
        out,
        n_elements,
        negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
