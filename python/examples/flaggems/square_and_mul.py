import torch
import triton
import triton.language as tl


@triton.jit
def square_and_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = (x * x) * y

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def square_and_mul_backward_kernel(
    grad_ptr,
    x_ptr,
    y_ptr,
    dx_ptr,
    dy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dx = grad * 2.0 * x * y
    dy = grad * x * x

    tl.store(dx_ptr + offsets, dx, mask=mask)
    tl.store(dy_ptr + offsets, dy, mask=mask)


def square_and_mul(x, y):
    x_broadcast, y_broadcast = torch.broadcast_tensors(x, y)
    x_contiguous = x_broadcast.contiguous()
    y_contiguous = y_broadcast.contiguous()
    output = torch.empty_like(x_contiguous)

    n_elements = x_contiguous.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    square_and_mul_kernel[grid](
        x_contiguous,
        y_contiguous,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return output.view_as(x_broadcast)


def square_and_mul_backward(grad_output, x, y):
    x_shape = x.shape
    y_shape = y.shape
    x_broadcast, y_broadcast, grad_broadcast = torch.broadcast_tensors(
        x, y, grad_output
    )
    x_contiguous = x_broadcast.contiguous()
    y_contiguous = y_broadcast.contiguous()
    grad_contiguous = grad_broadcast.contiguous()
    dx_broadcast = torch.empty_like(x_contiguous)
    dy_broadcast = torch.empty_like(y_contiguous)

    n_elements = x_contiguous.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    square_and_mul_backward_kernel[grid](
        grad_contiguous,
        x_contiguous,
        y_contiguous,
        dx_broadcast,
        dy_broadcast,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return (
        dx_broadcast.sum_to_size(x_shape),
        dy_broadcast.sum_to_size(y_shape),
    )
