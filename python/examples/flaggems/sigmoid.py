import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_forward(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    tl.store(
        out_ptr + offsets,
        1.0 / (1.0 + tl.math.exp2(-x_f32 * log2e)),
        mask=mask,
    )


@triton.jit
def sigmoid_backward_kernel(
    y_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    y = tl.load(y_ptr + offsets, mask=mask)
    dy = tl.load(dy_ptr + offsets, mask=mask)
    y_f32 = y.to(tl.float32)
    tl.store(dx_ptr + offsets, dy.to(tl.float32) * (1.0 - y_f32) * y_f32, mask=mask)


def sigmoid(x):
    A_c = x.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_forward[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x)


def sigmoid_backward(grad_output, output):
    grad_input = torch.empty_like(output)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_backward_kernel[grid](
        output, grad_output, grad_input, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return grad_input


def sigmoid_(A):
    result = sigmoid(A)
    A.copy_(result)
    return A
