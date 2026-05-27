import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["alpha", "scale", "input_scale"])
def elu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    alpha,
    scale,
    input_scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    out = tl.where(
        x > 0,
        scale * input_scale * x,
        scale * alpha * (tl.exp(x_f32 * input_scale) - 1.0),
    )
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit(do_not_specialize=["alpha", "scale", "input_scale"])
def elu_backward_kernel_with_self(
    grad_output_ptr,
    x_ptr,
    grad_input_ptr,
    n_elements,
    alpha,
    scale,
    input_scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    grad = tl.where(
        x > 0,
        grad_output * scale * input_scale,
        grad_output
        * (scale * alpha * tl.exp(x_fp32 * input_scale) * input_scale),
    )
    tl.store(grad_input_ptr + offsets, grad, mask=mask)


@triton.jit(do_not_specialize=["alpha", "scale", "input_scale"])
def elu_backward_kernel_with_result(
    grad_output_ptr,
    y_ptr,
    grad_input_ptr,
    n_elements,
    alpha,
    scale,
    input_scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    grad = tl.where(
        y > 0,
        grad_output * scale * input_scale,
        grad_output * ((y + scale * alpha) * input_scale),
    )
    tl.store(grad_input_ptr + offsets, grad, mask=mask)


@triton.jit(do_not_specialize=["alpha", "scale", "input_scale"])
def elu_grad_kernel(
    grad_output_ptr,
    self_or_result_ptr,
    grad_input_ptr,
    n_elements,
    alpha,
    scale,
    input_scale,
    IS_RESULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    v = tl.load(self_or_result_ptr + offsets, mask=mask)
    v_fp32 = v.to(tl.float32)
    if IS_RESULT:
        grad = tl.where(
            v > 0,
            grad_output * scale * input_scale,
            grad_output * ((v + scale * alpha) * input_scale),
        )
    else:
        grad = tl.where(
            v > 0,
            grad_output * scale * input_scale,
            grad_output
            * (scale * alpha * tl.exp(v_fp32 * input_scale) * input_scale),
        )
    tl.store(grad_input_ptr + offsets, grad, mask=mask)


def elu(x, alpha=1.0, scale=1.0, input_scale=1.0):
    A_c = x.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    elu_kernel[grid](
        A_c, out, n_elements, alpha, scale, input_scale, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view_as(x)


def elu_(x, alpha=1.0, scale=1.0, input_scale=1.0):
    result = elu(x, alpha, scale, input_scale)
    x.copy_(result)
    return x


def elu_backward(
    grad_output, alpha, scale, input_scale, is_result, self_or_result
):
    A_c = grad_output.contiguous()
    B_c = self_or_result.contiguous()
    grad_input = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    if is_result:
        elu_backward_kernel_with_result[grid](
            A_c,
            B_c,
            grad_input,
            n_elements,
            alpha,
            scale,
            input_scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        elu_backward_kernel_with_self[grid](
            A_c,
            B_c,
            grad_input,
            n_elements,
            alpha,
            scale,
            input_scale,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return grad_input.view_as(grad_output)
