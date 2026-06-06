import torch
import triton
import triton.language as tl


@triton.jit
def gelu_none_kernel(
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
    scale: tl.constexpr = 0.7071067811
    out = 0.5 * x * (1.0 + tl.erf(x_f32 * scale))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gelu_tanh_kernel(
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
    tanh_arg = x_f32 * 0.79788456 * (1.0 + 0.044715 * x_f32 * x_f32)
    exp2x = tl.exp(2.0 * tanh_arg)
    tanh_val = (exp2x - 1.0) / (exp2x + 1.0)
    out = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gelu_backward_none_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    scale1: tl.constexpr = 0.7071067811
    scale2: tl.constexpr = 0.3989422803
    dydx = (
        scale2 * x_f32 * tl.exp(-(scale1 * x_f32) * (scale1 * x_f32))
        + 0.5 * tl.erf(scale1 * x_f32)
        + 0.5
    )
    tl.store(dx_ptr + offsets, dydx * dy, mask=mask)


@triton.jit
def gelu_backward_tanh_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    dy = tl.load(dy_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    tanh_arg = 0.79788456 * x_f32 * (1.0 + 0.044715 * x_f32 * x_f32)
    exp2x = tl.exp(2.0 * tanh_arg)
    tanh_out = (exp2x - 1.0) / (exp2x + 1.0)
    dydx = 0.5 * x_f32 * (
        (1.0 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x_f32 * x_f32)
    ) + 0.5 * (1.0 + tanh_out)
    tl.store(dx_ptr + offsets, dydx * dy, mask=mask)


def gelu(x, approximate="none"):
    A_c = x.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    if approximate == "tanh":
        gelu_tanh_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        gelu_none_kernel[grid](A_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x)


def gelu_backward(grad_output, self, *, approximate="none"):
    n_elements = self.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    in_grad = torch.empty_like(self)
    if approximate == "tanh":
        gelu_backward_tanh_kernel[grid](
            self, grad_output, in_grad, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        gelu_backward_none_kernel[grid](
            self, grad_output, in_grad, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
    return in_grad


def gelu_(A, approximate="none"):
    result = gelu(A, approximate)
    A.copy_(result)
    return A
