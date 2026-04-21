import torch
import triton
import triton.language as tl


@triton.jit
def gelu_none_and_mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    x_fp32 = x.to(tl.float32)
    RCP_SQRT_2: tl.constexpr = 0.7071067811
    x_gelu = 0.5 * x_fp32 * (1 + tl.erf(x_fp32 * RCP_SQRT_2))
    result = x_gelu * y

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def gelu_none_and_mul_grad_kernel(
    x_ptr,
    y_ptr,
    dgrad_ptr,
    dx_ptr,
    dy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    dgrad = tl.load(dgrad_ptr + offsets, mask=mask)

    RCP_SQRT_2: tl.constexpr = 0.7071067811
    COEFF: tl.constexpr = 0.7978845608028654

    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + tl.erf(x_fp32 * RCP_SQRT_2))

    d_gelu = dgrad * y
    dx = (
        d_gelu
        * 0.5
        * (
            1.0
            + tl.erf(x_fp32 * RCP_SQRT_2)
            + x_fp32 * COEFF * tl.exp(-0.5 * x_fp32 * x_fp32)
        )
    )
    dy = dgrad * x_gelu

    tl.store(dx_ptr + offsets, dx, mask=mask)
    tl.store(dy_ptr + offsets, dy, mask=mask)


@triton.jit
def gelu_tanh_and_mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    x_fp32 = x.to(tl.float32)
    tanh_arg = x_fp32 * 0.79788456 * (1.0 + 0.044715 * (x_fp32 * x_fp32))
    exp2x = tl.exp(2.0 * tanh_arg)
    tanh_val = (exp2x - 1.0) / (exp2x + 1.0)

    x_gelu = 0.5 * x_fp32 * (1.0 + tanh_val)
    result = x_gelu * y

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def gelu_tanh_and_mul_grad_kernel(
    x_ptr,
    y_ptr,
    dgrad_ptr,
    dx_ptr,
    dy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    dgrad = tl.load(dgrad_ptr + offsets, mask=mask)

    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = x_fp32 * x_fp32 * x_fp32
    tanh_arg = sqrt_2_over_pi * (x_fp32 + 0.044715 * a_cubed)
    exp2x = tl.exp(2.0 * tanh_arg)
    tanh_result = (exp2x - 1.0) / (exp2x + 1.0)
    geglu_a = 0.5 * x_fp32 * (1 + tanh_result)
    dy = geglu_a * dgrad

    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = (
        0.5
        * x_fp32
        * (1 - tanh_sq)
        * (sqrt_2_over_pi * (1 + 3 * 0.044715 * x_fp32 * x_fp32))
    )
    dx = dgrad * y_fp32 * (term1 + term2)

    tl.store(dx_ptr + offsets, dx, mask=mask)
    tl.store(dy_ptr + offsets, dy, mask=mask)


class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, approximate="none"):
        ctx.save_for_backward(x, y)
        ctx.approximate = approximate
        output = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 64
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        if approximate == "none":
            gelu_none_and_mul_kernel[grid](
                x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
        elif approximate == "tanh":
            gelu_tanh_and_mul_kernel[grid](
                x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")
        return output

    @staticmethod
    def backward(ctx, dgrad):
        x, y = ctx.saved_tensors
        dx = torch.empty_like(x)
        dy = torch.empty_like(y)
        n_elements = x.numel()
        BLOCK_SIZE = 64
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        if ctx.approximate == "none":
            gelu_none_and_mul_grad_kernel[grid](
                x, y, dgrad, dx, dy, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            gelu_tanh_and_mul_grad_kernel[grid](
                x, y, dgrad, dx, dy, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
        return dx, dy, None


def gelu_and_mul(x, y, approximate="none"):
    return GeluAndMul.apply(x, y, approximate)
