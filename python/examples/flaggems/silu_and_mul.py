import torch
import triton
import triton.language as tl


@triton.jit
def silu_and_mul_kernel(
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
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    result = x_silu * y

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def silu_and_mul_grad_kernel(
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
    sig = 1 / (1 + tl.exp(-x_fp32))
    x_silu = x_fp32 * sig
    d_x_silu = sig * (1 + x_fp32 * (1 - sig))
    dx = d_x_silu * dgrad * y
    dy = dgrad * x_silu

    tl.store(dx_ptr + offsets, dx, mask=mask)
    tl.store(dy_ptr + offsets, dy, mask=mask)


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        output = torch.empty_like(A)
        n_elements = A.numel()
        BLOCK_SIZE = 64
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        silu_and_mul_kernel[grid](
            A, B, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)
        n_elements = A.numel()
        BLOCK_SIZE = 64
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        silu_and_mul_grad_kernel[grid](
            A,
            B,
            grad_output,
            grad_A,
            grad_B,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return grad_A, grad_B


def silu_and_mul(A, B):
    return SiluAndMul.apply(A, B)


def silu_and_mul_out(A, B, out):
    n_element = A.numel()
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_element, BLOCK_SIZE),)
    silu_and_mul_kernel[grid](A, B, out, n_element, BLOCK_SIZE=BLOCK_SIZE)
    return out
