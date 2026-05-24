import math

import torch
import triton
import triton.language as tl


# NOTE: angle_complex uses tl.math.atan2 which is NOT available in the
# triton-riscv MLIR backend. Use polynomial approximation or backend needs
# to add math.atan2 support.


@triton.jit
def angle_complex_kernel(
    ri_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    base = offsets * 2

    real = tl.load(ri_ptr + base, mask=mask).to(tl.float32)
    imag = tl.load(ri_ptr + base + 1, mask=mask).to(tl.float32)

    result = tl.math.atan2(imag, real)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def angle_real_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    real = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    zero = 0.0
    pi = math.pi
    real_positive = real >= zero
    result = tl.where(real_positive, zero, pi)

    tl.store(out_ptr + offsets, result, mask=mask)


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    if (
        input_tensor.dtype == torch.complex32
        or input_tensor.dtype == torch.complex64
    ):
        ri = torch.view_as_real(input_tensor).contiguous()
        n_elements = input_tensor.numel()
        out_dtype = input_tensor.real.dtype
        out = torch.empty(
            input_tensor.shape, dtype=out_dtype, device=input_tensor.device
        )
    else:
        n_elements = input_tensor.numel()
        out_dtype = input_tensor.dtype
        if not out_dtype.is_floating_point:
            out_dtype = torch.float32
        out = torch.empty(
            input_tensor.shape, dtype=out_dtype, device=input_tensor.device
        )

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    if (
        input_tensor.dtype == torch.complex32
        or input_tensor.dtype == torch.complex64
    ):
        angle_complex_kernel[grid](ri, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        x_c = input_tensor.contiguous()
        angle_real_kernel[grid](x_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
