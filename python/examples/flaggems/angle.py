import math

import torch
import triton
import triton.language as tl


@triton.jit
def _atan_unit_interval(z):
    """Approximate atan(z) for z in [0, 1] using range reduction."""
    reduce = z > 0.4142135623730950
    t = tl.where(reduce, (z - 1.0) / (z + 1.0), z)
    t2 = t * t
    poly = -1.0 / 13.0
    poly = 1.0 / 11.0 + t2 * poly
    poly = -1.0 / 9.0 + t2 * poly
    poly = 1.0 / 7.0 + t2 * poly
    poly = -1.0 / 5.0 + t2 * poly
    poly = 1.0 / 3.0 + t2 * poly
    atan_t = t - t * t2 * poly
    return tl.where(reduce, 0.7853981633974483 + atan_t, atan_t)


@triton.jit
def _atan2_approx(y, x):
    ax = tl.abs(x)
    ay = tl.abs(y)
    swap = ay > ax
    hi = tl.maximum(ax, ay)
    lo = tl.minimum(ax, ay)
    ratio = lo / tl.where(hi == 0.0, 1.0, hi)
    result = _atan_unit_interval(ratio)
    result = tl.where(swap, 1.5707963267948966 - result, result)
    result = tl.where(x < 0.0, 3.141592653589793 - result, result)
    return tl.where(y < 0.0, -result, result)


@triton.jit
def angle_complex_kernel(
    real_ptr,
    imag_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    real = tl.load(real_ptr + offsets, mask=mask).to(tl.float32)
    imag = tl.load(imag_ptr + offsets, mask=mask).to(tl.float32)

    result = _atan2_approx(imag, real)
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
        ri = torch.view_as_real(input_tensor)
        real = ri[..., 0].contiguous()
        imag = ri[..., 1].contiguous()
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
        angle_complex_kernel[grid](
            real, imag, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        x_c = input_tensor.contiguous()
        angle_real_kernel[grid](x_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
