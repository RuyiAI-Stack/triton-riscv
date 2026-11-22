import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _round_half_to_even(x):
    floor_x = tl.floor(x)
    fraction = x - floor_x
    floor_is_even = (floor_x % 2.0) == 0.0
    rounded = tl.where(
        fraction > 0.5,
        floor_x + 1.0,
        tl.where((fraction < 0.5) | floor_is_even, floor_x, floor_x + 1.0),
    )
    # floor() saturates non-finite inputs to huge finite values on some
    # backends (riscv64 CPU), which would round NaN/inf to a clamp bound.
    # Detect them from the fp32 exponent bits rather than by float comparison
    # so the passthrough does not depend on comparison semantics.
    abs_bits = x.to(tl.float32).to(tl.int32, bitcast=True) & 0x7FFFFFFF
    return tl.where(abs_bits >= 0x7F800000, x, rounded)


@triton.jit
def fake_quantize_per_channel_affine_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    n_elements,
    n_channels,
    channel_stride,
    quant_min,
    quant_max,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    channel_idx = (offsets // channel_stride) % n_channels
    scale = tl.load(scale_ptr + channel_idx, mask=mask, other=1.0)
    zero_point = tl.load(zero_point_ptr + channel_idx, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    scale_fp32 = scale.to(tl.float32)
    zero_point_fp32 = zero_point.to(tl.float32)
    # aten multiplies by the correctly-rounded fp32 reciprocal (probed on CPU:
    # rint(x * inv) matches bit-exactly, a true division flips tie values)
    inv_scale = tl.math.div_rn(1.0, scale_fp32)
    x_quantized = _round_half_to_even(x_fp32 * inv_scale) + zero_point_fp32
    x_clamped = tl.minimum(tl.maximum(x_quantized, quant_min), quant_max)
    output = (x_clamped - zero_point_fp32) * scale_fp32

    tl.store(output_ptr + offsets, output, mask=mask)


def fake_quantize_per_channel_affine(
    input, scale, zero_point, axis, quant_min, quant_max
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_CHANNEL_AFFINE")

    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    input = input.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    n_elements = input.numel()
    if n_elements == 0:
        return torch.empty_like(input)

    shape = input.shape
    n_channels = shape[axis]

    channel_stride = 1
    for i in range(axis + 1, len(shape)):
        channel_stride *= shape[i]

    output = torch.empty_like(input)

    BLOCK_SIZE = 1024

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    fake_quantize_per_channel_affine_kernel[grid](
        input,
        scale,
        zero_point,
        output,
        n_elements,
        n_channels,
        channel_stride,
        quant_min,
        quant_max,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
