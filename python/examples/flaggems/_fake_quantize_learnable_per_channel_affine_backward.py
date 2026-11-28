"""Backward of fake-quantize with learnable per-channel affine qparams.

Port notes (triton-riscv, vs the upstream flag_gems file):
- ``tl_extra_shim.rint`` does not lower here; replaced by the floor-based
  round-half-to-even helper (same rounding mode as ``rint``/``nearbyint``).
- The per-element gradient arithmetic is upstream's verbatim.
"""

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
def _fake_quantize_learnable_per_channel_affine_backward_kernel(
    grad_ptr,
    self_ptr,
    scale_ptr,
    zero_point_ptr,
    dX_ptr,
    dScale_vec_ptr,
    dZeroPoint_vec_ptr,
    numel,
    C,
    suffix,
    quant_min,
    quant_max,
    grad_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(self_ptr + offsets, mask=mask, other=0.0)

    channel = (offsets // suffix) % C
    scale = tl.load(scale_ptr + channel, mask=mask, other=0.0)
    zero_point = tl.load(zero_point_ptr + channel, mask=mask, other=0.0)

    # Rounded (and clamped) zero point, mirroring _get_rounded_zero_point.
    zp_rounded = _round_half_to_even(zero_point)
    zp_rounded = tl.minimum(tl.maximum(zp_rounded, quant_min), quant_max)

    inv_scale = 1.0 / scale

    xq = _round_half_to_even(x * inv_scale) + zp_rounded

    in_range = (xq >= quant_min) & (xq <= quant_max)
    dX = grad * in_range.to(tl.float32)

    xq_clamped = tl.minimum(tl.maximum(xq, quant_min), quant_max)
    xfq = (xq_clamped - zp_rounded) * scale

    sat = (xq < quant_min) | (xq > quant_max)
    sat_low = xq < quant_min
    dscale_small = (quant_min - zp_rounded).to(tl.float32)
    dscale_big = (quant_max - zp_rounded).to(tl.float32)

    dScale_sat = tl.where(sat_low, grad * dscale_small, grad * dscale_big) * grad_factor
    dScale_inrange = grad * (xfq - x) * inv_scale * grad_factor
    dScale_vec = tl.where(sat, dScale_sat, dScale_inrange)

    dZeroPoint_vec = tl.where(sat, grad * (-1.0) * scale * grad_factor, 0.0)

    tl.store(dX_ptr + offsets, dX, mask=mask)
    tl.store(dScale_vec_ptr + offsets, dScale_vec, mask=mask)
    tl.store(dZeroPoint_vec_ptr + offsets, dZeroPoint_vec, mask=mask)


def _fake_quantize_learnable_per_channel_affine_backward(
    grad,
    self,
    scale,
    zero_point,
    axis,
    quant_min,
    quant_max,
    grad_factor=1.0,
):
    logger.debug("GEMS FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE_BACKWARD")

    # The aten op requires all tensors to be float32; enforce it here too.
    assert grad.dtype == torch.float32, "grad must be float32"
    assert self.dtype == torch.float32, "self must be float32"
    assert scale.dtype == torch.float32, "scale must be float32"
    assert zero_point.dtype == torch.float32, "zero_point must be float32"
    grad = grad.contiguous()
    self = self.contiguous()
    scale = scale.contiguous()
    zero_point = zero_point.contiguous()

    numel = self.numel()
    dX = torch.empty_like(self)
    if numel == 0:
        dScale = torch.zeros_like(scale)
        dZeroPoint = torch.zeros_like(zero_point)
        return dX, dScale, dZeroPoint

    dScale_vec = torch.empty_like(self)
    dZeroPoint_vec = torch.empty_like(self)

    C = self.shape[axis]
    suffix = 1
    for s in self.shape[axis + 1 :]:
        suffix *= s
    suffix = max(int(suffix), 1)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    _fake_quantize_learnable_per_channel_affine_backward_kernel[grid](
        grad,
        self,
        scale,
        zero_point,
        dX,
        dScale_vec,
        dZeroPoint_vec,
        numel,
        C,
        suffix,
        float(quant_min),
        float(quant_max),
        float(grad_factor),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reduce dScale_vec / dZeroPoint_vec over all axes except `axis`.
    ndim = self.dim()
    reduce_axes = tuple(i for i in range(ndim) if i != axis)
    dScale = dScale_vec.sum(dim=reduce_axes)
    dZeroPoint = dZeroPoint_vec.sum(dim=reduce_axes)

    return dX, dScale, dZeroPoint
