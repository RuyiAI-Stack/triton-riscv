"""Fake-quantize with learnable per-channel affine qparams (forward).

Port notes (triton-riscv, vs the upstream flag_gems file):
- upstream's pointwise_dynamic broadcast is replaced by explicit channel
  indexing (``ch = (offset // stride_axis) % C`` over a contiguous view);
  the per-element arithmetic is upstream's verbatim fp32 pipeline
  ``y = (clamp(rint(x * rcp_rn(scale)) + zp, qmin, qmax) - zp) * scale``
  with the wrapper rounding/clamping the zero point like aten.
- ``nearbyint``/``rcp_rn`` do not lower here; replaced by the floor-based
  round-half-to-even helper and ``tl.math.div_rn`` (same rounding modes).
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
def fake_quantize_learnable_per_channel_affine_forward_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    zp_ptr,
    n_elements,
    C,
    stride_axis,
    qmin_val,
    qmax_val,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    ch = (offsets // stride_axis) % C
    s = tl.load(scale_ptr + ch, mask=mask, other=1.0).to(tl.float32)
    zp = tl.load(zp_ptr + ch, mask=mask, other=0.0).to(tl.float32)
    inv = tl.math.div_rn(1.0, s)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_to_even(x * inv) + zp
    q = tl.minimum(tl.maximum(q, qmin_val), qmax_val)
    y = (q - zp) * s
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def _reshape_per_channel_params(param, axis, ndim):
    shape = [1] * ndim
    shape[axis] = param.shape[0]
    return param.reshape(shape)


def _prepare_zero_point(zero_point, quant_min, quant_max):
    """Round, clamp and cast the learnable zero point to int, mirroring the
    behaviour of ``_get_rounded_zero_point`` followed by ``.to(kInt)`` in the
    PyTorch native implementation."""
    return zero_point.round().clamp_(quant_min, quant_max).to(torch.int32)


def _fake_quantize_learnable_per_channel_affine(
    self, scale, zero_point, axis, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS FAKE_QUANTIZE_LEARNABLE_PER_CHANNEL_AFFINE")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(
            f"unsupported dtype {self.dtype}; only floating-point dtypes are supported"
        )
    if axis < 0:
        axis = self.dim() + axis

    x = self.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    zero_point_rounded = _prepare_zero_point(zero_point, quant_min, quant_max)
    scale_c = scale.contiguous()
    zp_c = zero_point_rounded.contiguous()
    C = x.shape[axis]
    stride_axis = 1
    for d in range(axis + 1, x.dim()):
        stride_axis *= x.shape[d]

    qmin = (
        int(quant_min)
        if not isinstance(quant_min, torch.Tensor)
        else int(quant_min.item())
    )
    qmax = (
        int(quant_max)
        if not isinstance(quant_max, torch.Tensor)
        else int(quant_max.item())
    )

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    fake_quantize_learnable_per_channel_affine_forward_kernel[grid](
        x,
        out,
        scale_c,
        zp_c,
        n,
        C,
        stride_axis,
        float(qmin),
        float(qmax),
        BLOCK=BLOCK,
    )
    return out
