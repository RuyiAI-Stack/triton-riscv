"""Fake-quantize with learnable per-tensor affine qparams (forward).

Port notes (triton-riscv, vs the upstream flag_gems file):
- ``libdevice.nearbyint`` and the zero-point magic-add rounding do not lower /
  misround on this backend; both are replaced by the floor-based
  round-half-to-even helper shared with the other quantize ports (same
  rounding mode as ``std::nearbyint``).
- The grid-stride loop with a runtime single-sided tail ``if`` is replaced by
  one masked tile per program -- the structure the fake_quantize_per_tensor
  port uses on this backend; the per-element arithmetic is upstream's
  verbatim: ``y = (clamp(rint(x * div_rn(1,s)) + zp, qmin, qmax) - zp) * s``
  with the zero point rounded to nearest-even and clamped like aten.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# One program covers the tensor up to 128K elements (262K above 256 programs);
# grid.x stays below 65535 for any real tensor.
_BLOCK_MAX = 131072
_BLOCK_MAX_HUGE = 262144
_MAX_GRID_X = 65535


def _pick_block(n):
    block = 1 << max(10, (n - 1).bit_length())
    if block > _BLOCK_MAX:
        block = _BLOCK_MAX
    if (n + block - 1) // block > 256:
        block = _BLOCK_MAX_HUGE
    while (n + block - 1) // block > _MAX_GRID_X:
        block *= 2
    return block


def _pick_warps(block, nblocks):
    if block <= 16384:
        return 2
    if block <= 32768:
        return 4
    if nblocks <= 2:
        return 8
    if nblocks <= 4:
        return 4
    return 2


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
def _fake_quantize_learnable_per_tensor_affine_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    zp_ptr,
    n_elements,
    qmin_val,
    qmax_val,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    s = tl.load(scale_ptr).to(tl.float32)
    zpf = tl.load(zp_ptr).to(tl.float32)
    # aten rounds the zero point to nearest-even, then clamps it.
    zpf = _round_half_to_even(zpf)
    zpf = tl.minimum(tl.maximum(zpf, qmin_val), qmax_val)
    # aten computes x * (1.0f / scale) with a correctly-rounded reciprocal;
    # the plain Triton '/' is approximate and flips values near a .5 tie.
    inv = tl.math.div_rn(1.0, s)

    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q = _round_half_to_even(x * inv)
    q = q + zpf
    q = tl.minimum(tl.maximum(q, qmin_val), qmax_val)
    y = (q - zpf) * s
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def _fake_quantize_learnable_per_tensor_affine(
    self, scale, zero_point, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE")

    x = self
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor([float(scale)], dtype=torch.float32, device=x.device)
    if not isinstance(zero_point, torch.Tensor):
        zero_point = torch.as_tensor(
            [float(zero_point)], dtype=torch.float32, device=x.device
        )
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

    block = _pick_block(n)
    nblocks = (n + block - 1) // block
    _fake_quantize_learnable_per_tensor_affine_kernel[(nblocks,)](
        x,
        out,
        scale,
        zero_point,
        n,
        float(qmin),
        float(qmax),
        BLOCK=block,
        num_warps=_pick_warps(block, nblocks),
    )
    return out
