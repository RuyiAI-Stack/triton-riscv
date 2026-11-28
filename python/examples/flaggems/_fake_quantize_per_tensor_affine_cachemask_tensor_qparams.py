"""``_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`` port.

Port notes (triton-riscv, vs the upstream flag_gems file):
- The magic-add rounding does not lower reliably here; replaced by the
  floor-based round-half-to-even helper shared with the other quant ports
  (same rounding mode as ATen's ``std::nearbyint``).
- The grid-stride while-loops are one masked tile per program (house
  style); per-element arithmetic is upstream's verbatim, notably the
  rounding happens BEFORE the zero point is added, the NaN gate lands on
  quant_min, and the disabled path is a bit-exact copy of the raw load.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_MAXD = 8


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
def _fq_quantize(x, inv, zp, scale, qmin_f, qmax_f):
    # Rounding happens *before* zero_point is added (ATen's observable order
    # on exact ties).
    v = x * inv
    r = _round_half_to_even(v)
    q = r + zp
    in_range = (q >= qmin_f) & (q <= qmax_f)
    qc = tl.minimum(tl.maximum(q, qmin_f), qmax_f)
    # ATen's clamp does not propagate NaN: a NaN input lands on QUANT_MIN.
    qc = tl.where(x != x, qmin_f, qc)
    return (qc - zp) * scale, in_range


@triton.jit
def _fq_flat_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    enabled_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    QUANT_MIN,
    QUANT_MAX,
    BLOCK: tl.constexpr,
):
    scale = tl.load(scale_ptr).to(tl.float32)
    zp = tl.load(zp_ptr).to(tl.float32)
    enabled = tl.load(enabled_ptr) != 0
    not_enabled = ~enabled
    inv = 1.0 / scale
    qmin_f = QUANT_MIN.to(tl.float32)
    qmax_f = QUANT_MAX.to(tl.float32)

    out_ty = out_ptr.dtype.element_ty
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements

    xr = tl.load(x_ptr + offs, mask=m, other=0.0)
    x = xr.to(tl.float32)
    quantized, in_range = _fq_quantize(x, inv, zp, scale, qmin_f, qmax_f)
    # The disabled path is a bit-exact copy: routing the fp32 value back
    # through the output dtype would canonicalize a NaN payload.
    tl.store(out_ptr + offs, tl.where(enabled, quantized.to(out_ty), xr), mask=m)
    tl.store(mask_ptr + offs, in_range | not_enabled, mask=m)


@triton.jit
def _fq_strided_kernel(
    x_ptr,
    scale_ptr,
    zp_ptr,
    enabled_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    QUANT_MIN,
    QUANT_MAX,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    scale = tl.load(scale_ptr).to(tl.float32)
    zp = tl.load(zp_ptr).to(tl.float32)
    enabled = tl.load(enabled_ptr) != 0
    not_enabled = ~enabled
    inv = 1.0 / scale
    qmin_f = QUANT_MIN.to(tl.float32)
    qmax_f = QUANT_MAX.to(tl.float32)

    out_ty = out_ptr.dtype.element_ty
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements

    idx = offs
    off = offs * 0
    if NDIM > 0:
        off += (idx % s0) * t0
        idx = idx // s0
    if NDIM > 1:
        off += (idx % s1) * t1
        idx = idx // s1
    if NDIM > 2:
        off += (idx % s2) * t2
        idx = idx // s2
    if NDIM > 3:
        off += (idx % s3) * t3
        idx = idx // s3
    if NDIM > 4:
        off += (idx % s4) * t4
        idx = idx // s4
    if NDIM > 5:
        off += (idx % s5) * t5
        idx = idx // s5
    if NDIM > 6:
        off += (idx % s6) * t6
        idx = idx // s6
    if NDIM > 7:
        off += (idx % s7) * t7
        idx = idx // s7

    xr = tl.load(x_ptr + off, mask=m, other=0.0)
    x = xr.to(tl.float32)
    quantized, in_range = _fq_quantize(x, inv, zp, scale, qmin_f, qmax_f)
    tl.store(out_ptr + offs, tl.where(enabled, quantized.to(out_ty), xr), mask=m)
    tl.store(mask_ptr + offs, in_range | not_enabled, mask=m)


def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    self, scale, zero_point, fake_quant_enabled, quant_min, quant_max
):
    logger.debug("GEMS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK_TENSOR_QPARAMS")

    if fake_quant_enabled.dtype == torch.int64:
        # A 64-bit scalar load does not compile on this backend; int64 is
        # stored as one int32 word per element, so reinterpreting it as int32
        # yields the 0/1 flag aten passes (only ever tested != 0).
        fake_quant_enabled = fake_quant_enabled.reshape(1).view(torch.int32)

    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    mask = torch.empty(self.shape, dtype=torch.bool, device=self.device)

    n = self.numel()
    if n == 0:
        return out, mask

    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    if self.is_contiguous():
        _fq_flat_kernel[grid](
            self,
            scale,
            zero_point,
            fake_quant_enabled,
            out,
            mask,
            n,
            quant_min,
            quant_max,
            BLOCK=BLOCK,
        )
    else:
        rank = self.dim()
        # Logical row-major order decomposes from the innermost dimension, so
        # the per-dim shape/stride pairs are passed last-dim-first.
        shp = list(reversed(self.shape)) + [1] * (_MAXD - rank)
        st = list(reversed(self.stride())) + [0] * (_MAXD - rank)
        _fq_strided_kernel[grid](
            self,
            scale,
            zero_point,
            fake_quant_enabled,
            out,
            mask,
            n,
            shp[0],
            shp[1],
            shp[2],
            shp[3],
            shp[4],
            shp[5],
            shp[6],
            shp[7],
            st[0],
            st[1],
            st[2],
            st[3],
            st[4],
            st[5],
            st[6],
            st[7],
            quant_min,
            quant_max,
            NDIM=rank,
            BLOCK=BLOCK,
        )
    return out, mask
