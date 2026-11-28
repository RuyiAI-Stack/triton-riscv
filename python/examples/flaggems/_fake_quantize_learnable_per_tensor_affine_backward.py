"""Fake-quantize with learnable per-tensor affine qparams (backward).

Port notes (triton-riscv, vs the upstream flag_gems file):
- ``_rint_even``'s magic-add rounding misrounds negative fractional values on
  this backend (it behaves like round-to-multiple-of-0.5), and it feeds the
  ``dy * (r - q)`` scale-gradient term directly; replaced by the floor-based
  round-half-to-even helper shared with the other quantize ports.
- ``tl.fdiv(x, s, ieee_rounding=True)`` does not lower here; ``q = x / s``
  uses ``tl.math.div_rn`` instead (probed bit-exact against torch's fp32
  division, which is what the aten reference computes).
- The grid-stride ``range(pid, num_tiles, nprog)`` loop is rewritten as
  ``range(0, my_tiles)`` (same tiles per program, two-argument form only)
  because the two-argument dynamic range is the form verified to lower on
  this backend.
- Plain-float ``scale`` / ``zero_point`` are tensorized *before* the
  floating-point validation, so the pinned ValueError texts still fire from
  the original checks while scalar arguments stay accepted.
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
def _fq_tiny_kernel(
    grad_ptr,
    x_ptr,
    scale_ptr,
    zp_ptr,
    gs_ptr,
    out_s_ptr,
    out_z_ptr,
    n,
    num_tiles,
    qmin,
    qmax,
    gf,
    BLOCK: tl.constexpr,
):
    s = tl.load(scale_ptr).to(tl.float32)
    z = tl.load(zp_ptr).to(tl.float32)
    qminf = qmin.to(tl.float32)
    qmaxf = qmax.to(tl.float32)
    zr = tl.minimum(tl.maximum(_round_half_to_even(z), qminf), qmaxf)
    lo = qminf - zr
    hi = qmaxf - zr

    acc_ds = 0.0
    acc_dz = 0.0
    for tile in range(0, num_tiles):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        dy = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q = tl.math.div_rn(x, s)
        r = _round_half_to_even(q)
        u = r + zr
        inr = (u >= qminf) & (u <= qmaxf)
        tl.store(gs_ptr + offs, tl.where(inr, dy, 0.0), mask=mask)
        ds_v = tl.where(inr, dy * (r - q), tl.where(u < qminf, lo * dy, hi * dy))
        dz_v = tl.where(inr, 0.0, -dy * s)
        acc_ds += tl.sum(ds_v, axis=0)
        acc_dz += tl.sum(dz_v, axis=0)

    tl.store(out_s_ptr, acc_ds * gf)
    tl.store(out_z_ptr, acc_dz * gf)


@triton.jit
def _fake_quantize_learnable_per_tensor_affine_backward_kernel(
    grad_ptr,
    x_ptr,
    scale_ptr,
    zp_ptr,
    gs_ptr,
    pds_ptr,
    pdz_ptr,
    n,
    qmin,
    qmax,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_tiles = (n + (BLOCK - 1)) // BLOCK

    s = tl.load(scale_ptr).to(tl.float32)
    z = tl.load(zp_ptr).to(tl.float32)
    qminf = qmin.to(tl.float32)
    qmaxf = qmax.to(tl.float32)
    zr = tl.minimum(tl.maximum(_round_half_to_even(z), qminf), qmaxf)
    lo = qminf - zr
    hi = qmaxf - zr

    acc_ds = tl.zeros([BLOCK], dtype=tl.float32)
    acc_dz = tl.zeros([BLOCK], dtype=tl.float32)

    my_tiles = (num_tiles - pid + nprog - 1) // nprog
    for t in range(0, my_tiles):
        tile = pid + t * nprog
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        dy = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        q = tl.math.div_rn(x, s)
        r = _round_half_to_even(q)
        u = r + zr
        inr = (u >= qminf) & (u <= qmaxf)

        tl.store(gs_ptr + offs, tl.where(inr, dy, 0.0), mask=mask)

        acc_ds += tl.where(inr, dy * (r - q), tl.where(u < qminf, lo * dy, hi * dy))
        acc_dz += tl.where(inr, 0.0, -dy * s)

    tl.store(pds_ptr + pid, tl.sum(acc_ds, axis=0))
    tl.store(pdz_ptr + pid, tl.sum(acc_dz, axis=0))


@triton.jit
def _reduce_kernel(pds_ptr, pdz_ptr, out_s_ptr, out_z_ptr, g, gf, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < g
    ds = tl.load(pds_ptr + offs, mask=mask, other=0.0)
    dz = tl.load(pdz_ptr + offs, mask=mask, other=0.0)
    tl.store(out_s_ptr, tl.sum(ds, axis=0) * gf)
    tl.store(out_z_ptr, tl.sum(dz, axis=0) * gf)


_G = 256
_WORK_PER_PROG = 262144
_TINY_BLOCK = 4096
_TINY_LIMIT = 2 * _TINY_BLOCK
# Static launch geometry instead of a triton.autotune sweep (whose per-call
# dispatch is not free).  Same constants as upstream.
_MAIN_BLOCK = 32768
_MAIN_WARPS = 4


def _fake_quantize_learnable_per_tensor_affine_backward(
    grad, self, scale, zero_point, quant_min, quant_max, grad_factor=1.0
):
    logger.debug("GEMS FAKE_QUANTIZE_LEARNABLE_PER_TENSOR_AFFINE_BACKWARD")

    # Scalar qparams arrive as Python floats on this suite's entries; aten and
    # upstream receive shape-(1,) float tensors.  Tensorize first so the
    # floating-point validation below still sees tensors.
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor([float(scale)], dtype=torch.float32)
    if not isinstance(zero_point, torch.Tensor):
        zero_point = torch.as_tensor([float(zero_point)], dtype=torch.float32)

    if not (
        torch.is_floating_point(grad)
        and torch.is_floating_point(self)
        and torch.is_floating_point(scale)
        and torch.is_floating_point(zero_point)
    ):
        raise ValueError("All inputs must be floating-point tensors.")

    # The ATen reference rejects float16 gradients ("Expected dY_.scalar_type()
    # == Float"); only float32 and bfloat16 inputs are supported.
    if self.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Unsupported dtype {self.dtype}; expected float32 or bfloat16."
        )

    # The ATen reference implementation always returns float32 gradients
    # (it rejects float16 grads and promotes bfloat16 grads to float32), so we
    # match that contract: all three output gradients are float32.
    grad_self = torch.empty_like(self, dtype=torch.float32)
    grad_scale = torch.empty_like(scale, dtype=torch.float32)
    grad_zero_point = torch.empty_like(zero_point, dtype=torch.float32)

    n = self.numel()
    gf = float(grad_factor)
    qmin = int(quant_min)
    qmax = int(quant_max)

    if n <= _TINY_LIMIT:
        _fq_tiny_kernel[(1,)](
            grad,
            self,
            scale,
            zero_point,
            grad_self,
            grad_scale,
            grad_zero_point,
            n,
            (_TINY_BLOCK + n - 1) // _TINY_BLOCK if n > 0 else 0,
            qmin,
            qmax,
            gf,
            BLOCK=_TINY_BLOCK,
            num_warps=4,
        )
        return grad_self, grad_scale, grad_zero_point

    pds = torch.empty((_G,), dtype=torch.float32, device=self.device)
    pdz = torch.empty((_G,), dtype=torch.float32, device=self.device)

    g = min(_G, max(1, (n + _WORK_PER_PROG - 1) // _WORK_PER_PROG))

    _fake_quantize_learnable_per_tensor_affine_backward_kernel[(g,)](
        grad,
        self,
        scale,
        zero_point,
        grad_self,
        pds,
        pdz,
        n,
        qmin,
        qmax,
        BLOCK=_MAIN_BLOCK,
        num_warps=_MAIN_WARPS,
    )

    _reduce_kernel[(1,)](
        pds, pdz, grad_scale, grad_zero_point, g, gf, BLOCK=_G, num_warps=4
    )

    return grad_self, grad_scale, grad_zero_point
